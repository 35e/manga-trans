"""Text fragment detection (EasyOCR's CRAFT) and the memory budget it runs in."""

from __future__ import annotations

import math
import os
from pathlib import Path

from .geometry import Box

CANVAS_MAX = 2560  # EasyOCR's own default; there is nothing to gain above it
CANVAS_MIN = 640  # below this the detector stops finding dialogue
# Free memory drifts between runs; rounding the canvas down to a coarse step
# keeps the same page detecting at the same resolution anyway.
CANVAS_STEP = 128

# CRAFT's peak memory grows linearly with the canvas it runs on. Measured on
# CPU (podman, python:3.12-slim): ~1.4 kB per canvas pixel on top of a fixed
# ~400 MB for the interpreter, torch and the model weights. A 2894x4093 page at
# the default canvas of 2560 therefore wants ~4.4 GB and is simply killed on a
# 4 GB machine, which is what ``auto`` exists to prevent.
DETECT_BYTES_PER_PIXEL = 1400
DETECT_BASE_BYTES = 400 * 1024**2
# An OOM kill arrives as SIGKILL and cannot be caught, so stay well under.
MEMORY_SAFETY = 0.8


def available_memory_bytes() -> int | None:
    """Memory this process can realistically use, or ``None`` if unknown.

    In a container the cgroup limit is what actually kills the process; outside
    one, MemAvailable is the honest number. Whichever is smaller wins.
    """
    limits: list[int] = []
    for path, unlimited in (
        ("/sys/fs/cgroup/memory.max", "max"),  # cgroup v2
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", None),  # cgroup v1
    ):
        try:
            raw = Path(path).read_text().strip()
            value = int(raw) if raw != unlimited else 0
        except (OSError, ValueError):
            continue
        # cgroup v1 spells "unlimited" as a huge number rather than a word.
        if 0 < value < 1 << 60:
            limits.append(value)

    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                limits.append(int(line.split()[1]) * 1024)
                break
    except (OSError, IndexError, ValueError):
        pass

    if not limits:  # macOS and anything else without /proc
        try:
            limits.append(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError, AttributeError):
            return None
    return min(limits)


def auto_canvas_size(
    width: int,
    height: int,
    budget_bytes: int | None,
    ceiling: int = CANVAS_MAX,
    mag_ratio: float = 1.0,
) -> int:
    """Largest detection canvas that fits ``budget_bytes``, in pixels.

    EasyOCR scales the image's long side to ``mag_ratio`` times itself and then
    clamps that to the canvas size, padding both sides to a multiple of 32; the
    canvas holds about ``canvas**2 * short_side / long_side`` pixels.

    The clamp is why the canvas has to know about ``mag_ratio``. A page is never
    magnified for its own sake - at ``mag_ratio`` 1 the ceiling is the page's own
    long side - but when a magnification has been asked for, holding the canvas
    down to the original size would silently cancel it, and the small furigana it
    was asked for would go on being missed.
    """
    long_side, short_side = max(width, height), max(1, min(width, height))
    wanted = int(long_side * max(1.0, mag_ratio))
    ceiling = min(ceiling, wanted)
    if budget_bytes is None:
        return max(CANVAS_MIN, ceiling)

    usable = budget_bytes * MEMORY_SAFETY - DETECT_BASE_BYTES
    if usable <= 0:
        return CANVAS_MIN
    canvas = int(math.sqrt(usable / DETECT_BYTES_PER_PIXEL * long_side / short_side))
    canvas -= canvas % CANVAS_STEP
    return max(CANVAS_MIN, min(ceiling, canvas))


def is_memory_error(exc: BaseException) -> bool:
    """Does this exception mean "ran out of memory"?

    torch reports a failed CPU allocation as a plain ``RuntimeError``, so the
    message is all there is to go on.
    """
    if isinstance(exc, MemoryError):
        return True
    text = str(exc).lower()
    return any(
        s in text
        for s in ("out of memory", "can't allocate", "cannot allocate", "bad_alloc")
    )


def build_detector(gpu: bool, detect_network: str, verbose: bool):
    try:
        import easyocr  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise SystemExit(
            "easyocr is required for text detection: pip install easyocr"
        ) from exc

    kwargs = dict(gpu=gpu, detect_network=detect_network, verbose=verbose)
    try:
        # The recognition model is never used - manga-ocr does the reading.
        return easyocr.Reader(["ja", "en"], recognizer=False, **kwargs)
    except TypeError:
        return easyocr.Reader(["ja", "en"], **kwargs)


def detect_fragments(
    reader, image_grey, args, canvas_size, log=lambda _msg: None
) -> list[Box]:
    """Return raw text fragment boxes, with EasyOCR's own line merging disabled.

    ``canvas_size`` is already budgeted for the machine; the retry loop only
    catches the allocation failures torch reports as exceptions. A hard OOM kill
    never gets here - that is what the budget is for.
    """
    height, width = image_grey.shape[:2]

    while True:
        try:
            horizontal_agg, free_agg = reader.detect(
                image_grey,
                min_size=args.min_size,
                text_threshold=args.text_threshold,
                low_text=args.low_text,
                link_threshold=args.link_threshold,
                canvas_size=canvas_size,
                mag_ratio=args.mag_ratio,
                # Zeroed so EasyOCR does not merge fragments into "lines" using
                # horizontal-text assumptions - our own grouping handles that.
                slope_ths=0.0,
                ycenter_ths=0.0,
                height_ths=0.0,
                width_ths=0.0,
                add_margin=0.0,
            )
            break
        except (MemoryError, RuntimeError) as exc:
            if canvas_size <= CANVAS_MIN or not is_memory_error(exc):
                raise
            canvas_size = max(CANVAS_MIN, canvas_size // 2)
            log(f"  detection ran out of memory, retrying at canvas {canvas_size}")

    boxes: list[Box] = []
    for x_min, x_max, y_min, y_max in horizontal_agg[0]:
        boxes.append(Box(int(x_min), int(y_min), int(x_max), int(y_max)))
    for poly in free_agg[0]:
        xs = [int(p[0]) for p in poly]
        ys = [int(p[1]) for p in poly]
        boxes.append(Box(min(xs), min(ys), max(xs), max(ys)))

    # Boxes come back in original-image coordinates and can overshoot the edges
    # by a pixel or two once scaled back up from the canvas.
    boxes = [b.clipped(width, height) for b in boxes]
    return [b for b in boxes if b.w > 0 and b.h > 0]


class CraftDetector:
    """CRAFT behind the common detector interface.

    Fragments only: CRAFT finds runs of glyphs and has no opinion about which of
    them form one utterance, so the grouping and the bubble-finding downstream
    have to reconstruct that from geometry. See :mod:`.detectors` for what a
    trained comic detector supplies instead.
    """

    name = "craft"

    def __init__(self, args, log=lambda _msg: None) -> None:
        self.args = args
        self.reader = build_detector(
            gpu=not args.cpu,
            detect_network=args.detect_network,
            verbose=not args.quiet,
        )

    def __call__(self, image, log=lambda _msg: None):
        import cv2  # noqa: PLC0415

        from .detectors import DetectionResult  # noqa: PLC0415

        args = self.args
        height, width = image.shape[:2]
        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        canvas_size = args.canvas_size
        if canvas_size is None:  # --canvas-size auto
            canvas_size = auto_canvas_size(
                width, height, available_memory_bytes(), mag_ratio=args.mag_ratio
            )
            if canvas_size < max(width, height) * max(1.0, args.mag_ratio):
                log(f"  {width}x{height} page, detecting at canvas {canvas_size}")

        fragments = detect_fragments(self.reader, grey, args, canvas_size, log=log)
        return DetectionResult(fragments=fragments)
