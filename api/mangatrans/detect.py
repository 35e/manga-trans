"""Finding the lettering on a page, and the balloons it was written in.

:class:`Regions` boxes both, told apart by class. :class:`Letters` is kept for
its segmentation head alone — the per-pixel ink map. See DOCS.md.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .geometry import Box

# --- comic-text-detector, for the ink mask ----------------------------------

MODEL_NAME = "comictextdetector.pt.onnx"
MODEL_URL = (
    "https://github.com/zyddnys/manga-image-translator/releases/download/"
    "beta-0.2.1/" + MODEL_NAME
)
MODEL_ENV = "MANGA_TRANS_MODEL"
MODEL_DIRS = ("/opt/models", "~/.cache/manga-trans")

INPUT_SIZE = 1024

# The segmentation head answers per pixel, between 0 and 1.
SEG_THRESHOLD = 0.5

# How far the ink mask is grown to cover the halo around a hidden letter.
# Measured in the *detector's* pixels, not the page's: held in page pixels the
# same value is three canvas pixels of allowance on a small page and barely one
# on a 300 dpi scan. Anything else measured against the mask is in these units.
GROW = 4
GROW_MAX = 64

# --- comic-text-and-bubble-detector, for the boxes --------------------------

REGIONS_REPO = "ogkalu/comic-text-and-bubble-detector"
REGIONS_FILE = "detector_int8.onnx"
REGIONS_ENV = "MANGA_TRANS_REGIONS"

# A plain resize to a square, rescaled to 0..1, no normalisation — straight from
# the model's own preprocessor_config.json.
REGIONS_SIZE = 640

BUBBLE, TEXT_BUBBLE, TEXT_FREE = 0, 1, 2

SPEECH, FREE = "speech", "free"
KINDS = {TEXT_BUBBLE: SPEECH, TEXT_FREE: FREE}

# The floor of what the dashboard is offered, not the line between sure and
# unsure — that is UNSURE (0.8), and it is the front end's.
REGIONS_CONF = 0.35

# Two boxes covering this much of the smaller of them are one thing found twice.
DUPLICATE = 0.75

# A margin left around every block, as a share of its shorter side: the head
# boxes lettering tightly and sometimes inside it, clipping the edge of a glyph.
PAD = 0.04
PAD_MIN = 2


@dataclass(frozen=True)
class Block:
    """One piece of lettering the detector found."""

    box: Box
    confidence: float = 1.0
    # Empty where nothing said: a block drawn by hand, or asked about on its own.
    kind: str = ""


def model_path(explicit: str | None = None) -> Path:
    """The ink model's weights: where it already is, else where it should be put."""
    chosen = explicit or os.environ.get(MODEL_ENV)
    if chosen:
        return Path(chosen).expanduser()
    candidates = [Path(directory).expanduser() / MODEL_NAME for directory in MODEL_DIRS]
    return next((path for path in candidates if path.is_file()), candidates[-1])


# The wait between tries is the half that matters: whatever drops one of these
# drops the next few as well, so tries in a row all land in the same bad seconds.
TRIES = 6
BACKOFF = 8

# The default `Python-urllib/3.x` is refused out of hand by GitHub's release
# assets, which close the connection rather than saying so.
AGENT = "Mozilla/5.0 (compatible; manga-trans)"


def ensure_model(explicit: str | None = None) -> Path:
    """Download the ink model's weights (~95 MB) unless they are already there."""
    path = model_path(explicit)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")

    asked = urllib.request.Request(MODEL_URL, headers={"User-Agent": AGENT})
    for attempt in range(1, TRIES + 1):
        print(f"mangatrans: downloading {MODEL_URL} ({attempt}/{TRIES})")
        try:
            with urllib.request.urlopen(asked) as answer, partial.open("wb") as file:
                shutil.copyfileobj(answer, file)
            break
        except Exception as exc:  # noqa: BLE001 — every failure here is the transfer
            partial.unlink(missing_ok=True)
            if attempt == TRIES:
                raise
            wait = BACKOFF * attempt
            print(f"mangatrans: {type(exc).__name__}: {exc}; again in {wait}s")
            time.sleep(wait)

    partial.replace(path)
    print(f"mangatrans: saved {path} ({path.stat().st_size / 1e6:.0f} MB)")
    return path


def ensure_regions(explicit: str | None = None) -> str:
    """Fetch the region detector (~44 MB) into the Hugging Face cache."""
    chosen = explicit or os.environ.get(REGIONS_ENV)
    if chosen:
        return chosen

    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(REGIONS_REPO, REGIONS_FILE, local_files_only=True)
    except Exception:  # noqa: BLE001 — not cached, so go and get it
        print(f"mangatrans: downloading {REGIONS_REPO}/{REGIONS_FILE}")
        path = hf_hub_download(REGIONS_REPO, REGIONS_FILE)
        print(f"mangatrans: saved {path}")
        return path


def letterbox(image, size: int = INPUT_SIZE):
    """Scale into a square canvas, padding bottom and right as the model expects."""
    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    new_w, new_h = round(width * scale), round(height * scale)
    if (width, height) != (new_w, new_h):
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = size - new_w, size - new_h
    canvas = cv2.copyMakeBorder(
        image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    return canvas, pad_w, pad_h


def page_mask(
    seg, width: int, height: int, pad_w: int, pad_h: int, grow: int = GROW
) -> np.ndarray:
    """The model's padded square of per-pixel text as a mask the page's size."""
    seg_h, seg_w = seg.shape[:2]
    kept = seg[
        : max(1, round(seg_h * (INPUT_SIZE - pad_h) / INPUT_SIZE)),
        : max(1, round(seg_w * (INPUT_SIZE - pad_w) / INPUT_SIZE)),
    ]
    # Stretched while still a probability, so the edge of a letter lands where it
    # should before anything is decided about it.
    full = cv2.resize(kept, (width, height), interpolation=cv2.INTER_LINEAR)
    mask = ((full > SEG_THRESHOLD) * 255).astype(np.uint8)

    # How many page pixels one canvas pixel became — see :data:`GROW`.
    spread = round(grow * max(width, height) / INPUT_SIZE)
    if spread > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (spread * 2 + 1, spread * 2 + 1)
        )
        mask = cv2.dilate(mask, kernel)
    return mask


def padded(box: Box, width: int, height: int) -> Box:
    """One block with its margin, still on the page."""
    by = max(PAD_MIN, round(PAD * min(box.w, box.h)))
    return box.grown(by).clipped(width, height)


def suppressed(blocks: list[Block]) -> list[Block]:
    """The same thing found twice, thinned down to the surest of them."""
    kept: list[Block] = []
    for block in sorted(blocks, key=lambda block: -block.confidence):
        if not any(block.box.covers(other.box) >= DUPLICATE for other in kept):
            kept.append(block)
    return kept


def decode(
    labels: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    width: int,
    height: int,
):
    """The model's rows to (class, box, score), in the page's own pixels.

    It answers with a fixed 300 queries whether or not it found that many
    things, so the score is what says which are real.
    """
    found = []
    for at in np.flatnonzero(scores >= REGIONS_CONF):
        x0, y0, x1, y1 = boxes[at]
        box = Box(
            int(round(float(x0))),
            int(round(float(y0))),
            int(round(float(x1))),
            int(round(float(y1))),
        ).clipped(width, height)
        if box.w > 1 and box.h > 1:
            found.append((int(labels[at]), box, float(scores[at])))
    return found


class Regions:
    """The region detector: where the balloons are, and the lettering in them.

    One page at a time — an onnxruntime session is not reentrant — and the last
    page's pass is kept, because the same page goes through more than once as a
    matter of course.
    """

    def __init__(self, weights: str | Path | None = None) -> None:
        import onnxruntime as ort

        from .read import quieted

        path = ensure_regions(str(weights) if weights else None)
        with quieted():
            self.session = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
        self.answers = [tensor.name for tensor in self.session.get_outputs()]
        self._lock = threading.Lock()
        self._last: tuple[bytes, tuple] | None = None

    def run(self, image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One pass: the classes, the boxes and the scores, in the page's pixels.

        ``orig_target_sizes`` is **(width, height)** — measured, not assumed: the
        other way round returns the same balloons transposed.
        """
        page = np.ascontiguousarray(image)
        key = hashlib.blake2b(page, digest_size=16, key=b"mangatrans").digest()

        with self._lock:
            if self._last is not None and self._last[0] == key:
                return self._last[1]

            height, width = page.shape[:2]
            square = cv2.resize(
                page, (REGIONS_SIZE, REGIONS_SIZE), interpolation=cv2.INTER_LINEAR
            )
            blob = (square.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
            outputs = self.session.run(
                None,
                {
                    "images": blob,
                    "orig_target_sizes": np.array([[width, height]], dtype=np.int64),
                },
            )
            got = dict(zip(self.answers, outputs))
            answer = (got["labels"][0], got["boxes"][0], got["scores"][0])
            self._last = (key, answer)
            return answer

    def __call__(self, image, rtl: bool = True) -> tuple[list[Block], list[Box]]:
        """Every block of lettering on one RGB page array, and every balloon.

        ``rtl`` is the caller's rather than measured: the same layout of balloons
        is read both ways round by different languages.
        """
        height, width = image.shape[:2]
        labels, boxes, scores = self.run(image)
        found = decode(labels, boxes, scores, width, height)

        # Per class: a balloon and the text filling it cover each other almost
        # entirely and are not two findings of one thing.
        balloons = suppressed(
            [Block(box, score) for kind, box, score in found if kind == BUBBLE]
        )
        # Thinned on the tight boxes and padded after, never the other way round:
        # a margin put on first can make two neighbours look like one finding.
        blocks = [
            Block(padded(block.box, width, height), block.confidence, block.kind)
            for block in suppressed(
                [
                    Block(box, score, KINDS[kind])
                    for kind, box, score in found
                    if kind in KINDS
                ]
            )
        ]

        # `lib/order.ts` sorts by the same key and must go on agreeing.
        across = (lambda box: -box.x1) if rtl else (lambda box: box.x0)
        blocks.sort(key=lambda block: (block.box.y0, across(block.box)))
        return blocks, [balloon.box for balloon in balloons]


class Letters:
    """comic-text-detector's segmentation head: where the ink is, pixel by pixel.

    One page at a time: an OpenCV net is not reentrant. The last page's pass is
    kept — it is seconds where everything downstream of it is a millisecond.
    """

    def __init__(self, weights: str | Path | None = None) -> None:
        path = ensure_model(str(weights) if weights else None)
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self._lock = threading.Lock()
        self._last: tuple[bytes, tuple] | None = None

    def run(self, image):
        """One pass: the per-pixel text map and the padding put on to get it.

        Some OpenCV builds return the outputs in a different order than they were
        asked for, so they are told apart by shape rather than by position.
        """
        page = np.ascontiguousarray(image)
        key = hashlib.blake2b(page, digest_size=16, key=b"mangatrans").digest()

        with self._lock:
            if self._last is not None and self._last[0] == key:
                return self._last[1]

            canvas, pad_w, pad_h = letterbox(page)
            blob = cv2.dnn.blobFromImage(
                canvas, scalefactor=1 / 255.0, size=(INPUT_SIZE, INPUT_SIZE)
            )
            self.net.setInput(blob)
            outputs = self.net.forward(("blk", "seg", "det"))

            seg = next(
                output
                for output in outputs
                if output.ndim == 4 and output.shape[1] == 1
            )
            answer = (seg[0, 0], pad_w, pad_h)
            self._last = (key, answer)
            return answer

    def __call__(self, image, grow: int = GROW) -> np.ndarray:
        """A mask of the lettering itself, pixel by pixel, the page's size."""
        height, width = image.shape[:2]
        seg, pad_w, pad_h = self.run(image)
        return page_mask(seg, width, height, pad_w, pad_h, grow)
