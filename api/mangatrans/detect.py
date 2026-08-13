"""Finding the lettering on a page, and the balloons it was written in.

Two models, because they answer two different questions.

:class:`Regions` is comic-text-and-bubble-detector, an RT-DETRv2 trained on manga,
webtoons, manhua and western comics. One pass gives both the balloons and the
lettering, told apart by class — which is what lets a translation be lettered into
the room it belongs in rather than into a rectangle guessed from the page. It runs
on onnxruntime.

:class:`Letters` is comic-text-detector, kept for its segmentation head alone: the
per-pixel map of where the ink is, which is what lets a clean hide the words and
leave the art they were drawn over. Its own block head is not used — boxing by
region beats boxing by lettering, and a region detector does not run two balloons
together the way a lettering detector does. It runs on OpenCV's ONNX backend, so
it needs neither torch nor onnxruntime.
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

# Letters have soft edges and the mask stops at the ink, so the mask is grown to
# cover the halo left ringing a hidden letter — screentone, JPEG ringing, the
# pale rim of an outlined letter. Four suits clean lettering; scans want more,
# which is why callers can say `grow`.
#
# Measured in the *detector's* pixels rather than the page's. The mask is worked
# out on a 1024-square canvas and stretched to the page, so its edge is only ever
# accurate to a canvas pixel — which is one page pixel on a small page and three
# and a half on an A4 scan at 300 dpi. Fixed in page pixels, the same `grow`
# covers three canvas pixels of slop on the one and barely one on the other, and
# a big scan comes back with the feet of its letters still on it: measured,
# 100% of the lettering covered at 1000x1400 against 94.5% at 2480x3508.
# In canvas pixels it means the same thing at any size the page was scanned at.
GROW = 4
GROW_MAX = 64

# --- comic-text-and-bubble-detector, for the boxes --------------------------

REGIONS_REPO = "ogkalu/comic-text-and-bubble-detector"
REGIONS_FILE = "detector_int8.onnx"
REGIONS_ENV = "MANGA_TRANS_REGIONS"

# What the model was trained at, and how it is fed: a plain resize to a square,
# rescaled to 0..1, with no normalisation — straight from the model's own
# preprocessor_config.json. The graph itself takes any size, but a page put
# through at its own resolution is both slower and worse than one put through at
# the size the weights were fitted to. A plain resize rather than a letterbox
# because the graph is told the page's shape separately and puts it back itself.
REGIONS_SIZE = 640

# The classes it answers with.
BUBBLE, TEXT_BUBBLE, TEXT_FREE = 0, 1, 2

# What a block turned out to be, in the words the answer carries it in. Finding,
# tracing and cleaning treat the two alike and always will — they are ink on a
# page either way — but a translation must not: lettering in a balloon is someone
# speaking, and lettering outside one is a sound effect, a caption or a sign, which
# is not spoken and is not written the same way. The model is asked both questions
# in the one pass, so this costs nothing to keep and cannot be recovered later.
SPEECH, FREE = "speech", "free"
KINDS = {TEXT_BUBBLE: SPEECH, TEXT_FREE: FREE}

# Below this a box is not worth showing. The dashboard leaves anything under
# UNSURE (0.8) unselected for review, so this is the floor of what it is offered
# rather than the line between sure and unsure.
REGIONS_CONF = 0.35

# Two boxes covering this much of the smaller of them are the same thing found
# twice. RT-DETR matches one query to one object and needs no NMS, but a balloon
# and the text filling it are different classes and are deduped separately.
DUPLICATE = 0.75

# A margin left around every block, as a share of its shorter side. The head
# boxes lettering tightly and sometimes inside it, clipping the edge of a glyph:
# a slightly wider box holds the whole of what it found, gives the reader the
# stroke it was cutting through, and covers the whole letter when the block is
# cleaned by its box rather than by the traced ink.
PAD = 0.04
PAD_MIN = 2


@dataclass(frozen=True)
class Block:
    """One piece of lettering the detector found."""

    box: Box
    confidence: float = 1.0
    # SPEECH or FREE where the detector said which. Empty where nothing did: a
    # block drawn by hand, or one asked about on its own.
    kind: str = ""


def model_path(explicit: str | None = None) -> Path:
    """The ink model's weights: where it already is, else where it should be put."""
    chosen = explicit or os.environ.get(MODEL_ENV)
    if chosen:
        return Path(chosen).expanduser()
    candidates = [Path(directory).expanduser() / MODEL_NAME for directory in MODEL_DIRS]
    return next((path for path in candidates if path.is_file()), candidates[-1])


# A hundred megabytes in one response is enough for a container network that
# carries small files fine — a VM on a Mac, most proxies — to drop it part way
# through, and urlretrieve does not retry. The weights are the same every time,
# so starting over is always safe; it is only ever the transfer that failed.
#
# Waiting between tries is the half that matters. Whatever drops one of these
# drops the next few as well, so attempts in a row all land in the same bad few
# seconds and the retry buys nothing. Long enough, by the last try, to outlast
# the minute-scale throttle a release asset gets when it is fetched repeatedly.
TRIES = 6
BACKOFF = 8

# urlretrieve cannot set headers, and the default `Python-urllib/3.x` is turned
# away by enough of the internet — GitHub's release assets among them, which
# close the connection before answering rather than saying so — that it is worth
# not using it. Nothing here depends on being taken for a browser; it only has to
# not be the one string that is refused out of hand.
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

    # Straight from the cache when it is already there, which is what the image
    # bakes in; only otherwise is anything fetched. Said this way round because
    # announcing a download that did not happen is how a log stops being read.
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

    # The letterbox puts the page's longer side on the canvas, so this is how
    # many page pixels one canvas pixel became. See :data:`GROW`.
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

    This export carries RT-DETR's own postprocessing inside the graph, so what
    comes out is already a class, a corner-to-corner box and a score rather than
    logits to be talked down from. The boxes are in the pixels of whatever was
    passed as ``orig_target_sizes`` and can fall slightly outside them, which is
    what the clip is for. The model answers with a fixed 300 queries whether or
    not it found that many things, so the score is what says which are real.
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

        The graph takes the page at whatever size it is handed and is told
        separately what to scale its answer back to, so the page goes over at the
        size the model was trained on and the boxes come back on the page. That
        second input is (width, height) — measured, not assumed: the other way
        round returns the same balloons transposed.
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

        ``rtl`` is which way reading order runs across the page — right to left
        for Japanese and for the Chinese set in columns, left to right for Korean
        and for a webcomic. It is the caller's rather than measured because it is
        a property of what is being read rather than of the page: the same layout
        of balloons is read both ways round by different languages.
        """
        height, width = image.shape[:2]
        labels, boxes, scores = self.run(image)
        found = decode(labels, boxes, scores, width, height)

        # Per class. A balloon and the text filling it cover each other almost
        # entirely and are not two findings of one thing.
        balloons = suppressed(
            [Block(box, score) for kind, box, score in found if kind == BUBBLE]
        )
        # Thinned on the tight boxes and padded after, never the other way round:
        # a margin put on first can make two neighbours look like one finding.
        # Speech and free lettering are thinned together though they are kept
        # apart: a block the model called both is one block found twice.
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

        # Down the page, then across it the way the language is read.
        # `lib/order.ts` sorts by the same key.
        across = (lambda box: -box.x1) if rtl else (lambda box: box.x0)
        blocks.sort(key=lambda block: (block.box.y0, across(block.box)))
        return blocks, [balloon.box for balloon in balloons]


class Letters:
    """comic-text-detector's segmentation head: where the ink is, pixel by pixel.

    One page at a time: an OpenCV net is not reentrant. The last page's pass is
    kept, because a dashboard asks for the mask again every time the spread is
    changed, and the pass is seconds where everything downstream of it is a
    millisecond.
    """

    def __init__(self, weights: str | Path | None = None) -> None:
        path = ensure_model(str(weights) if weights else None)
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self._lock = threading.Lock()
        self._last: tuple[bytes, tuple] | None = None

    def run(self, image):
        """One pass: the per-pixel text map and the padding put on to get it.

        Some OpenCV builds return the outputs in a different order than they were
        asked for, so they are told apart by shape: the lettering is the
        one-channel map — the two-channel one is where the lines of text run,
        which nothing here wants, and the three-dimensional one is the block head,
        which :class:`Regions` answers better.
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
