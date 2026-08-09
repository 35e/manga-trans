"""Text detection with comic-text-detector, on OpenCV's ONNX backend.

No torch and no onnxruntime: OpenCV reads the ONNX file itself. The weights are
not on PyPI, so :func:`ensure_model` fetches them once.
"""

from __future__ import annotations

import os
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .geometry import Box

MODEL_NAME = "comictextdetector.pt.onnx"
MODEL_URL = (
    "https://github.com/zyddnys/manga-image-translator/releases/download/"
    "beta-0.2.1/" + MODEL_NAME
)
MODEL_ENV = "MANGA_TRANS_MODEL"
MODEL_DIRS = ("/opt/models", "~/.cache/manga-trans")

INPUT_SIZE = 1024
CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.35

# The segmentation head answers per pixel, between 0 and 1.
SEG_THRESHOLD = 0.5

# Letters have soft edges and the mask stops at the ink, so the mask is grown to
# cover the halo left ringing a hidden letter — screentone, JPEG ringing, the
# pale rim of an outlined letter. Four suits clean lettering; scans want more,
# which is why callers can say `grow`.
GROW = 4
GROW_MAX = 64

# A margin left around every block, as a share of one character. The block head
# boxes lettering tightly and sometimes inside it, clipping the edge of a glyph:
# a box a fraction of a character wider holds the whole of what it found, gives
# the reader the stroke it was cutting through, and covers the whole letter when
# the block is cleaned by its box rather than by the traced ink.
#
# Measured in characters rather than pixels or a share of the box, so it comes
# out the same on a page scanned at any size and on a block of any shape.
PAD = 0.25
PAD_MIN = 2

# Two blocks covering this much of the smaller of them are the same lettering
# found twice. See :func:`suppressed` for why NMS does not already catch it.
DUPLICATE = 0.6


@dataclass(frozen=True)
class Block:
    """One piece of lettering the detector found."""

    box: Box
    confidence: float = 1.0


def model_path(explicit: str | None = None) -> Path:
    """The weights file: where it already is, else where it should be put."""
    chosen = explicit or os.environ.get(MODEL_ENV)
    if chosen:
        return Path(chosen).expanduser()
    candidates = [Path(directory).expanduser() / MODEL_NAME for directory in MODEL_DIRS]
    return next((path for path in candidates if path.is_file()), candidates[-1])


def ensure_model(explicit: str | None = None) -> Path:
    """Download the weights (~95 MB) unless they are already there."""
    path = model_path(explicit)
    if path.is_file():
        return path
    print(f"mangatrans: downloading {MODEL_URL}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(MODEL_URL, partial)
    partial.replace(path)
    print(f"mangatrans: saved {path} ({path.stat().st_size / 1e6:.0f} MB)")
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


def decode_blocks(raw, conf_threshold: float, nms_threshold: float):
    """YOLO head rows to (boxes xyxy, confidences) in canvas pixels."""
    empty = (np.zeros((0, 4), np.float32), np.zeros(0, np.float32))
    if raw.size == 0:
        return empty

    scores = raw[:, 5:]
    confidence = raw[:, 4] * scores.max(1)
    keep = confidence > conf_threshold
    raw, confidence = raw[keep], confidence[keep]
    if not len(raw):
        return empty

    cx, cy, w, h = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
    xywh = np.stack([cx - w / 2, cy - h / 2, w, h], axis=1)
    kept = cv2.dnn.NMSBoxes(
        xywh.tolist(), confidence.tolist(), conf_threshold, nms_threshold
    )
    if len(kept) == 0:
        return empty

    kept = np.asarray(kept).flatten()
    xywh, confidence = xywh[kept], confidence[kept]
    xyxy = np.stack(
        [xywh[:, 0], xywh[:, 1], xywh[:, 0] + xywh[:, 2], xywh[:, 1] + xywh[:, 3]],
        axis=1,
    )
    return xyxy, confidence


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

    if grow > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (grow * 2 + 1, grow * 2 + 1)
        )
        mask = cv2.dilate(mask, kernel)
    return mask


def suppressed(blocks: list[Block]) -> list[Block]:
    """The same lettering found twice, thinned down to the surest of them.

    The head sometimes draws a box around two balloons *and* a box around one of
    them, overlapping too little for NMS to throw either away. Splitting the
    first then turns that pair into an exact duplicate — and a duplicated block
    is read twice, translated twice, and lettered twice into the same place.

    NMS cannot do this itself: it runs on what the head said, before anything has
    been cut, so it never sees the pieces. Run on the tight boxes, before they
    are padded, or a margin could make two neighbours look like one.
    """
    kept: list[Block] = []
    for block in sorted(blocks, key=lambda block: -block.confidence):
        if not any(block.box.covers(other.box) >= DUPLICATE for other in kept):
            kept.append(block)
    return kept


class Detector:
    """The loaded network. One page at a time: an OpenCV net is not reentrant."""

    def __init__(self, weights: str | Path | None = None) -> None:
        path = ensure_model(str(weights) if weights else None)
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self._lock = threading.Lock()

    def run(self, image):
        """One pass: the block rows, the per-pixel text map, and the padding.

        Some OpenCV builds return the outputs in a different order than they were
        asked for, so they are told apart by shape: the blocks are the only
        3-dimensional one, and the lettering is the one-channel map — the
        two-channel one is where the lines of text run, which nothing here wants.
        """
        canvas, pad_w, pad_h = letterbox(image)
        blob = cv2.dnn.blobFromImage(
            canvas, scalefactor=1 / 255.0, size=(INPUT_SIZE, INPUT_SIZE)
        )
        with self._lock:
            self.net.setInput(blob)
            outputs = self.net.forward(("blk", "seg", "det"))

        blocks = next(output for output in outputs if output.ndim == 3)
        seg = next(
            output for output in outputs if output.ndim == 4 and output.shape[1] == 1
        )
        return blocks, seg[0, 0], pad_w, pad_h

    def letters(self, image, grow: int = GROW) -> np.ndarray:
        """A mask of the lettering itself, pixel by pixel, the page's size.

        The boxes say which bubble; this says which ink, so a clean can hide the
        words and leave the art they were drawn over.
        """
        height, width = image.shape[:2]
        _, seg, pad_w, pad_h = self.run(image)
        return page_mask(seg, width, height, pad_w, pad_h, grow)

    def __call__(self, image) -> list[Block]:
        """Every block of lettering on one RGB page array, in reading order.

        A block holding two balloons the detector ran together is cut back into
        one block each — see :mod:`mangatrans.split`. It is done here rather than
        left to the caller because the segmentation this needs is already in hand
        from the same pass, and because a merged block is wrong for everything
        downstream: it is read as one string and translated as one line.
        """
        # Imported here, not at the top: the Dockerfile copies this module in
        # before the model prefetch and `split` after it, so that editing a
        # threshold there does not send the next build back for 550 MB of
        # weights. At the top, the prefetch step would not find it.
        from . import split

        height, width = image.shape[:2]
        raw_blocks, seg, pad_w, pad_h = self.run(image)

        scale_x = width / (INPUT_SIZE - pad_w)
        scale_y = height / (INPUT_SIZE - pad_h)
        boxes, confidences = decode_blocks(raw_blocks[0], CONF_THRESHOLD, NMS_THRESHOLD)

        found = []
        for xyxy, confidence in zip(boxes, confidences):
            box = Box(
                int(round(xyxy[0] * scale_x)),
                int(round(xyxy[1] * scale_y)),
                int(round(xyxy[2] * scale_x)),
                int(round(xyxy[3] * scale_y)),
            ).clipped(width, height)
            if box.w > 1 and box.h > 1:
                found.append(Block(box=box, confidence=float(confidence)))

        # Ungrown: growing the mask to cover the halo around a letter also closes
        # the gaps the split is measuring.
        text = page_mask(seg, width, height, pad_w, pad_h, 0) > 0

        # Split first, then pad. A margin put on before would close the very gaps
        # the split is looking for, and would push two neighbours together.
        pieces = [
            Block(box=piece, confidence=block.confidence)
            for block in found
            for piece in split.pieces(text, block.box)
        ]

        blocks = []
        for block in suppressed(pieces):
            crop = text[block.box.y0 : block.box.y1, block.box.x0 : block.box.x1]
            by = (
                max(PAD_MIN, round(PAD * split.character(crop)))
                if crop.any()
                else PAD_MIN
            )
            blocks.append(
                Block(
                    box=block.box.grown(by).clipped(width, height),
                    confidence=block.confidence,
                )
            )

        # Down the page, then right to left. `lib/order.ts` sorts by the same key.
        # After splitting, so the halves of a cut block land where they are read.
        blocks.sort(key=lambda block: (block.box.y0, -block.box.x1))
        return blocks
