"""Text detection with comic-text-detector, run on OpenCV's ONNX backend.

No torch and no onnxruntime: OpenCV reads the ONNX file itself. The weights are
not on PyPI, so :func:`ensure_model` fetches them once from the release they are
published in.
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
# Letters are drawn with soft edges, and the mask stops at the ink. Growing it
# covers the halo that would otherwise be left ringing the place a letter used
# to be — screentone, the ring JPEG leaves around a hard edge, the pale rim of
# an outlined letter, none of which the segmentation calls text.
#
# Four is a starting point, not a finding: on clean black-on-white lettering two
# already leaves nothing behind at any size of page. What it is really covering
# is scanned material, where the right amount depends on the scan. Callers say
# `grow` when they want more.
GROW = 4
GROW_MAX = 64


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
    """The model's padded square of per-pixel text as a mask the page's size.

    The padding letterbox() added goes back off, what is left is stretched to
    the page, and everything the model was sure enough about is grown by
    ``grow`` pixels so no halo is left around a letter that has been hidden.
    """
    seg_h, seg_w = seg.shape[:2]
    kept = seg[
        : max(1, round(seg_h * (INPUT_SIZE - pad_h) / INPUT_SIZE)),
        : max(1, round(seg_w * (INPUT_SIZE - pad_w) / INPUT_SIZE)),
    ]
    # Stretched while still a probability, so the edge of a letter lands where
    # it should before anything is decided about it.
    full = cv2.resize(kept, (width, height), interpolation=cv2.INTER_LINEAR)
    mask = ((full > SEG_THRESHOLD) * 255).astype(np.uint8)

    if grow > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (grow * 2 + 1, grow * 2 + 1)
        )
        mask = cv2.dilate(mask, kernel)
    return mask


class Detector:
    """The loaded network. One page at a time: an OpenCV net is not reentrant."""

    def __init__(self, weights: str | Path | None = None) -> None:
        path = ensure_model(str(weights) if weights else None)
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self._lock = threading.Lock()

    def run(self, image):
        """One pass: the block rows, the per-pixel text map, and the padding.

        Some OpenCV builds hand the outputs back in a different order than they
        were asked for, so they are told apart by shape: the blocks are the only
        one with three dimensions, and the lettering is the one-channel map —
        the two-channel one is where the lines of text run, which nothing here
        wants.
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

        The boxes say which bubble; this says which ink. Hiding by box paints out
        the whole rectangle, artwork and all; hiding by this paints out only what
        was written.
        """
        height, width = image.shape[:2]
        _, seg, pad_w, pad_h = self.run(image)
        return page_mask(seg, width, height, pad_w, pad_h, grow)

    def __call__(self, image) -> list[Block]:
        """Every block of lettering on one RGB page array."""
        height, width = image.shape[:2]
        raw_blocks, _, pad_w, pad_h = self.run(image)

        scale_x = width / (INPUT_SIZE - pad_w)
        scale_y = height / (INPUT_SIZE - pad_h)
        boxes, confidences = decode_blocks(raw_blocks[0], CONF_THRESHOLD, NMS_THRESHOLD)

        blocks = []
        for xyxy, confidence in zip(boxes, confidences):
            box = Box(
                int(round(xyxy[0] * scale_x)),
                int(round(xyxy[1] * scale_y)),
                int(round(xyxy[2] * scale_x)),
                int(round(xyxy[3] * scale_y)),
            ).clipped(width, height)
            if box.w > 1 and box.h > 1:
                blocks.append(Block(box=box, confidence=float(confidence)))

        blocks.sort(key=lambda block: (block.box.y0, -block.box.x1))
        return blocks
