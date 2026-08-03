"""Text detection with comic-text-detector, run on OpenCV's ONNX backend."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
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
MASK_THRESHOLD = 0.3


class ModelMissing(RuntimeError):
    pass


@dataclass
class Block:
    box: Box
    confidence: float = 1.0


@dataclass
class Detection:
    """One page: the text blocks, and a per-pixel mask of the lettering."""
    blocks: list[Block] = field(default_factory=list)
    mask: object = None


def model_path(explicit: str | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    elif os.environ.get(MODEL_ENV):
        candidates.append(Path(os.environ[MODEL_ENV]).expanduser())
    else:
        candidates = [Path(d).expanduser() / MODEL_NAME for d in MODEL_DIRS]
    return next((path for path in candidates if path.is_file()), None)


def missing_model_message(explicit: str | None = None) -> str:
    looked = explicit or os.environ.get(MODEL_ENV) or ", ".join(MODEL_DIRS)
    return (
        f"{MODEL_NAME} was not found (looked in {looked}).\n"
        f"Fetch it with: python scripts/fetch_models.py"
    )


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


class Detector:
    def __init__(self, weights: str | Path | None = None) -> None:

        path = model_path(str(weights) if weights else None)
        if path is None:
            raise ModelMissing(missing_model_message(str(weights) if weights else None))
        self.net = cv2.dnn.readNetFromONNX(str(path))

    def __call__(self, image) -> Detection:
        """Detect on one RGB page array."""

        height, width = image.shape[:2]
        canvas, pad_w, pad_h = letterbox(image)
        blob = cv2.dnn.blobFromImage(
            canvas, scalefactor=1 / 255.0, size=(INPUT_SIZE, INPUT_SIZE)
        )
        self.net.setInput(blob)
        outputs = self.net.forward(("blk", "seg", "det"))

        # Some OpenCV builds hand the outputs back in a different order than they
        # were asked for; shape tells them apart unambiguously.
        raw_blocks = next(o for o in outputs if o.ndim == 3)
        raw_mask = next(o for o in outputs if o.ndim == 4 and o.shape[1] == 1)

        mask = (raw_mask[0, 0] > MASK_THRESHOLD).astype(np.uint8) * 255
        mask = mask[: INPUT_SIZE - pad_h, : INPUT_SIZE - pad_w]
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        scale_x = width / (INPUT_SIZE - pad_w)
        scale_y = height / (INPUT_SIZE - pad_h)
        boxes, confidences = decode_blocks(
            raw_blocks[0], CONF_THRESHOLD, NMS_THRESHOLD
        )

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

        blocks.sort(key=lambda b: (b.box.y0, -b.box.x1))
        return Detection(blocks=blocks, mask=mask)
