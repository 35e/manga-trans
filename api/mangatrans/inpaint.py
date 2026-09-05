"""Filling in what was hidden with the art around it, rather than with white."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

RADIUS = 3

EDGE = 2

LAMA_REPO = "ogkalu/lama-manga-onnx-dynamic"
LAMA_FILE = "lama-manga-dynamic.onnx"
LAMA_ENV = "MANGA_TRANS_LAMA"

BLOCK = 8

CONTEXT = 1.0
LEAST = 64

APART = 24

LARGEST = 1_000_000


def grown(mask: np.ndarray, by: int) -> np.ndarray:
    """The mask spread outwards to take in everything within ``by`` pixels."""
    if by <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (by * 2 + 1, by * 2 + 1))
    return cv2.dilate(mask, kernel)


def model_path(explicit: str | None = None) -> str:
    """Where LaMa's weights are, fetching them if this is the first time."""
    chosen = explicit or os.environ.get(LAMA_ENV)
    if chosen:
        return str(Path(chosen).expanduser())

    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(LAMA_REPO, LAMA_FILE, local_files_only=True)
    except Exception:  # noqa: BLE001
        print(f"mangatrans: downloading {LAMA_REPO}/{LAMA_FILE}")
        path = hf_hub_download(LAMA_REPO, LAMA_FILE)
        print(f"mangatrans: saved {path}")
        return path


def ensure_model(explicit: str | None = None) -> str:
    """Fetch LaMa's weights (~206 MB) unless they are already there."""
    return model_path(explicit)


def patches(hole: np.ndarray, width: int, height: int) -> list[tuple[int, int, int, int]]:
    """The page cut into the pieces worth sending through, one per mark."""
    near = grown(hole, APART)
    _, _, stats, _ = cv2.connectedComponentsWithStats(
        cv2.compare(near, 0, cv2.CMP_GT), connectivity=8
    )

    found = []
    for x0, y0, wide, tall, _ in stats[1:]:
        x0, y0, wide, tall = map(int, (x0, y0, wide, tall))
        x1, y1 = x0 + wide, y0 + tall
        room = max(LEAST, round(CONTEXT * max(x1 - x0, y1 - y0)))
        found.append(
            (
                max(0, x0 - room),
                max(0, y0 - room),
                min(width, x1 + room),
                min(height, y1 + room),
            )
        )
    return found


class Lama:
    """LaMa, fine-tuned on manga, behind one onnxruntime session."""

    def __init__(self, weights: str | None = None) -> None:
        import onnxruntime as ort

        from .read import quieted

        with quieted():
            self.session = ort.InferenceSession(
                model_path(weights), providers=["CPUExecutionProvider"]
            )
        self._lock = threading.Lock()

    def patch(self, crop: np.ndarray, hole: np.ndarray) -> np.ndarray:
        """One crop with its hole filled in, the same size it came in.

        The padding is a reflection of the crop rather than black: an edge
        invented out of nothing is an edge the model tries to continue.
        """
        height, width = crop.shape[:2]
        small = self.working(crop.shape[:2])
        if small is not None:
            crop = cv2.resize(crop, small, interpolation=cv2.INTER_AREA)
            hole = (cv2.resize(hole, small, interpolation=cv2.INTER_AREA) > 0).astype(
                np.uint8
            ) * 255

        tall, wide = crop.shape[:2]
        down = (-tall) % BLOCK
        right = (-wide) % BLOCK
        if down or right:
            crop = cv2.copyMakeBorder(crop, 0, down, 0, right, cv2.BORDER_REFLECT_101)
            hole = cv2.copyMakeBorder(hole, 0, down, 0, right, cv2.BORDER_CONSTANT, value=0)

        image = (crop.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        marks = (hole.astype(np.float32) / 255.0)[None, None]

        with self._lock:
            (filled,) = self.session.run(None, {"image": image, "mask": marks})

        out = np.clip(filled[0].transpose(1, 2, 0), 0.0, 1.0) * 255.0
        out = out.round().astype(np.uint8)[:tall, :wide]
        if small is not None:
            out = cv2.resize(out, (width, height), interpolation=cv2.INTER_LINEAR)
        return out

    @staticmethod
    def working(shape: tuple[int, int]) -> tuple[int, int] | None:
        """The size to put a crop of this shape through at, or None to leave it."""
        height, width = shape
        pixels = width * height
        if pixels <= LARGEST:
            return None
        scale = (LARGEST / pixels) ** 0.5
        return (max(BLOCK, round(width * scale)), max(BLOCK, round(height * scale)))

    def __call__(self, page: np.ndarray, hole: np.ndarray) -> np.ndarray:
        """The page with every marked piece of it made afresh.

        Each crop is read from ``page`` rather than from ``out``: two crops that
        overlap must not be made out of each other.
        """
        height, width = page.shape[:2]
        out = page.copy()
        for x0, y0, x1, y1 in patches(hole, width, height):
            out[y0:y1, x0:x1] = self.patch(
                np.ascontiguousarray(page[y0:y1, x0:x1]),
                np.ascontiguousarray(hole[y0:y1, x0:x1]),
            )
        return out


def telea(page: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """OpenCV's inpainting: no model, no weights, and no idea what a line is."""
    return cv2.inpaint(page, hole, RADIUS, cv2.INPAINT_TELEA)


def fill(image: Image.Image, mask: Image.Image, painter=None) -> Image.Image:
    """A copy of the page with what ``mask`` marks filled in from around it.

    The mask is greyscale and page-sized: white filled in, black left alone,
    greys how much of the fill to lay on. Telea is used when no painter is given.
    """
    page = np.array(image.convert("RGB"))
    marks = np.array(mask.convert("L"))
    if not marks.any():
        return Image.fromarray(page, "RGB")

    hole = grown(cv2.compare(marks, 0, cv2.CMP_GT), EDGE)

    if hole.all():
        filled = np.full_like(page, 255)
    else:
        filled = (painter or telea)(page, hole)

    return Image.composite(
        Image.fromarray(filled, "RGB"),
        Image.fromarray(page, "RGB"),
        Image.fromarray(marks, "L"),
    )
