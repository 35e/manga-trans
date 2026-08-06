"""Reading the lettering, with manga-ocr.

The detector says where the text is; this says what it says. manga-ocr is a
model trained on manga specifically — vertical lines, stylised fonts, furigana —
which is why it manages what general OCR does not. It runs on the CPU.

It is the one thing here that needs torch, and it is why the image is as large
as it is. Nothing else in this package imports it, and the import is deferred to
the moment a page is first read, so an API that is only ever asked to detect,
clean or render never pays for it.
"""

from __future__ import annotations

import os
import threading
from typing import Sequence

from PIL import Image

from .geometry import Box

MODEL_NAME = "kha-white/manga-ocr-base"
MODEL_ENV = "MANGA_TRANS_OCR_MODEL"

# The detector boxes lettering tightly. A little air around it reads better,
# but not so much that the next bubble is drawn in.
PAD = 0.03
PAD_MIN = 2


def model_name(explicit: str | None = None) -> str:
    """The model to read with: the one asked for, else the one trained on manga."""
    return explicit or os.environ.get(MODEL_ENV) or MODEL_NAME


def ensure_model(explicit: str | None = None) -> str:
    """Fetch the weights (~450 MB) into the Hugging Face cache unless they are there.

    Downloading them is not the same as loading them: the image is built with
    this, which needs no torch, rather than by standing a whole model up.
    """
    from huggingface_hub import snapshot_download

    name = model_name(explicit)
    print(f"mangatrans: downloading {name}")
    path = snapshot_download(name)
    print(f"mangatrans: saved {path}")
    return name


def padded(box: Box, width: int, height: int) -> Box:
    """The box with a margin, still on the page."""
    pad = max(PAD_MIN, round(PAD * min(box.w, box.h)))
    return Box(
        box.x0 - pad, box.y0 - pad, box.x1 + pad, box.y1 + pad
    ).clipped(width, height)


class Reader:
    """The loaded OCR model. One box at a time: torch generation is not reentrant."""

    def __init__(self, model: str | None = None, ocr=None) -> None:
        # ``ocr`` is anything that turns one image into one string. Passing it
        # runs the cropping and the loop without half a gigabyte of weights,
        # which is what the tests do.
        self.ocr = ocr if ocr is not None else self.load(model)
        self._lock = threading.Lock()

    @staticmethod
    def load(model: str | None = None):
        """The manga-ocr model itself. Imported here, so torch loads on first use."""
        from manga_ocr import MangaOcr

        return MangaOcr(
            pretrained_model_name_or_path=model_name(model), force_cpu=True
        )

    def __call__(self, image: Image.Image, boxes: Sequence[Box]) -> list[str]:
        """What each box says, in the order the boxes were given.

        A box too small to hold lettering reads as "" rather than as whatever the
        model makes of four pixels.
        """
        texts = []
        for box in boxes:
            if box.w < 4 or box.h < 4:
                texts.append("")
                continue
            crop = padded(box, image.width, image.height)
            piece = image.crop((crop.x0, crop.y0, crop.x1, crop.y1))
            # The lock is held for one crop, not the whole page, so a second
            # request is only ever a box behind rather than a page behind.
            with self._lock:
                texts.append(self.ocr(piece).strip())
        return texts
