"""Filling in what was hidden with the art around it, rather than with white.

White over the inside of a speech bubble is invisible; white over a sound effect
drawn across a face is a hole. So the page is looked at instead: what surrounds
the marked pixels is carried into them, and a screentone runs on through.

This is OpenCV's Telea inpainting, which is in opencv-python-headless already:
no model, no network, no torch.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

# How far around a pixel is looked at to make it with. Wide enough to carry a
# tone across a letter, narrow enough that the fill stays sharp; wider is slower
# and blurrier, not better.
RADIUS = 3

# Lettering has soft edges, so a mask that stops at the ink leaves a rim of
# half-ink just outside it — read as art, that rim is carried inwards and the
# letter comes back as a smudge. The hole is grown by this much to say where the
# fill may *not* be read from. What is replaced is still only what was marked.
EDGE = 2


def grown(mask: np.ndarray, by: int) -> np.ndarray:
    """The mask spread outwards to take in everything within ``by`` pixels.

    Square rather than round: this is a neighbourhood being ruled out, not a
    shape being widened, and the corner of a letter is as much the letter.
    """
    if by <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (by * 2 + 1, by * 2 + 1))
    return cv2.dilate(mask, kernel)


def fill(image: Image.Image, mask: Image.Image) -> Image.Image:
    """A copy of the page with what ``mask`` marks filled in from around it.

    The mask is a greyscale page of the same size: white is filled in, black is
    left alone, and the greys between are how much of the fill to lay on — which
    keeps a brushed edge from coming out as a staircase.
    """
    page = np.array(image.convert("RGB"))
    marks = np.array(mask.convert("L"))
    if not marks.any():
        return Image.fromarray(page, "RGB")

    # Anything marked at all is unknown, however faintly: a pixel half covered by
    # a letter is half ink, and ink is what is being taken out.
    hole = grown(((marks > 0) * 255).astype(np.uint8), EDGE)

    if hole.all():
        # Every pixel is to be filled, so there is nothing left to look at.
        filled = np.full_like(page, 255)
    else:
        filled = cv2.inpaint(page, hole, RADIUS, cv2.INPAINT_TELEA)

    # `hole` was grown and `marks` was not: the rim just outside the mark is kept
    # out of what the fill is made of, but it is not painted over.
    lay = (marks / 255.0)[:, :, None]
    out = page * (1.0 - lay) + filled * lay
    return Image.fromarray(out.round().astype(np.uint8), "RGB")
