"""Filling in what was hidden with the art around it, rather than with white.

White over the inside of a speech bubble is invisible, because the bubble was
white already. White over a sound effect drawn across a face is a hole. Here the
page is looked at instead: what surrounds the marked pixels is carried into
them, so a screentone runs on through, a bubble's outline joins up again, and
the line that ran under a letter comes out the other side of it.

This is OpenCV's Telea inpainting — the fast marching method, which fills a hole
from its rim inwards, each pixel taken from the ones already known around it. It
is in opencv-python-headless, so it costs nothing that is not already here: no
model, no network, no torch.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

# How far around a pixel is looked at to make it with, in pixels. Three is what
# OpenCV's own tutorial uses and what taking text out wants: wide enough to
# carry a tone across a letter, narrow enough that the fill stays sharp. Wider
# is not better — it is slower and blurrier.
RADIUS = 3

# The fill is made of the pixels around the mark, so the pixels around the mark
# had better not be the thing being taken out. Lettering is drawn with soft
# edges, and a mask that stops at the ink leaves a rim of half-ink just outside
# it; read as art, that rim gets carried inwards and the letter is put back as a
# smudge. So the hole is grown by this much before it is filled, which only says
# where the fill may *not* be read from. What is replaced is still only ever
# what was marked.
EDGE = 2


def grown(mask: np.ndarray, by: int) -> np.ndarray:
    """The mask spread outwards to take in everything within ``by`` pixels.

    Square rather than round, unlike the growing the detector does: this is not
    a shape being widened but a neighbourhood being ruled out, and the corner of
    a letter is as much the letter as the side of it is.
    """
    if by <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (by * 2 + 1, by * 2 + 1))
    return cv2.dilate(mask, kernel)


def fill(image: Image.Image, mask: Image.Image) -> Image.Image:
    """A copy of the page with what ``mask`` marks filled in from around it.

    The mask is a greyscale page of the same size: white is filled in, black is
    left alone, and the greys between are how much of the fill to lay on — which
    is what keeps a brushed edge from coming out as a staircase.
    """
    page = np.array(image.convert("RGB"))
    marks = np.array(mask.convert("L"))
    if not marks.any():
        return Image.fromarray(page, "RGB")  # nothing marked, nothing to fill

    # Anything marked at all is unknown, however faintly it was marked: a pixel
    # half covered by a letter is half ink, and ink is what is being taken out.
    hole = grown(((marks > 0) * 255).astype(np.uint8), EDGE)

    if hole.all():
        # Every pixel is to be filled, so there is nothing left to look at.
        # White is all that can be said, and it is what the old behaviour said.
        filled = np.full_like(page, 255)
    else:
        # Which channel is which does not matter to inpainting — it treats the
        # three the same — so RGB goes in and RGB comes back.
        filled = cv2.inpaint(page, hole, RADIUS, cv2.INPAINT_TELEA)

    # `hole` was grown and `marks` was not: the rim just outside the mark is
    # kept out of what the fill is made of, but it is not painted over.
    lay = (marks / 255.0)[:, :, None]
    out = page * (1.0 - lay) + filled * lay
    return Image.fromarray(out.round().astype(np.uint8), "RGB")
