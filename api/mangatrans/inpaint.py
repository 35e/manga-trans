"""Filling in what was hidden with the art around it, rather than with white.

White over the inside of a speech bubble is invisible; white over a sound effect
drawn across a face is a hole. So the page is looked at instead: what surrounds
the marked pixels is carried into them, and a screentone runs on through.

Two ways of doing that. :func:`lama` is LaMa fine-tuned on manga — it has seen
line art and knows a hatched edge continues, which is what a clean has to get
right. :func:`telea` is OpenCV's, which propagates colour inward from the rim of
the hole with no notion of structure: fine over flat tone, a smear over anything
drawn. LaMa is the default and Telea is what is left when its weights are not
there.

Either way the seam is the same, and it is the part that matters: the hole is
grown before the fill is *sampled*, so the soft rim of half-ink just outside a
letter is never read as art, but only what was actually marked is painted over.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# How far around a pixel Telea looks to make it with. Wide enough to carry a
# tone across a letter, narrow enough that the fill stays sharp; wider is slower
# and blurrier, not better.
RADIUS = 3

# Lettering has soft edges, so a mask that stops at the ink leaves a rim of
# half-ink just outside it — read as art, that rim is carried inwards and the
# letter comes back as a smudge. The hole is grown by this much to say where the
# fill may *not* be read from. What is replaced is still only what was marked.
EDGE = 2

LAMA_REPO = "ogkalu/lama-manga-onnx-dynamic"
LAMA_FILE = "lama-manga-dynamic.onnx"
LAMA_ENV = "MANGA_TRANS_LAMA"

# The graph takes any size but only in whole multiples of this: it halves the
# image three times over and a stray row has nowhere to go. Measured — a size
# that is not a multiple of eight fails inside a Mul rather than being padded.
BLOCK = 8

# How much of the art around a hole goes over with it. LaMa is told what to make
# the hole out of by what surrounds it, so a crop cut tight to the mark has
# nothing to go on; this much of the mark's own size again on every side, and at
# least LEAST pixels, is enough to carry a tone in without sending the whole page
# through for one sound effect.
CONTEXT = 1.0
LEAST = 64

# Marks closer together than this go through as one crop. Two letters of the same
# word are not worth two passes, and the context around one would hold the other
# as art to copy from — which is how a letter comes back beside where it was.
APART = 24


def grown(mask: np.ndarray, by: int) -> np.ndarray:
    """The mask spread outwards to take in everything within ``by`` pixels.

    Square rather than round: this is a neighbourhood being ruled out, not a
    shape being widened, and the corner of a letter is as much the letter.
    """
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

    # Straight from the cache when it is already there, which is what the image
    # bakes in; only otherwise is anything fetched. Said this way round because
    # announcing a download that did not happen is how a log stops being read.
    try:
        return hf_hub_download(LAMA_REPO, LAMA_FILE, local_files_only=True)
    except Exception:  # noqa: BLE001 — not cached, so go and get it
        print(f"mangatrans: downloading {LAMA_REPO}/{LAMA_FILE}")
        path = hf_hub_download(LAMA_REPO, LAMA_FILE)
        print(f"mangatrans: saved {path}")
        return path


def ensure_model(explicit: str | None = None) -> str:
    """Fetch LaMa's weights (~206 MB) unless they are already there."""
    return model_path(explicit)


def patches(hole: np.ndarray, width: int, height: int) -> list[tuple[int, int, int, int]]:
    """The page cut into the pieces worth sending through, one per mark.

    A page is mostly art that is staying, and LaMa is slow enough that sending
    the whole of one through to take out four balloons is minutes rather than
    seconds. Marks near each other are taken together, both because two passes
    over one word is waste and because a crop around one letter would hold the
    next as material to copy it from.
    """
    near = grown(hole, APART)
    count, labels = cv2.connectedComponents((near > 0).astype(np.uint8), connectivity=8)

    found = []
    for mark in range(1, count):
        ys, xs = np.nonzero(labels == mark)
        if not len(xs):
            continue
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
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
    """LaMa, fine-tuned on manga, behind one onnxruntime session.

    One crop at a time: an onnxruntime session is not reentrant. Stood up on
    first use like the other models, so an API only ever asked to detect never
    pays for it.
    """

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

        Padded out to a whole number of blocks on the way in and cut back on the
        way out. The padding is a reflection of the crop rather than black: an
        edge invented out of nothing is an edge the model tries to continue.
        """
        height, width = crop.shape[:2]
        down = (-height) % BLOCK
        right = (-width) % BLOCK
        if down or right:
            crop = cv2.copyMakeBorder(crop, 0, down, 0, right, cv2.BORDER_REFLECT_101)
            hole = cv2.copyMakeBorder(hole, 0, down, 0, right, cv2.BORDER_CONSTANT, value=0)

        image = (crop.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        marks = (hole.astype(np.float32) / 255.0)[None, None]

        with self._lock:
            (filled,) = self.session.run(None, {"image": image, "mask": marks})

        # Out in the same 0..1 the image went in as.
        out = np.clip(filled[0].transpose(1, 2, 0), 0.0, 1.0) * 255.0
        return out.round().astype(np.uint8)[:height, :width]

    def __call__(self, page: np.ndarray, hole: np.ndarray) -> np.ndarray:
        """The page with every marked piece of it made afresh.

        What comes back is a whole page rather than only the holes, and a crop is
        pasted back whole — the model rebuilds everything it is shown, so the art
        around a mark comes back very slightly redrawn. That is not this
        function's problem to solve: :func:`fill` lays only the marked pixels of
        this over the original, so everything else is the page's own. Each crop is
        read from ``page`` rather than from ``out`` for the same reason — two
        crops that overlap must not be made out of each other.
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

    The mask is a greyscale page of the same size: white is filled in, black is
    left alone, and the greys between are how much of the fill to lay on — which
    keeps a brushed edge from coming out as a staircase.

    ``painter`` takes the page and the hole and hands back a whole page with that
    hole made afresh; Telea is what is used when none is given.
    """
    page = np.array(image.convert("RGB"))
    marks = np.array(mask.convert("L"))
    if not marks.any():
        return Image.fromarray(page, "RGB")

    # Anything marked at all is unknown, however faintly: a pixel half covered by
    # a letter is half ink, and ink is what is being taken out.
    hole = grown(((marks > 0) * 255).astype(np.uint8), EDGE)

    if hole.all():
        # Every pixel is to be filled, so there is nothing left to make it out of.
        # Ahead of the painter rather than inside it: a model handed a page that
        # is entirely hole does not say so, it invents one.
        filled = np.full_like(page, 255)
    else:
        filled = (painter or telea)(page, hole)

    # `hole` was grown and `marks` was not: the rim just outside the mark is kept
    # out of what the fill is made of, but it is not painted over.
    lay = (marks / 255.0)[:, :, None]
    out = page * (1.0 - lay) + filled * lay
    return Image.fromarray(out.round().astype(np.uint8), "RGB")
