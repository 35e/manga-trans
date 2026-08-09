"""Cutting a block the detector ran together back into one block per bubble.

Two balloons that overlap are often boxed as one. That is worse than it looks:
the block is read as one string, so two speakers arrive at the translator as a
single line, and the lettering that comes back is set into one balloon.

What separates them is blank. Inside a balloon the lettering is set solid —
columns of vertical Japanese sit about a fifth of a character apart — while
between two balloons there is padding, an outline and usually some art. So the
per-pixel text mask is projected onto each axis and cut across any run of blank
wide enough to be a wall rather than a line gap, over and over until nothing is
wide enough. Cuts are axis-aligned because a block is a rectangle; a gap no
straight line can follow is left alone.

This is measured in characters rather than pixels, since a page may be lettered
at any size. See :func:`character` for how that size is read off the mask.
"""

from __future__ import annotations

import cv2
import numpy as np

from .geometry import Box

# How much blank means two balloons rather than two lines of one, in characters.
# Measured on both: the worst gap inside a single column — small kana and
# punctuation, which leave most of their cell empty — reaches about 1.1, and the
# tightest gap between two balloons the detector had run together was about 2.6.
# This sits between them with room either side.
GAP = 1.6

# A floor in pixels, for text small enough that 1.6 characters is a few pixels
# and a stray speck of blank would cut a line in half.
GAP_MIN = 8

# How large one character is, as a percentile of the ink marks. A median is what
# this wants to be, but punctuation and small kana are a large enough minority of
# the marks in a line to drag one down — on a column reading あっ、、、そうっ、か
# the median says 23px where the characters are 42.
CHARACTER = 75


def character(text: np.ndarray) -> float:
    """How large one character is, in pixels, read off the ink itself.

    Every mark is measured by its longer side, which is a whole character for a
    kana and a radical for a kanji, then capped at the region's shorter side —
    without that cap, characters set solid enough to touch come back as one mark
    the length of the column they are in.

    Deliberately not the writing direction: this holds for lines running down the
    page and across it, which the projections below cannot tell apart.
    """
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        text.astype(np.uint8), connectivity=8
    )
    limit = min(text.shape)
    if count < 2 or limit < 1:
        return float(max(1, limit))
    marks = [
        min(max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]), limit)
        for i in range(1, count)
    ]
    return max(1.0, float(np.percentile(marks, CHARACTER)))


def widest_blank(profile: np.ndarray) -> tuple[int, int]:
    """The widest run of blank with ink on both sides, as (start, length).

    Blank at either end is only slack in the box and is no use for cutting, so
    the search runs between the first and last ink.
    """
    ink = np.flatnonzero(profile)
    if len(ink) < 2:
        return (0, 0)

    best_at, best = 0, 0
    at = ink[0]
    for edge in ink[1:]:
        run = int(edge - at) - 1
        if run > best:
            best_at, best = int(at) + 1, run
        at = edge
    return best_at, best


def inked(text: np.ndarray) -> Box | None:
    """The box around everything set in ``text``, or None if nothing is."""
    across = np.flatnonzero(text.any(axis=0))
    down = np.flatnonzero(text.any(axis=1))
    if not len(across) or not len(down):
        return None
    return Box(int(across[0]), int(down[0]), int(across[-1]) + 1, int(down[-1]) + 1)


def where(text: np.ndarray, widest: int) -> tuple[int, int] | None:
    """Which axis to cut across and where, or None if no gap is wide enough.

    The wider of the two gaps wins, so a page cut both ways comes apart at its
    plainest seam first and the rest is left to the recursion.
    """
    across = widest_blank(text.any(axis=0))
    down = widest_blank(text.any(axis=1))
    axis, (start, run) = (0, across) if across[1] >= down[1] else (1, down)
    if run < widest:
        return None
    return axis, start + run // 2


def parts(text: np.ndarray, widest: int) -> list[Box]:
    """``text`` cut apart at every wide enough blank, boxed, in its own pixels."""
    found = where(text, widest)
    if found is None:
        here = inked(text)
        return [here] if here else []

    axis, cut = found
    if axis == 0:
        return parts(text[:, :cut], widest) + [
            box.moved(cut, 0) for box in parts(text[:, cut:], widest)
        ]
    return parts(text[:cut, :], widest) + [
        box.moved(0, cut) for box in parts(text[cut:, :], widest)
    ]


def pieces(text: np.ndarray, box: Box) -> list[Box]:
    """``box`` as one box per balloon, or ``[box]`` where it only held one.

    ``text`` is the page-sized per-pixel text mask, and it must be the ungrown
    one: growing it to cover the halo around a letter also closes the gaps this
    is here to measure.

    A block that does not come apart is handed back exactly as the detector drew
    it, untightened — this only ever answers differently for a block that was
    really two.
    """
    crop = text[box.y0 : box.y1, box.x0 : box.x1]
    if crop.size == 0 or not crop.any():
        return [box]

    widest = max(GAP_MIN, round(GAP * character(crop)))
    found = parts(crop, widest)
    if len(found) < 2:
        return [box]
    return [piece.moved(box.x0, box.y0) for piece in found]
