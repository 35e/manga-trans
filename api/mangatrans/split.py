"""Cutting a block the detector ran together back into one block per bubble.

Two balloons that overlap are often boxed as one. That is worse than it looks:
the block is read as one string, so two speakers arrive at the translator as a
single line, and the lettering that comes back is set into one balloon.

What separates them is blank. Inside a balloon the lettering is set solid, while
between two balloons there is padding, an outline and usually some art. So the
per-pixel text mask is projected onto each axis and cut across any run of blank
wide enough to be a wall rather than the gap between two lines, over and over
until nothing is wide enough. Cuts are axis-aligned because a block is a
rectangle; a gap no straight line can follow is left alone.

How much blank that takes depends on what the cut would stand through, which is
the one thing that makes this safe to do at all — see :data:`GAP`. Everything is
measured in characters rather than pixels, since a page may be lettered at any
size; see :func:`character`.
"""

from __future__ import annotations

import cv2
import numpy as np

from .geometry import Box

# How much blank means two balloons rather than two lines of one, in characters,
# for a cut standing through several lines at once.
#
# Such a cut is a strong signal: every line it crosses has to fall blank in the
# same place at the same time, which lettering set inside one balloon does not do
# — the widest gap measured through a whole balloon was 0.37 of a character. So
# barely more than a line gap is already worth cutting on.
GAP = 0.8

# The same, for a cut running the length of a single line, where there are no
# other lines to agree with it.
#
# Every gap between two characters is then a candidate, and small kana and
# punctuation leave most of their cell empty: a column reading あっ、、、そうっ、か
# has gaps of 1.1 characters in it and is still one line of one balloon. This has
# to clear that, so a lone line is only ever cut on a plainly larger blank.
GAP_ALONE = 1.5

# How many lines a cut must stand through to be read the first way. Two would be
# the natural reading, but a block exactly two characters across is the awkward
# middle — that is one column of vertical Japanese with its punctuation as often
# as it is two — so the strict measure is held on a little past it.
LINES = 2.5

# Lines of one block share an edge across a cut: vertical Japanese starts every
# column at the same height and simply stops early on the last one. Two blocks
# set beside each other share nothing, and their spans come out shifted the same
# way at *both* ends. Where that happens the blank between them barely matters —
# they were never one block — so this much shift is enough to cut on.
#
# Both ends have to agree, which is what tells a second block from a short last
# column or from columns centred against each other. Measured: a balloon of two
# centred columns is out by 3.0 characters at the start and back 3.0 at the end,
# where two balloons a character apart are out by 1.0 at both.
STAGGER = 0.5

# The blank a staggered pair still needs, so lettering that runs together is
# never cut on alignment alone.
GAP_STAGGERED = 0.3

# A floor in pixels, for text small enough that a character is a few pixels and a
# stray speck of blank would cut a line in half.
GAP_MIN = 8

# How large one character is, as a percentile of the ink marks. A median is what
# this wants to be, but punctuation and small kana are a large enough minority of
# the marks in a line to drag one down — on the column above, the median says
# 23px where the characters are 42.
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


def blanks(profile: np.ndarray) -> list[tuple[int, int]]:
    """Every run of blank with ink on both sides, as (start, length).

    Blank at either end is only slack in the box and is no use for cutting, so
    the search runs between the first and last ink. Every run is wanted and not
    just the widest: two balloons can sit as close together as the columns
    inside one of them, and then it is the alignment that tells them apart.
    """
    ink = np.flatnonzero(profile)
    if len(ink) < 2:
        return []
    runs = np.diff(ink) - 1
    return [
        (int(ink[at]) + 1, int(run)) for at, run in enumerate(runs.tolist()) if run > 0
    ]


def widest_blank(profile: np.ndarray) -> tuple[int, int]:
    """The widest of those runs, or (0, 0) where there is none."""
    found = blanks(profile)
    return max(found, key=lambda run: run[1]) if found else (0, 0)


def reaches(part: np.ndarray, axis: int) -> tuple[int, int] | None:
    """Where the ink in one side of a cut starts and ends, across that cut."""
    along = part.any(axis=1) if axis == 0 else part.any(axis=0)
    on = np.flatnonzero(along)
    return (int(on[0]), int(on[-1])) if len(on) else None


def staggered(text: np.ndarray, axis: int, cut: int, em: float) -> bool:
    """Whether the two sides of a cut were set as separate blocks.

    True when their spans are shifted the same way at both ends — see
    :data:`STAGGER`. One side lying inside the other is not a stagger: that is a
    column that stopped early, or two columns centred against one another, and
    both belong to the block they are in.
    """
    before = reaches(text[:, :cut] if axis == 0 else text[:cut, :], axis)
    after = reaches(text[:, cut:] if axis == 0 else text[cut:, :], axis)
    if before is None or after is None:
        return False

    start, end = after[0] - before[0], after[1] - before[1]
    if start > 0 and end > 0:
        shift = min(start, end)
    elif start < 0 and end < 0:
        shift = min(-start, -end)
    else:
        return False
    return shift >= STAGGER * em


def wide_enough(run: int, through: float, em: float) -> bool:
    """Whether a blank run is a wall between balloons rather than a line gap.

    ``through`` is how much lettering the cut would stand through, in characters,
    measured across the cut. A cut crossing several lines has all of them
    agreeing that the page is blank there; a cut along a lone line has only that
    line's own gaps to go on, and has to clear the widest of them.
    """
    need = GAP if through >= LINES else GAP_ALONE
    return run >= max(GAP_MIN, round(need * em))


def inked(text: np.ndarray) -> Box | None:
    """The box around everything set in ``text``, or None if nothing is."""
    across = np.flatnonzero(text.any(axis=0))
    down = np.flatnonzero(text.any(axis=1))
    if not len(across) or not len(down):
        return None
    return Box(int(across[0]), int(down[0]), int(across[-1]) + 1, int(down[-1]) + 1)


def where(text: np.ndarray, em: float) -> tuple[int, int] | None:
    """Which axis to cut across and where, or None if nothing there is a wall.

    A run of blank is a wall if it is wide enough on its own, or — narrower than
    that — if the lettering either side of it is staggered, which says the two
    were never set as one block however close together they sit.

    The widest wall wins, so a block that comes apart several ways is cut at its
    plainest seam first and the rest is left to the recursion.
    """
    height, width = text.shape[:2]
    narrowest = max(GAP_MIN, round(GAP_STAGGERED * em))

    walls = []
    # A cut across x is a wall standing through the height, and the other way about.
    for axis, through in ((0, height / em), (1, width / em)):
        for start, run in blanks(text.any(axis=axis)):
            cut = start + run // 2
            if wide_enough(run, through, em) or (
                run >= narrowest and staggered(text, axis, cut, em)
            ):
                walls.append((run, axis, cut))

    if not walls:
        return None
    _, axis, cut = max(walls)
    return axis, cut


def parts(text: np.ndarray, em: float) -> list[Box]:
    """``text`` cut apart at every wide enough blank, boxed, in its own pixels."""
    found = where(text, em)
    if found is None:
        here = inked(text)
        return [here] if here else []

    axis, cut = found
    if axis == 0:
        return parts(text[:, :cut], em) + [
            box.moved(cut, 0) for box in parts(text[:, cut:], em)
        ]
    return parts(text[:cut, :], em) + [
        box.moved(0, cut) for box in parts(text[cut:, :], em)
    ]


def pieces(text: np.ndarray, box: Box) -> list[Box]:
    """``box`` as one box per balloon, or ``[box]`` where it only held one.

    ``text`` is the page-sized per-pixel text mask, and it must be the ungrown
    one: growing it to cover the halo around a letter also closes the gaps this
    is here to measure.

    A block that does not come apart is handed back exactly as it came in,
    untightened — this only ever answers differently for a block that was really
    two. The character size is read once, off the whole block, rather than again
    for each piece: half a bubble is a small sample, and the size does not change
    partway down a block.
    """
    crop = text[box.y0 : box.y1, box.x0 : box.x1]
    if crop.size == 0 or not crop.any():
        return [box]

    found = parts(crop, character(crop))
    if len(found) < 2:
        return [box]
    return [piece.moved(box.x0, box.y0) for piece in found]
