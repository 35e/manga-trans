"""Cutting a block the detector ran together back into one block per bubble.

**Not in the pipeline.** The detector boxes by region, so there is nothing left
to cut apart. Kept, with its tests, until enough real chapters have been through
the new detector to say it never runs two balloons together; delete it once that
is settled. The thresholds below came out of a grid search and are not cheap to
work out again — the measurements behind them are in DOCS.md.
"""

from __future__ import annotations

import cv2
import numpy as np

from .geometry import Box

GAP = 0.8

GAP_ALONE = 1.5

LINES = 2.5

STAGGER = 0.5

GAP_STAGGERED = 0.3

GAP_MIN = 8

CHARACTER = 75


def character(text: np.ndarray) -> float:
    """How large one character is, in pixels, read off the ink itself.

    Each mark is measured by its longer side, capped at the region's shorter one:
    without that cap, characters set solid enough to touch come back as one mark
    the length of the column they are in.
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
    """Every run of blank with ink on both sides, as (start, length)."""
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

    One side lying inside the other is not a stagger: that is a column that
    stopped early, or two centred against one another.
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

    ``through`` is how much lettering the cut would stand through, in characters.
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

    The widest wall wins, so a block that comes apart several ways is cut at its
    plainest seam first and the rest is left to the recursion.
    """
    height, width = text.shape[:2]
    narrowest = max(GAP_MIN, round(GAP_STAGGERED * em))

    walls = []
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

    ``text`` must be the *ungrown* per-pixel mask: growing it to cover the halo
    around a letter also closes the gaps this is here to measure.
    """
    crop = text[box.y0 : box.y1, box.x0 : box.x1]
    if crop.size == 0 or not crop.any():
        return [box]

    found = parts(crop, character(crop))
    if len(found) < 2:
        return [box]
    return [piece.moved(box.x0, box.y0) for piece in found]
