"""Boxes, distances and the proximity clustering used to build text groups."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in pixel coordinates, ``x1``/``y1`` exclusive."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def glyph_size(self) -> int:
        """Short side of the box ~ font size (char height for a horizontal run,
        char width for a vertical column)."""
        return max(1, min(self.w, self.h))

    def padded(self, px: float, width: int, height: int) -> "Box":
        px = int(round(px))
        return Box(
            max(0, self.x0 - px),
            max(0, self.y0 - px),
            min(width, self.x1 + px),
            min(height, self.y1 + px),
        )

    def clipped(self, width: int, height: int) -> "Box":
        """Confine the box to the image; the detector may overshoot its edges."""
        return Box(
            min(max(0, self.x0), width),
            min(max(0, self.y0), height),
            min(max(0, self.x1), width),
            min(max(0, self.y1), height),
        )

    def intersection(self, other: "Box") -> "Box":
        return Box(
            max(self.x0, other.x0),
            max(self.y0, other.y0),
            min(self.x1, other.x1),
            min(self.y1, other.y1),
        )

    def overlaps(self, other: "Box") -> bool:
        return (
            self.x0 < other.x1
            and other.x0 < self.x1
            and self.y0 < other.y1
            and other.y0 < self.y1
        )

    def as_list(self) -> list[int]:
        return [self.x0, self.y0, self.x1, self.y1]


def union_box(boxes: list[Box]) -> Box:
    return Box(
        min(b.x0 for b in boxes),
        min(b.y0 for b in boxes),
        max(b.x1 for b in boxes),
        max(b.y1 for b in boxes),
    )


def box_gap(a: Box, b: Box) -> float:
    """Edge-to-edge distance between two boxes (0 when they touch or overlap)."""
    dx = max(0, max(a.x0, b.x0) - min(a.x1, b.x1))
    dy = max(0, max(a.y0, b.y0) - min(a.y1, b.y1))
    return math.hypot(dx, dy)


def overlaps_any(box: Box, others: list[Box]) -> bool:
    return any(box.overlaps(o) for o in others)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[rj] = ri


# How much two fragments must line up along an axis before they count as parts
# of one column (or one line) rather than as neighbours that merely happen to be
# close.
ALIGNMENT = 0.45
# Fragments this close belong together whatever their shapes say.
TOUCHING = 0.25


def _overlap(a0: int, a1: int, b0: int, b1: int, span: int) -> float:
    return max(0, min(a1, b1) - max(a0, b0)) / max(1, span)


def joins(a: Box, b: Box, gap: float, stack: float) -> bool:
    """Are these two fragments part of the same block of text?

    Japanese runs in columns, and the two ways a block continues are not
    interchangeable. *Along* a column the type may break for a beat - a short
    line, an ellipsis, a trailing "?" - and still be the same sentence, so a
    wide gap is expected there. *Across* columns the spacing is set by the
    typesetting and is tight; a wide gap means a different block, which is
    precisely what happens when two bubbles are drawn overlapping and share one
    blob of paper.

    Judging the two directions with the same number is what made a single
    bubble split while a pair of bubbles merged. So the axis the fragments line
    up on decides which limit applies: ``stack`` when one continues the other,
    ``gap`` when they sit side by side. (Horizontal text works out the same way
    with the roles of the axes swapped.)
    """
    distance = box_gap(a, b)
    glyph = min(a.glyph_size, b.glyph_size)
    # Practically adjacent: join whatever the shapes say, but never past the
    # limits themselves - those are the caller's last word.
    if distance <= min(TOUCHING, gap, stack) * glyph:
        return True
    along_x = _overlap(a.x0, a.x1, b.x0, b.x1, min(a.w, b.w))
    along_y = _overlap(a.y0, a.y1, b.y0, b.y1, min(a.h, b.h))
    if max(along_x, along_y) < ALIGNMENT:
        return False  # neither continues the other; two different blocks
    limit = stack if along_x >= along_y else gap
    return distance <= limit * glyph


def group_boxes(
    boxes: list[Box],
    gap_factor: float = 1.2,
    stack_factor: float = 3.0,
    min_gap_px: float = 0.0,
    max_gap_px: float | None = None,
) -> list[list[int]]:
    """Cluster boxes into blocks of text; returns lists of indices into ``boxes``.

    Thresholds are multiples of the glyph size of the smaller box, clamped to
    [``min_gap_px``, ``max_gap_px``] in absolute pixels. Clustering is
    single-linkage, so a chain of neighbouring fragments forms one block.
    """
    uf = _UnionFind(len(boxes))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            # Clamping is given in pixels, so convert it into the multiples of
            # glyph size the rule itself works in.
            glyph = min(a.glyph_size, b.glyph_size)
            lo = min_gap_px / glyph
            hi = (max_gap_px / glyph) if max_gap_px is not None else float("inf")
            gap = min(max(gap_factor, lo), hi)
            stack = min(max(stack_factor, lo), hi)
            if joins(a, b, gap, stack):
                uf.union(i, j)

    clusters: dict[int, list[int]] = {}
    for idx in range(len(boxes)):
        clusters.setdefault(uf.find(idx), []).append(idx)
    return list(clusters.values())


def sort_reading_order(groups: list, order: str = "rtl") -> list:
    """Best-effort reading order for objects exposing a ``bbox``.

    ``rtl`` (default, normal for Japanese manga): top to bottom, right to left.
    Groups whose vertical starts fall within a row tolerance count as one row.
    """
    if order == "none" or not groups:
        return groups

    row_tol = max(1.0, 0.6 * statistics.median([g.bbox.h for g in groups]))
    sign = 1 if order == "ltr" else -1
    return sorted(groups, key=lambda g: (round(g.bbox.y0 / row_tol), sign * g.bbox.cx))
