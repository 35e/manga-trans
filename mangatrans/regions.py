"""Finding the shapes text sits on: speech bubbles, caption boxes, signs.

The old filter asked "how white is it behind this text?", which cannot tell a
bubble from a sound effect painted on a pale panel. What actually separates the
two is *shape*: a bubble is a bounded, compact, flat island of paper that is
drawn around its lettering. So the page is segmented into regions first and the
text is matched to them afterwards, rather than the other way round.

A region is a connected run of page colour that

* does not touch the page edge - it is an island, not the background;
* is a plausible size for a bubble;
* is compact (its area is most of its convex hull), which rejects the ragged
  gaps between pieces of artwork;
* is *flat* - almost none of it is mid-grey. Bubble paper is paper; a highlight
  on a face or a sleeve carries shading and screentone;
* and contains text the detector actually found.

That last point is what does the heavy lifting. Everything before it is cheap
geometry that narrows the page down to a few dozen candidates; the detector then
confirms which of them are being used to hold dialogue.

Both polarities are segmented: pale regions hold the usual black-on-white
bubble, dark regions the inverted caption box that a flashback or a title page
tends to use.
"""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import Box

# Thresholds are fractions of the page's own white point rather than absolute
# levels, so a flat scan and a dim one are read the same way.
PALE_OF_WHITE = 0.80
INK_OF_WHITE = 0.43

# A bubble is drawn around its text, so it is never a speck and never the page.
MIN_AREA_FRACTION = 0.0003
MAX_AREA_FRACTION = 0.35
# Bubbles are convex-ish: a round one fills its hull entirely, a scalloped
# thought balloon or a spiky shout still keeps most of it. The gaps between
# pieces of artwork are stringy and fill far less.
MIN_SOLIDITY = 0.62
# Share of the region allowed to be neither paper nor ink. Lettering is ink on
# paper with a thin anti-aliased rim; shading and screentone are mid-grey.
MAX_MIDTONE = 0.12
# Anti-aliased outlines are a pixel or two of mid-grey that the page colour can
# seep through. Eroding by this much before labelling seals those leaks; the
# region is grown back afterwards.
SEAL_PX = 1
# Bands of artwork showing through a translucent bubble this thin are absorbed
# into it rather than splitting it up.
BRIDGE_PX = 3
# Share of the shape that must be the tone it was found by, rather than what it
# encloses. Lettering is a small part of a bubble; an outline is nearly all of a
# ring.
MIN_SOLID = 0.35
# Share of a fragment that must sit on a region before it counts as its text.
CONTAINS = 0.6


def _disc(radius: int):
    import cv2  # noqa: PLC0415

    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)


def thresholds(grey) -> tuple[int, int]:
    """``(pale, ink)`` cut-offs for this page, scaled to its own white point."""
    import numpy as np  # noqa: PLC0415

    white = float(np.percentile(grey, 92))
    white = min(255.0, max(120.0, white))
    return int(round(white * PALE_OF_WHITE)), int(round(white * INK_OF_WHITE))


@dataclass
class PageMasks:
    """The page reduced to paper / ink / everything else, computed once."""

    grey: object  # np.ndarray, uint8
    pale: object  # np.ndarray, bool - paper
    ink: object  # np.ndarray, bool - line art
    pale_threshold: int
    ink_threshold: int
    # Per-pixel lettering mask, when a detector produced one. Where it exists it
    # beats asking "what here is not paper?": it was trained to find lettering,
    # so it takes a glyph standing on a screentone and leaves the screentone,
    # and it stops at a bubble's drawn outline without being told where that is.
    text: object | None = None  # np.ndarray, uint8

    @property
    def shape(self) -> tuple[int, int]:
        return self.grey.shape[:2]

    @property
    def midtone(self):
        """Neither paper nor ink: screentone, shading, gradients."""
        return ~self.pale & ~self.ink


def page_masks(grey) -> PageMasks:
    pale_t, ink_t = thresholds(grey)
    return PageMasks(
        grey=grey,
        pale=grey >= pale_t,
        ink=grey <= ink_t,
        pale_threshold=pale_t,
        ink_threshold=ink_t,
    )


@dataclass
class Region:
    """A bubble, caption box or sign: a flat island of page colour.

    ``mask`` is the region's interior in ``box``-local coordinates, with the
    lettering filled back in, so it describes the shape the bubble would have if
    it were empty.
    """

    box: Box
    mask: object  # np.ndarray, uint8, shape == (box.h, box.w)
    polarity: str  # "light" (paper bubble) or "dark" (inverted caption)
    area: int
    solidity: float
    midtone: float

    def coverage(self, other: Box) -> float:
        """Share of ``other`` that lies on this region, 0-1."""
        import numpy as np  # noqa: PLC0415

        inter = self.box.intersection(other)
        if inter.w <= 0 or inter.h <= 0 or other.area == 0:
            return 0.0
        sub = self.mask[
            inter.y0 - self.box.y0 : inter.y1 - self.box.y0,
            inter.x0 - self.box.x0 : inter.x1 - self.box.x0,
        ]
        return float(np.count_nonzero(sub)) / other.area

    def interior(self, erode_px: int = 0):
        """Full-page-independent copy of the mask, optionally pulled in from the
        outline so a fill or a line of type never touches the drawn edge."""
        import cv2  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        if erode_px <= 0:
            return self.mask.copy()
        k = 2 * int(erode_px) + 1
        return cv2.erode(self.mask, np.ones((k, k), np.uint8))

    def to_dict(self) -> dict:
        return {
            "bbox": self.box.as_list(),
            "polarity": self.polarity,
            "solidity": round(self.solidity, 3),
            "midtone": round(self.midtone, 3),
        }


def _fill_holes(mask):
    """Close every hole in ``mask``.

    The holes are what the region encloses: the lettering, mostly. Filling them
    turns "the paper around the text" into "the inside of the bubble", which is
    the shape we want to measure, to repaint and to set type into.

    Filling *all* of them rather than only the letter-sized ones is what keeps
    this honest. Anything else a region encloses - a hand drawn over the bubble,
    a character standing in a blank panel - is filled in too and then counts
    against the flatness test below, so the region is rejected outright instead
    of being kept with a bite out of it. Being generous here and strict there
    beats guessing at a size cut-off, which gets small bubbles wrong: a sign
    whose lettering nearly fills it has "holes" as big as itself.
    """
    import cv2  # noqa: PLC0415

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return mask
    filled = mask.copy()
    for i, node in enumerate(hierarchy[0]):
        if node[3] != -1:  # has a parent, so it is a hole
            cv2.drawContours(filled, contours, i, 1, -1)
    return filled


def _segment(masks: PageMasks, backing, polarity: str, *, seal: int, limits: dict):
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    height, width = masks.shape
    page = height * width
    solid = backing.astype(np.uint8)
    kernel = np.ones((2 * seal + 1,) * 2, np.uint8) if seal else None
    labelled = cv2.erode(solid, kernel) if kernel is not None else solid

    count, labels, stats, _ = cv2.connectedComponentsWithStats(labelled, 4)
    midtone = masks.midtone
    regions: list[Region] = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if not limits["min_area"] * page <= area <= limits["max_area"] * page:
            continue
        if x <= 0 or y <= 0 or x + w >= width or y + h >= height:
            continue  # an island, not the page background

        comp = (labels[y : y + h, x : x + w] == i).astype(np.uint8)
        if kernel is not None:  # undo the seal, but only back onto real backing
            comp = cv2.dilate(comp, kernel) & solid[y : y + h, x : x + w]
        # Bubbles are not always opaque: one drawn over hatching or a gradient
        # lets it show through in bands, which breaks the paper into strips that
        # reach the bubble's edge and so are not holes to be filled. Closing
        # first absorbs bands that thin, leaving one shape to fill.
        comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, _disc(BRIDGE_PX))
        interior = _fill_holes(comp)

        contours, _ = cv2.findContours(
            interior, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(max(contours, key=cv2.contourArea)))
        filled_area = int(np.count_nonzero(interior))
        if hull_area <= 0:
            continue
        solidity = filled_area / hull_area
        if solidity < limits["min_solidity"]:
            continue

        # Most of the shape has to be the tone it was found by. A bubble is
        # paper with some lettering on it; its drawn *outline* is a closed ring
        # of ink that fills to the same ellipse, and without this it would be
        # offered as an inverted caption box every single time.
        if int(np.count_nonzero(comp)) < limits["min_solid"] * filled_area:
            continue

        selected = interior.astype(bool)
        mid = float(midtone[y : y + h, x : x + w][selected].mean())
        if mid > limits["max_midtone"]:
            continue

        regions.append(
            Region(
                box=Box(int(x), int(y), int(x + w), int(y + h)),
                mask=interior,
                polarity=polarity,
                area=filled_area,
                solidity=solidity,
                midtone=mid,
            )
        )
    return regions


def find_regions(
    masks: PageMasks,
    *,
    seal: int = SEAL_PX,
    min_area: float = MIN_AREA_FRACTION,
    max_area: float = MAX_AREA_FRACTION,
    min_solidity: float = MIN_SOLIDITY,
    max_midtone: float = MAX_MIDTONE,
    inverted: bool = True,
) -> list[Region]:
    """Every flat, bounded island on the page - bubble candidates, unconfirmed."""
    limits = {
        "min_area": min_area,
        "max_area": max_area,
        "min_solidity": min_solidity,
        "max_midtone": max_midtone,
        "min_solid": MIN_SOLID,
    }
    found = _segment(masks, masks.pale, "light", seal=seal, limits=limits)
    if inverted:
        found += _segment(masks, masks.ink, "dark", seal=seal, limits=limits)
    return found


def assign_regions(
    regions: list[Region], fragments: list[Box], contains: float = CONTAINS
) -> tuple[list[tuple[Region, list[int]]], list[int]]:
    """Match text fragments to the region each one sits on.

    Returns the regions that hold text, each with the indices of its fragments,
    and the indices of the fragments that sit on none of them. A fragment goes
    to the *smallest* region covering it, so a bubble always beats the pale
    panel it happens to be drawn on.
    """
    owner: dict[int, Region] = {}
    for index, fragment in enumerate(fragments):
        best: Region | None = None
        for region in regions:
            if region.area >= (best.area if best else float("inf")):
                continue
            if region.coverage(fragment) >= contains:
                best = region
        if best is not None:
            owner[index] = best

    held: list[tuple[Region, list[int]]] = []
    for region in regions:
        members = [i for i, r in owner.items() if r is region]
        if members:
            held.append((region, sorted(members)))
    free = [i for i in range(len(fragments)) if i not in owner]
    return held, free


def backing_of(masks: PageMasks, box: Box, glyph: int) -> tuple[str, float]:
    """What is behind free-standing text: ``("light"|"dark", plainness 0-1)``.

    Measured on the ring just outside ``box`` rather than inside it, so the
    lettering does not count as its own background. Plainness is the share of
    that ring which is one flat tone: narration lettered onto the page scores
    near 1, a sound effect painted over artwork scores low, because artwork is
    lines, gradients and screentone.
    """
    import numpy as np  # noqa: PLC0415

    height, width = masks.shape
    grown = box.padded(max(3, round(0.6 * glyph)), width, height)
    ring = np.ones((grown.h, grown.w), dtype=bool)
    ring[
        box.y0 - grown.y0 : box.y1 - grown.y0, box.x0 - grown.x0 : box.x1 - grown.x0
    ] = False
    if not ring.any():
        return "light", 0.0

    window = (slice(grown.y0, grown.y1), slice(grown.x0, grown.x1))
    pale_share = float(masks.pale[window][ring].mean())
    ink_share = float(masks.ink[window][ring].mean())
    if pale_share >= ink_share:
        return "light", pale_share
    return "dark", ink_share
