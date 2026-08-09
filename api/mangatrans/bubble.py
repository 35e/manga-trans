"""Finding the balloon a block of lettering sits in.

The detector boxes the *lettering*, and Japanese lettering runs down the page: a
line of a dozen characters comes back as a column forty pixels across and three
hundred tall. English set into a box that shape wraps to about a letter a line
and has to be set tiny to fit at all, so every translated line ends up dragged
out to its balloon by hand before it can be read. What a translation wants is
not the box the words came out of but the space they were written in — and that
space is nearly always wider than it is tall, which is what English needs.

Nothing draws that space, but it is not hard to see. A balloon is a light shape
closed by a dark outline, so the light pixels that can be reached from the
lettering without crossing anything dark *are* the balloon, and the largest
rectangle that fits inside them is where the translation goes. The one thing
that has to be done first is to paint the block itself in: Japanese down the
middle of a balloon cuts its ground into a left half and a right half, and a
flood started in one of them measures the gap beside the words rather than the
balloon around them.

It is a guess, and it says so. Plenty of lettering is in no balloon at all — a
sound effect over artwork, a caption in the margin — and a balloon whose outline
a scan has broken leaks into the page around it. Every answer is checked against
what a balloon ought to look like, and :func:`around` hands back ``None`` rather
than a wrong one, which leaves the caller with the box it already had.
"""

from __future__ import annotations

import cv2
import numpy as np

from .geometry import Box

# How far around a block to look, as a share of its longer side. Close first: a
# window reaching as far as the next balloon only gives a flood somewhere else
# to go. Wider only if what the close look found ran off the edge of it, which
# is what a balloon much bigger than the words in it does — two words in the
# middle of a large one, or a short line answering a long one.
MARGINS = (0.8, 2.2)

# The smallest a block is treated as, when the window is measured, as a share of
# the page's shorter side. A balloon around one character is not one character
# wide, and a window scaled to the block alone would be inside it.
REACH = 0.05

# A "balloon" covering more of the page than this got out through a broken
# outline, or was never a balloon.
LEAK = 0.25

# How much of the block the shape has to hold before it is believed to be the
# balloon that block is in. Short of this the flood is in some sliver beside the
# words, or the block overhangs the outline it sits against.
COVER = 0.85

# Left clear inside the outline, as a share of the balloon's shorter side, so
# the lettering does not come to rest against the line it is drawn inside.
INSET = 0.06

# The block is shrunk by this share of its shorter side before it is painted in.
# A detector box often takes in a little of the outline it sits against, and a
# seed laid across the outline bridges it — the flood is then outside the
# balloon before it has begun.
SHRINK = 0.12

# The rectangle is searched for on a grid no larger than this on its longer
# side. The search is the one thing here that is not a single OpenCV call, and a
# balloon does not need measuring to the pixel: a fraction of a percent of its
# width is well inside the margin left around it anyway.
GRID = 128

# How much of a full-size pixel has to be inside the balloon for its square on
# that grid to count as inside it. Well over half, because a rectangle that
# reaches past the outline is worse than one that stops short of it.
MOSTLY = 200


def moved(box: Box, dx: int, dy: int) -> Box:
    """The same box, somewhere else — a page's box in a window's pixels, or back."""
    return Box(box.x0 + dx, box.y0 + dy, box.x1 + dx, box.y1 + dy)


def window(box: Box, width: int, height: int, spread: float) -> Box:
    """The part of the page to look in for ``box``'s balloon."""
    reach = max(box.w, box.h, REACH * min(width, height))
    margin = max(8, round(spread * reach))
    return Box(
        box.x0 - margin, box.y0 - margin, box.x1 + margin, box.y1 + margin
    ).clipped(width, height)


def shrunk(box: Box) -> Box:
    """``box`` pulled in a little, to be painted in without crossing an outline."""
    inset = max(1, round(SHRINK * min(box.w, box.h)))
    pulled = Box(box.x0 + inset, box.y0 + inset, box.x1 - inset, box.y1 - inset)
    return pulled if pulled.w > 0 and pulled.h > 0 else box


def solid(region: np.ndarray) -> np.ndarray:
    """``region`` with everything it encloses counted as part of it.

    The lettering, and any tone or line inside the balloon, is a hole in the
    ground the flood spread over. A rectangle measured around those holes is the
    rectangle between two lines of text, so whatever the outside cannot reach is
    filled in. The border of blank added first is what makes "the outside"
    something there is always a corner of, even where the shape runs to the very
    edge of the window.
    """
    padded = cv2.copyMakeBorder(region, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    scratch = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(padded, scratch, (0, 0), 255)
    return cv2.bitwise_or(region, cv2.bitwise_not(padded[1:-1, 1:-1]))


def interior(ground: np.ndarray, seed: Box) -> np.ndarray:
    """The one piece of ``ground`` the seed sits in, holes and all.

    ``seed`` is painted in before anything is asked about the shape, which is
    what joins the two halves of a balloon a column of Japanese has cut in two.
    """
    joined = ground.copy()
    joined[seed.y0 : seed.y1, seed.x0 : seed.x1] = 255
    _, labels = cv2.connectedComponents(joined, connectivity=4)
    mine = labels == labels[int(seed.cy), int(seed.cx)]
    return solid(np.where(mine, 255, 0).astype(np.uint8))


def under(heights: list[int]) -> tuple[int, int, int, int]:
    """The largest rectangle standing on the baseline of one histogram.

    Its area, its left edge, its right edge and its height. The usual stack: a
    bar can only be closed off by a shorter one, so each waits there until one
    arrives and at that moment knows how far it could have run. The zero on the
    end is what closes off whatever is still standing when the row runs out.
    """
    best = (0, 0, 0, 0)
    stack: list[tuple[int, int]] = []
    for x, tall in enumerate([*heights, 0]):
        start = x
        while stack and stack[-1][1] >= tall:
            left, high = stack.pop()
            area = high * (x - left)
            if area > best[0]:
                best = (area, left, x, high)
            start = left
        if tall:
            stack.append((start, tall))
    return best


def standing(mask: np.ndarray) -> Box | None:
    """The largest axis-aligned rectangle of set pixels in ``mask``.

    Row by row, each column carrying how far up it has been set without a break:
    the largest rectangle whose bottom edge is on this row is then the largest
    rectangle under that histogram, and the largest anywhere has its bottom edge
    on some row.
    """
    height, width = mask.shape[:2]
    best_area, best = 0, None
    heights = np.zeros(width, np.int32)
    for y in range(height):
        heights = np.where(mask[y], heights + 1, 0)
        area, x0, x1, tall = under(heights.tolist())
        if area > best_area:
            best_area, best = area, Box(x0, y + 1 - tall, x1, y + 1)
    return best


def largest(mask: np.ndarray) -> Box | None:
    """:func:`standing`, measured on a coarse grid and scaled back up.

    Rounding is always inwards, so what comes back is inside the shape rather
    than nearly inside it.
    """
    height, width = mask.shape[:2]
    scale = min(1.0, GRID / max(height, width, 1))
    if scale == 1.0:
        return standing(mask > 0)

    grid = cv2.resize(
        mask,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    found = standing(grid >= MOSTLY)
    if found is None:
        return None
    return Box(
        int(np.ceil(found.x0 / scale)),
        int(np.ceil(found.y0 / scale)),
        int(found.x1 / scale),
        int(found.y1 / scale),
    )


def roomiest(region: np.ndarray) -> Box | None:
    """The largest rectangle inside ``region``, kept clear of its edge."""
    _, _, wide, tall = cv2.boundingRect(region)
    if wide < 4 or tall < 4:
        return None
    inset = max(1, round(INSET * min(wide, tall)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (inset * 2 + 1, inset * 2 + 1)
    )
    return largest(cv2.erode(region, kernel))


def stopped(region: np.ndarray, view: Box, width: int, height: int) -> bool:
    """Whether the shape came to a stop on its own rather than at the window.

    A shape running to an edge of the window was stopped by the window: either
    there was no outline there to stop it, or the balloon carries on past where
    we looked. The page's own edges do not count — a balloon can be drawn right
    up against them, and the window has nowhere further to go on that side.
    """
    edges = (
        (region[0].any(), view.y0 > 0),
        (region[-1].any(), view.y1 < height),
        (region[:, 0].any(), view.x0 > 0),
        (region[:, -1].any(), view.x1 < width),
    )
    return not any(reached and further for reached, further in edges)


def believable(
    region: np.ndarray, block: Box, view: Box, width: int, height: int
) -> bool:
    """Whether what was flooded looks like the balloon ``block`` is written in.

    Three things have to hold. It has to have stopped somewhere of its own
    accord. It has to hold the block, or the flood is in some sliver beside the
    words rather than in the room around them — which is also what a block
    overhanging the outline it sits against comes back as. And it has to be a
    balloon rather than a quarter of the page, which is what a flood that got out
    through a broken outline comes back as.
    """
    if not stopped(region, view, width, height):
        return False
    held = np.count_nonzero(region[block.y0 : block.y1, block.x0 : block.x1])
    if held < COVER * block.w * block.h:
        return False
    return np.count_nonzero(region) <= LEAK * width * height


def fitted(
    ground: np.ndarray, block: Box, view: Box, width: int, height: int
) -> Box | None:
    """The rectangle to letter in, in the window's own pixels."""
    region = interior(ground, shrunk(block))
    if not believable(region, block, view, width, height):
        return None
    return roomiest(region)


def around(grey: np.ndarray, box: Box) -> Box | None:
    """The balloon ``box`` sits in, as the largest rectangle that fits inside it.

    ``grey`` is the whole page in one channel. ``None`` says no balloon could be
    made out and the caller should keep the box it has, which is the right answer
    for a sound effect over artwork, for a caption in the margin, and for a
    balloon this cannot follow.

    A rectangle smaller than the block is not an answer either. It means the
    balloon is drawn no wider than the words already are — there is nothing to
    be won by moving them — or that the flood found something that was not the
    balloon at all.
    """
    height, width = grey.shape[:2]
    block = box.clipped(width, height)
    if block.w < 4 or block.h < 4:
        return None

    for spread in MARGINS:
        view = window(block, width, height, spread)
        patch = np.ascontiguousarray(grey[view.y0 : view.y1, view.x0 : view.x1])
        local = moved(block, -view.x0, -view.y0)
        # The threshold is taken from the window rather than the page: what
        # counts as the pale part of one corner of a page is not what counts as
        # the pale part of another.
        _, light = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Light ground first: nearly every balloon is dark words on a pale shape.
        # The other way round happens too — a shout set white on black — and
        # asking costs one more pass over a window a few hundred pixels across.
        for ground in (light, cv2.bitwise_not(light)):
            found = fitted(ground, local, view, width, height)
            if found is not None and found.w * found.h >= block.w * block.h:
                return moved(found, view.x0, view.y0).clipped(width, height)
    return None


def bubbles(image: np.ndarray, boxes: list[Box]) -> list[Box | None]:
    """The balloon each box sits in, in the order the boxes were given.

    ``None`` where none could be made out. The page is flattened to one channel
    once and every box is measured against that: colour says nothing about where
    a balloon ends that its own lightness does not.
    """
    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return [around(grey, box) for box in boxes]
