"""The room a block of lettering was written in, measured inside its balloon.

The detector says where the balloon is and where the lettering inside it is. The
balloon's box is not the answer on its own: a balloon is an oval, and a line set
to the corners of its bounding box runs outside the outline. So the inside of the
balloon is thresholded within the box it was found in, and the largest rectangle
in *that* which still holds the block is the room.

Around the block, not simply the largest rectangle in the shape. The words belong
where they were written, and all a balloon is asked is how much wider or taller
the room around them runs; a line set anywhere else is one the reader has to go
looking for. Everything downstream leans on this — ``lines.roomFor`` letters into
whatever comes back, and there is nothing else on the page saying where the words
belong.

``None`` is a real answer, and the right one for a sound effect over artwork: the
caller keeps the block's own box.
"""

from __future__ import annotations

import cv2
import numpy as np

from .geometry import Box

# How much of the block the balloon's inside has to hold to be believed. Short of
# this the threshold found some sliver beside the words rather than the ground
# they were set on, and the block's own box is the honest answer.
COVER = 0.85

# Left clear inside the outline, as a share of the balloon's shorter side.
INSET = 0.06

# The block is shrunk by this before being painted in: a detector box often takes
# in a little of the outline, and a seed laid across the outline bridges it.
SHRINK = 0.12

# The rectangle is searched for on a grid no larger than this on its longer side.
# The search is the one thing here that is not a single OpenCV call, and a
# balloon does not need measuring to the pixel.
GRID = 128

# How much of a full-size pixel must be inside the balloon for its square on that
# grid to count. Well over half: reaching past the outline is worse than stopping
# short of it.
MOSTLY = 200

# How much of a block a balloon has to hold to be the balloon it was written in.
# Well inside, or this is a balloon merely reaching over a sound effect beside it.
HELD = 0.85


def shrunk(box: Box) -> Box:
    """``box`` pulled in, to be painted in without crossing an outline."""
    inset = max(1, round(SHRINK * min(box.w, box.h)))
    pulled = Box(box.x0 + inset, box.y0 + inset, box.x1 - inset, box.y1 - inset)
    return pulled if pulled.w > 0 and pulled.h > 0 else box


def solid(region: np.ndarray) -> np.ndarray:
    """``region`` with everything it encloses counted as part of it.

    The lettering, and any tone inside the balloon, is a hole in the ground, and a
    rectangle measured around those holes is the rectangle between two lines of
    text. So whatever the outside cannot reach is filled in; the border of blank
    added first is what makes "the outside" something there is always a corner of.
    """
    padded = cv2.copyMakeBorder(region, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    scratch = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(padded, scratch, (0, 0), 255)
    return cv2.bitwise_or(region, cv2.bitwise_not(padded[1:-1, 1:-1]))


def joined(ground: np.ndarray, seed: Box) -> np.ndarray:
    """The one piece of ``ground`` the seed sits in, holes and all.

    ``seed`` is painted in first, which is what joins the two halves of a balloon
    that a column of Japanese down the middle has cut in two.
    """
    painted = ground.copy()
    painted[seed.y0 : seed.y1, seed.x0 : seed.x1] = 255
    _, labels = cv2.connectedComponents(painted, connectivity=4)
    mine = labels == labels[int(seed.cy), int(seed.cx)]
    return solid(np.where(mine, 255, 0).astype(np.uint8))


def run(strip: np.ndarray) -> np.ndarray:
    """How far the set pixels reach unbroken from the near edge of each row."""
    return np.cumprod(strip, axis=1).sum(axis=1).astype(np.int32)


def holding(mask: np.ndarray, block: Box) -> Box | None:
    """The largest rectangle of set pixels that holds the whole of ``block``.

    A row the block is not clear across closes the search off: there is no
    rectangle holding the block on the far side of one. Between them a rectangle
    is decided by which rows it spans, being as wide as the shortest reach past
    the block among them, so every span holding the block is measured.
    """
    height = mask.shape[0]
    if block.w <= 0 or block.h <= 0:
        return None
    across = mask[:, block.x0 : block.x1].all(axis=1)
    if not across[block.y0 : block.y1].all():
        return None

    top, bottom = block.y0, block.y1
    while top > 0 and across[top - 1]:
        top -= 1
    while bottom < height and across[bottom]:
        bottom += 1

    rows = mask[top:bottom]
    left = block.x0 - run(rows[:, : block.x0][:, ::-1])
    right = block.x1 + run(rows[:, block.x1 :])

    best, biggest = None, 0
    for y0 in range(top, block.y0 + 1):
        # Every rectangle starting on this row, by the row it ends on: no wider
        # than the narrowest reach among the rows it has taken in.
        near = np.maximum.accumulate(left[y0 - top :])
        far = np.minimum.accumulate(right[y0 - top :])
        # The first that reaches the foot of the block; above that it is no
        # rectangle of the block's.
        first = block.y1 - y0 - 1
        areas = (far[first:] - near[first:]) * np.arange(first + 1, len(near) + 1)
        pick = int(areas.argmax())
        if areas[pick] > biggest:
            biggest, at = int(areas[pick]), first + pick
            best = Box(int(near[at]), y0, int(far[at]), y0 + at + 1)
    return best


def largest(mask: np.ndarray, block: Box) -> Box | None:
    """:func:`holding`, measured on a coarse grid and scaled back up.

    The block is rounded outwards on the way in and the answer inwards on the
    way out, so what comes back is inside the shape rather than nearly inside
    it — and still holds every pixel of the block it was measured around.
    """
    height, width = mask.shape[:2]
    scale = min(1.0, GRID / max(height, width, 1))
    if scale == 1.0:
        return holding(mask > 0, block)

    grid = cv2.resize(
        mask,
        (max(1, int(np.ceil(width * scale))), max(1, int(np.ceil(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    found = holding(
        grid >= MOSTLY,
        Box(
            int(block.x0 * scale),
            int(block.y0 * scale),
            int(np.ceil(block.x1 * scale)),
            int(np.ceil(block.y1 * scale)),
        ),
    )
    if found is None:
        return None
    return Box(
        int(np.ceil(found.x0 / scale)),
        int(np.ceil(found.y0 / scale)),
        int(found.x1 / scale),
        int(found.y1 / scale),
    )


def roomiest(region: np.ndarray, block: Box) -> Box | None:
    """The largest rectangle in ``region`` holding ``block``, clear of its edge.

    The margin is the first thing given up. A balloon drawn tight around its
    words has none to spare, and neither has one whose outline the block is
    already up against: lettering set to the outline still beats lettering set
    beside the words.
    """
    _, _, wide, tall = cv2.boundingRect(region)
    if wide < 4 or tall < 4:
        return None
    inset = max(1, round(INSET * min(wide, tall)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (inset * 2 + 1, inset * 2 + 1)
    )
    return largest(cv2.erode(region, kernel), block) or largest(region, block)


def inside(grey: np.ndarray, balloon: Box, block: Box) -> Box | None:
    """The room inside ``balloon``, measured around ``block``.

    Both boxes are the page's. The balloon is thresholded in its own box rather
    than against the page: what counts as the pale part of one corner of a page
    is not what counts in another, and the balloon's box is exactly the
    neighbourhood the question is about.

    Light ground first — nearly every balloon is dark words on a pale shape —
    then the other way round, for a shout set white on black.
    """
    height, width = grey.shape[:2]
    view = balloon.clipped(width, height)
    held = block.clipped(width, height)
    if view.w < 4 or view.h < 4 or held.w < 4 or held.h < 4:
        return None

    patch = np.ascontiguousarray(grey[view.y0 : view.y1, view.x0 : view.x1])
    local = held.moved(-view.x0, -view.y0)
    # The block has to sit in the patch for any of this to mean anything; a
    # balloon that does not hold its own block is not the room it was written in.
    if local.x0 < 0 or local.y0 < 0 or local.x1 > view.w or local.y1 > view.h:
        return None

    _, light = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for ground in (light, cv2.bitwise_not(light)):
        region = joined(ground, shrunk(local))
        covered = np.count_nonzero(region[local.y0 : local.y1, local.x0 : local.x1])
        if covered < COVER * local.w * local.h:
            continue
        found = roomiest(region, local)
        if found is not None and found != local:
            return found.moved(view.x0, view.y0).clipped(width, height)
    return None


def cropped(box: Box, cell: Box) -> Box:
    """The part of ``box`` inside ``cell``, empty where the two do not meet.

    Here rather than on :class:`Box` because :mod:`mangatrans.geometry` is copied
    into the image before the models are baked in, and adding to it sends the
    next build back for half a gigabyte of weights.
    """
    x0, y0 = max(box.x0, cell.x0), max(box.y0, cell.y0)
    x1, y1 = min(box.x1, cell.x1), min(box.y1, cell.y1)
    return Box(x0, y0, max(x0, x1), max(y0, y1))


def within(room: Box, block: Box) -> float:
    """How much of ``block`` lies inside ``room``, 0 to 1.

    Not :meth:`Box.covers`, which divides by whichever of the two is smaller:
    the question here is only ever about the block.
    """
    wide = min(room.x1, block.x1) - max(room.x0, block.x0)
    tall = min(room.y1, block.y1) - max(room.y0, block.y0)
    if wide <= 0 or tall <= 0 or block.w <= 0 or block.h <= 0:
        return 0.0
    return wide * tall / (block.w * block.h)


def assigned(blocks: list[Box], balloons: list[Box]) -> list[int | None]:
    """Which balloon each block was written in, as an index into ``balloons``.

    The smallest balloon that holds the block, so a balloon drawn inside another
    — a shout within a thought — wins over the one around it. A block no balloon
    holds is lettering on the art: ``None``, and it keeps its own box.
    """
    found: list[int | None] = []
    for block in blocks:
        holds = [
            (balloon.w * balloon.h, at)
            for at, balloon in enumerate(balloons)
            if within(balloon, block) >= HELD
        ]
        found.append(min(holds)[1] if holds else None)
    return found


def span(box: Box, axis: int) -> tuple[int, int]:
    """Where a box starts and ends along one axis."""
    return (box.x0, box.x1) if axis == 0 else (box.y0, box.y1)


def cut(space: Box, axis: int, at: int) -> tuple[Box, Box]:
    """``space`` in two along one axis, the near side first."""
    low, high = span(space, axis)
    at = min(max(at, low + 1), high - 1)
    if axis == 0:
        near = Box(space.x0, space.y0, at, space.y1)
        return near, Box(at, space.y0, space.x1, space.y1)
    return Box(space.x0, space.y0, space.x1, at), Box(space.x0, at, space.x1, space.y1)


def seam(blocks: list[Box], among: list[int], axis: int) -> tuple[int, int, list[int]]:
    """Where a group of blocks comes apart along one axis.

    The widest blank between them, as (how wide, where to cut, which of them lie
    on the near side of it). A group that overlaps everywhere still comes back
    with its narrowest overlap, so there is always somewhere to cut.
    """
    order = sorted(among, key=lambda at: span(blocks[at], axis))
    best = (0, 0, order[:1])
    reach = span(blocks[order[0]], axis)[1]
    for edge in range(1, len(order)):
        low, high = span(blocks[order[edge]], axis)
        gap = low - reach
        if edge == 1 or gap > best[0]:
            best = (gap, (low + reach) // 2, order[:edge])
        reach = max(reach, high)
    return best


def divided(space: Box, blocks: list[Box]) -> list[Box]:
    """``space`` cut up between the blocks in it, one piece each.

    Two blocks in one balloon would otherwise be answered with that same balloon
    twice, and their translations set one on top of the other. They were told
    apart by the blank between them in the first place, so the space they share
    is cut the same way: at the widest blank between them, on whichever axis that
    blank is widest — and then each side again, until every block has a piece to
    itself.

    Cutting once along one axis is not enough. Four blocks set two across and two
    down are not in a row, and a single line of cuts hands the two on the right a
    left half and a right half of a balloon they are stacked inside.
    """
    shares: list[Box] = [space] * len(blocks)

    def share(part: Box, among: list[int]) -> None:
        if len(among) == 1:
            shares[among[0]] = part
            return
        (_, at, near), axis = max(
            ((seam(blocks, among, axis), axis) for axis in (0, 1)),
            key=lambda option: option[0][0],
        )
        first, second = cut(part, axis, at)
        rest = [held for held in among if held not in set(near)]
        share(first, near)
        share(second, rest)

    share(space, list(range(len(blocks))))
    return shares


def rooms(
    image: np.ndarray, blocks: list[Box], balloons: list[Box]
) -> list[Box | None]:
    """The room each block goes in, in the order the blocks were given.

    ``None`` where the block is in no balloon, or where its balloon turned out to
    have nothing more to offer than the block already has.

    Where several blocks share one balloon, each keeps *its own* answer cropped to
    its own cell rather than a share of the balloon handed round: an answer is
    always measured from the block it belongs to, so it is always around that
    block's words. Colour says nothing about where a balloon ends that its own
    lightness does not, so the page is flattened to one channel once.
    """
    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    height, width = grey.shape[:2]
    home = assigned(blocks, balloons)

    found: list[Box | None] = [
        None if at is None else inside(grey, balloons[at], block)
        for block, at in zip(blocks, home)
    ]

    page = Box(0, 0, width, height)
    for at in set(home) - {None}:
        group = [where for where, mine in enumerate(home) if mine == at]
        if len(group) < 2:
            continue
        held = [blocks[where] for where in group]
        for where, cell in zip(group, divided(page, held)):
            room = found[where]
            if room is None:
                continue
            share = cropped(room, cell)
            # A cell only cuts across a block where the blocks themselves
            # overlap, and there the box it came in with is the honest answer.
            found[where] = share if within(share, blocks[where]) >= HELD else None
    return found
