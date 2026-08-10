"""Finding the balloon a block of lettering sits in.

The detector boxes the *lettering*, and Japanese runs down the page: a dozen
characters come back as a column forty pixels across. English set in that shape
wraps to about a letter a line. What a translation wants is the space the words
were written in, which is nearly always wider than it is tall.

Nothing draws that space, but it is not hard to see: a balloon is a light shape
closed by a dark outline, so the light pixels reachable from the lettering
without crossing anything dark *are* the balloon, and the largest rectangle
inside them *around the block* is where the translation goes. Around the block,
not simply the largest: the words belong where they were written, and all a
balloon is asked is how much wider or taller the room around them runs. The
block itself has to be painted in first — Japanese down the middle of a balloon
cuts its ground in two, and a flood started in one half measures the gap beside
the words.

It is a guess and it says so. Plenty of lettering is in no balloon at all, and a
balloon whose outline a scan has broken leaks into the page. Every answer is
checked, and :func:`around` hands back ``None`` rather than a wrong one.
"""

from __future__ import annotations

import cv2
import numpy as np

from .geometry import Box

# How far around a block to look, as a share of its longer side. Close first: a
# window reaching the next balloon only gives the flood somewhere else to go.
# Each wider one is tried only where the last ran off the edge of its window,
# which is what a balloon far larger than the words in it does — one holding
# several separate lines, or two words in the middle of a big one.
MARGINS = (0.8, 2.2, 5.0)

# The smallest a block is treated as when the window is measured, as a share of
# the page's shorter side: a balloon around one character is wider than it.
REACH = 0.05

# A "balloon" covering more of the page than this got out through a broken
# outline, or was never a balloon.
LEAK = 0.25

# How much of the block the shape must hold to be believed. Short of this the
# flood is in some sliver beside the words.
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

# How much of a block a balloon has to hold before a block no balloon of its own
# could be made out for takes a share of it. Well inside, or this is not the room
# it was written in but a neighbour's reaching over it.
HELD = 0.85


def window(box: Box, width: int, height: int, spread: float) -> Box:
    """The part of the page to look in for ``box``'s balloon."""
    reach = max(box.w, box.h, REACH * min(width, height))
    margin = max(8, round(spread * reach))
    return Box(
        box.x0 - margin, box.y0 - margin, box.x1 + margin, box.y1 + margin
    ).clipped(width, height)


def shrunk(box: Box) -> Box:
    """``box`` pulled in, to be painted in without crossing an outline."""
    inset = max(1, round(SHRINK * min(box.w, box.h)))
    pulled = Box(box.x0 + inset, box.y0 + inset, box.x1 - inset, box.y1 - inset)
    return pulled if pulled.w > 0 and pulled.h > 0 else box


def solid(region: np.ndarray) -> np.ndarray:
    """``region`` with everything it encloses counted as part of it.

    The lettering, and any tone inside the balloon, is a hole in the ground the
    flood spread over, and a rectangle measured around those holes is the
    rectangle between two lines of text. So whatever the outside cannot reach is
    filled in; the border of blank added first is what makes "the outside"
    something there is always a corner of.
    """
    padded = cv2.copyMakeBorder(region, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    scratch = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(padded, scratch, (0, 0), 255)
    return cv2.bitwise_or(region, cv2.bitwise_not(padded[1:-1, 1:-1]))


def interior(ground: np.ndarray, seed: Box) -> np.ndarray:
    """The one piece of ``ground`` the seed sits in, holes and all.

    ``seed`` is painted in first, which is what joins the two halves of a balloon
    a column of Japanese has cut in two.
    """
    joined = ground.copy()
    joined[seed.y0 : seed.y1, seed.x0 : seed.x1] = 255
    _, labels = cv2.connectedComponents(joined, connectivity=4)
    mine = labels == labels[int(seed.cy), int(seed.cx)]
    return solid(np.where(mine, 255, 0).astype(np.uint8))


def run(strip: np.ndarray) -> np.ndarray:
    """How far the set pixels reach unbroken from the near edge of each row."""
    return np.cumprod(strip, axis=1).sum(axis=1).astype(np.int32)


def holding(mask: np.ndarray, block: Box) -> Box | None:
    """The largest rectangle of set pixels that holds the whole of ``block``.

    Not the largest rectangle in the shape, which is somewhere else in it as
    often as not: a balloon with a tail, one drawn round two lines with the
    words in one of them, one whose outline a scan has broken into the panel
    beside it. A translation belongs where its Japanese was — the balloon is
    only being asked how much further the words can be opened out — and a line
    set anywhere else is one the reader has to go looking for.

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


def stopped(region: np.ndarray, view: Box, width: int, height: int) -> bool:
    """Whether the shape came to a stop on its own rather than at the window.

    A shape running to an edge of the window was stopped by the window. The
    page's own edges do not count: a balloon can be drawn against them, and the
    window has nowhere further to go on that side.
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

    It has to have stopped of its own accord; it has to hold the block, or the
    flood is in some sliver beside the words; and it has to be a balloon rather
    than a quarter of the page, which is what leaking through a broken outline
    comes back as.
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
    return roomiest(region, block)


def around(grey: np.ndarray, box: Box) -> Box | None:
    """The balloon ``box`` sits in, as the largest rectangle around it that fits.

    ``grey`` is the whole page in one channel. ``None`` says no balloon could be
    made out and the caller should keep the box it has — the right answer for a
    sound effect over artwork, and for a balloon this cannot follow.

    What comes back always holds the block: the words stay where they were
    written and the answer only says how much wider or taller the room around
    them is. So the block itself is no answer either — a balloon drawn no wider
    than the words already are has nothing to offer.
    """
    height, width = grey.shape[:2]
    block = box.clipped(width, height)
    if block.w < 4 or block.h < 4:
        return None

    for spread in MARGINS:
        view = window(block, width, height, spread)
        patch = np.ascontiguousarray(grey[view.y0 : view.y1, view.x0 : view.x1])
        local = block.moved(-view.x0, -view.y0)
        # Thresholded on the window rather than the page: what counts as the pale
        # part of one corner of a page is not what counts in another.
        _, light = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Light ground first — nearly every balloon is dark words on a pale shape
        # — then the other way round, for a shout set white on black.
        for ground in (light, cv2.bitwise_not(light)):
            found = fitted(ground, local, view, width, height)
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


def sharing(rooms: list[Box]) -> list[list[int]]:
    """Which blocks would be lettered one on top of the other, gathered together.

    Whether two answers *collide*, not whether they agree. Agreement cannot be
    asked for: an oval holds a wide short rectangle and a tall narrow one of
    nearly the same area, and a pixel of the flood decides which of them wins, so
    one balloon measured from two of the blocks in it comes back half a balloon
    apart — under any threshold for "the same balloon twice", and lettered one
    over the other.

    ``rooms`` is where each block goes if nothing is done, so a block no balloon
    could be made out for is in here as its own box: a balloon reaching over a
    neighbour it cannot see is the same bug from the other side.

    Transitively, because A over B and B over C is one balloon holding three
    blocks however little A and C themselves touch.
    """
    groups: list[list[int]] = []
    for at, room in enumerate(rooms):
        joined, apart = [at], []
        for group in groups:
            if any(room.covers(rooms[other]) > 0 for other in group):
                joined.extend(group)
            else:
                apart.append(group)
        groups = [*apart, sorted(joined)]
    return groups


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


def borrowed(answers: list[Box], block: Box) -> Box | None:
    """A neighbour's balloon, for a block no balloon of its own was made out for.

    :func:`around` fails on plenty of lettering that is in a balloon all the
    same, and a neighbour written in that balloon flooded it successfully. The
    balloon has to hold the block to be lent, or this is not the room the block
    was written in but a neighbour's reaching over it — a sound effect beside a
    balloon is in none, and moving it into one would take the words off the art
    they belong to.
    """
    room = max(answers, key=lambda box: within(box, block), default=None)
    return room if room is not None and within(room, block) >= HELD else None


def bubbles(image: np.ndarray, boxes: list[Box]) -> list[Box | None]:
    """The balloon each box sits in, in the order the boxes were given.

    ``None`` where none could be made out. Colour says nothing about where a
    balloon ends that its own lightness does not, so the page is flattened to one
    channel once and every box measured against that.

    Where several blocks turn out to be written in the same balloon — which is
    what a block cut in two by :mod:`mangatrans.split` looks like from here —
    each is cut back to its own side of the blank between them rather than handed
    the balloon whole. No two answers may overlap when this is done: two that do
    are two translations set one on top of the other, and that is what the
    cutting is for.
    """
    height, width = image.shape[:2]
    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    found = [around(grey, box) for box in boxes]
    # Where a block would be lettered as things stand: its balloon, or its own
    # box where none was made out. Two of these overlapping is the whole bug.
    rooms = [box if room is None else room for room, box in zip(found, boxes)]

    page = Box(0, 0, width, height)
    for group in sharing(rooms):
        if len(group) < 2:
            continue
        held = [boxes[at] for at in group]
        answers = [found[at] for at in group if found[at] is not None]
        # The page cut into a cell per block, along the blanks that told the
        # blocks apart in the first place. Each block keeps its own balloon,
        # cropped to its own cell: that parts two answers while leaving each of
        # them round the words it was measured from. Cutting one balloon up
        # between the whole group instead is what sends the odd one out — the
        # block in a different balloon that merely touched this one — to a piece
        # of a balloon its words are nowhere near.
        for at, cell in zip(group, divided(page, held)):
            block = boxes[at]
            room = found[at] or borrowed(answers, block)
            share = None if room is None else cropped(room, cell)
            # A cell only cuts across a block where the blocks themselves
            # overlap, and there the box it came in with is the honest answer.
            held_by = share is not None and within(share, block) >= HELD
            found[at] = share if held_by else None
    return found
