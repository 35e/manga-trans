"""Setting the translation into the space the original text came out of.

A Japanese bubble is tall and narrow because the text runs in vertical columns;
English wants the opposite shape. The old code guessed at that by stretching the
*text's* bounding box into a few aspect ratios and hoping the bubble around it
had room. Now that the bubble itself is known, the type is fitted to the bubble:
for a handful of aspect ratios we take the largest rectangle of that shape that
fits wholly inside it, and keep whichever one lets the type be biggest. Two
utterances sharing one blob of paper get half of it each rather than both
claiming all of it.

Text with no bubble behind it falls back to reshaping its own footprint, held to
the plain paper it was lettered on so a wide line cannot wander onto the artwork.
"""

from __future__ import annotations

import functools

from .geometry import Box, overlaps_any
from .pipeline import ART, TextGroup

# The slim base image has no fonts; fonts-dejavu-core provides this one. Falls
# back to Pillow's built-in font so the script still runs outside the container.
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
)

FONT_MIN = 9
# Aspect ratios (width / height) tried when looking for room inside a bubble.
ASPECTS = (0.7, 1.0, 1.4, 1.9, 2.6, 3.4)
# Same idea for text with no bubble: widen by k, shorten by k, so every
# candidate covers about the area the original lettering already occupied.
SHAPE_SCALES = (1.0, 1.25, 1.5, 2.0, 2.5)
# Detected boxes hug the glyphs, so there is a little slack around them.
AREA_BONUS = 1.1
# A reshaped block has to stay this much on the plain paper the original was
# lettered on; below that it is reaching onto the artwork.
ON_BACKING = 0.97
# Latin type reads larger than Japanese at the same nominal size - kana fill
# their em, Latin lower case does not - so matching the original glyph size
# already gives slightly bigger-looking lettering. This is the ceiling.
MAX_FONT_PER_GLYPH = 1.15
# Keep the type off the drawn edge of the bubble.
BUBBLE_INSET = 0.06


@functools.lru_cache(maxsize=256)
def load_font(path: str | None, size: int):
    from PIL import ImageFont  # noqa: PLC0415

    for candidate in [path] if path else list(FONT_CANDIDATES):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    if path:
        raise SystemExit(f"could not load font: {path}")
    return ImageFont.load_default(size=size)


def _break_word(word: str, font, max_width: float) -> list[str]:
    """Split a word too long for a line of its own, hyphenating."""
    parts, current = [], ""
    for char in word:
        if current and font.getlength(f"{current}{char}-") > max_width:
            parts.append(f"{current}-")
            current = char
        else:
            current += char
    if current:
        parts.append(current)
    return parts or [word]


def wrap_text(text: str, font, max_width: float, hyphenate: bool = False) -> list[str]:
    """Greedy word wrap measured with the font itself.

    ``hyphenate`` is off by default and deliberately so: a word allowed to break
    always "fits", so the size search would answer every question with a bigger
    font and a line of confetti. Leaving the long word to overflow instead lets
    the caller see that the size is too big and come down. Only a word that
    cannot fit the width at any size gets broken.
    """
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}" if current else word
        if current and font.getlength(candidate) > max_width:
            lines.append(current)
            current, candidate = "", word
        if hyphenate and not current and font.getlength(candidate) > max_width:
            pieces = _break_word(candidate, font, max_width)
            lines.extend(pieces[:-1])
            current = pieces[-1]
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _measure(draw, lines: list[str], font, spacing: float):
    block = "\n".join(lines)
    left, top, right, bottom = draw.multiline_textbbox(
        (0, 0), block, font=font, spacing=spacing, align="center"
    )
    return block, right - left, bottom - top


def fit_text(draw, text: str, box_w: int, box_h: int, args, max_size: int):
    """Largest font size at which ``text`` wraps inside ``box_w`` x ``box_h``.

    Returns ``(font, lines, spacing)``, or ``None`` if it will not fit at all.
    Words are only hyphenated when nothing fits without it.
    """
    for hyphenate in (False, True):
        lo, hi, best = FONT_MIN, max(FONT_MIN, min(box_h, max_size)), None
        while lo <= hi:
            size = (lo + hi) // 2
            font = load_font(args.font, size)
            lines = wrap_text(text, font, box_w, hyphenate)
            if not lines:
                return None
            spacing = size * args.line_spacing
            _, width, height = _measure(draw, lines, font, spacing)
            if width <= box_w and height <= box_h:
                best = (font, lines, spacing)
                lo = size + 1
            else:
                hi = size - 1
        if best is not None:
            return best
    return None


# ---------------------------------------------------------------------------
# finding room
# ---------------------------------------------------------------------------


def _integral_of_holes(mask):
    """Summed-area table of everything *outside* the mask.

    A rectangle lies wholly inside the mask exactly when its sum here is zero,
    which makes each fit test a handful of additions.
    """
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    return cv2.integral((mask == 0).astype(np.uint8))


def _inside(integral, x0: int, y0: int, x1: int, y1: int) -> bool:
    return (
        integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    ) == 0


def inscribed_rect(mask, centre: tuple[int, int], aspect: float) -> Box | None:
    """Largest rectangle of ratio ``aspect`` centred on ``centre`` inside ``mask``.

    Coordinates are local to ``mask``. Grown by binary search on the height,
    which is exact for the convex-ish shapes a bubble actually has.
    """
    height, width = mask.shape[:2]
    cx, cy = centre
    if not (0 <= cx < width and 0 <= cy < height) or not mask[cy, cx]:
        return None
    integral = _integral_of_holes(mask)

    def fits(h: int) -> Box | None:
        w = max(1, round(h * aspect))
        x0, y0 = cx - w // 2, cy - h // 2
        x1, y1 = x0 + w, y0 + h
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            return None
        return Box(x0, y0, x1, y1) if _inside(integral, x0, y0, x1, y1) else None

    lo, hi, best = 1, min(height, max(1, round(width / max(aspect, 1e-3)))), None
    while lo <= hi:
        mid = (lo + hi) // 2
        found = fits(mid)
        if found is not None:
            best, lo = found, mid + 1
        else:
            hi = mid - 1
    return best


def _deepest_point(mask) -> tuple[int, int]:
    """The point furthest from the edge of the mask - the middle of a bubble."""
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    index = int(np.argmax(distance))
    return index % mask.shape[1], index // mask.shape[1]


def _distance_to(box: Box, xs, ys):
    import numpy as np  # noqa: PLC0415

    dx = np.maximum(np.maximum(box.x0 - xs, xs - box.x1), 0)
    dy = np.maximum(np.maximum(box.y0 - ys, ys - box.y1), 0)
    return dx * dx + dy * dy


def region_slots(region, members: list[TextGroup]) -> dict[int, object]:
    """Divide a region's interior between the groups sitting in it.

    One bubble to one utterance is the usual case and gets the whole interior.
    When bubbles were drawn overlapping they share a blob of paper, and each
    utterance takes the part of it nearest to where it was originally set -
    otherwise both would be lettered across the whole shape, on top of each
    other.
    """
    import numpy as np  # noqa: PLC0415

    inset = max(2, round(BUBBLE_INSET * min(region.box.w, region.box.h)))
    interior = region.interior(inset)
    if len(members) == 1:
        return {id(members[0]): interior}

    ys, xs = np.mgrid[
        region.box.y0 : region.box.y1, region.box.x0 : region.box.x1
    ]
    distances = np.stack([_distance_to(g.bbox, xs, ys) for g in members])
    nearest = distances.argmin(axis=0)
    return {
        id(group): (interior & (nearest == i).astype(interior.dtype))
        for i, group in enumerate(members)
    }


def bubble_boxes(region, slot, size: tuple[int, int]) -> list[Box]:
    """Candidate lettering areas inside a bubble, in page coordinates."""
    import numpy as np  # noqa: PLC0415

    if slot is None or not np.any(slot):
        return []
    centre = _deepest_point(slot)
    boxes = []
    for aspect in ASPECTS:
        rect = inscribed_rect(slot, centre, aspect)
        if rect is None or rect.w < FONT_MIN or rect.h < FONT_MIN:
            continue
        boxes.append(
            Box(
                rect.x0 + region.box.x0,
                rect.y0 + region.box.y0,
                rect.x1 + region.box.x0,
                rect.y1 + region.box.y0,
            )
        )
    return boxes


def _plain_backing(masks, group: TextGroup, window: Box):
    """``window`` of the page as "plain paper or not".

    Read from the page *after* the original lettering has been erased, so the
    holes the old text punched in the paper are already gone and what is left is
    the background the translation will actually sit on.
    """
    base = masks.pale if group.polarity == "light" else masks.ink
    return base[window.y0 : window.y1, window.x0 : window.x1]


def footprint_boxes(
    group: TextGroup, size: tuple[int, int], others: list[Box], masks
) -> list[Box]:
    """Candidate areas for text with no bubble: its own footprint, reshaped.

    Every candidate holds roughly the area the original lettering did, laid out
    the other way round. A candidate is dropped if it would reach into a
    neighbouring block or off the plain paper the original sat on; the text's own
    footprint is always kept as the fallback.
    """
    import numpy as np  # noqa: PLC0415

    image_w, image_h = size
    box = group.bbox
    shapes = []
    for scale in SHAPE_SCALES:
        box_w = min(round(box.w * scale * AREA_BONUS), image_w)
        box_h = min(round(box.h / scale * AREA_BONUS), image_h)
        if box_h >= FONT_MIN and box_w >= FONT_MIN:
            shapes.append(
                Box(
                    round(box.cx - box_w / 2),
                    round(box.cy - box_h / 2),
                    round(box.cx + box_w / 2),
                    round(box.cy + box_h / 2),
                ).clipped(image_w, image_h)
            )
    if not shapes:
        return []

    window = Box(
        min(s.x0 for s in shapes),
        min(s.y0 for s in shapes),
        max(s.x1 for s in shapes),
        max(s.y1 for s in shapes),
    )
    backing = _plain_backing(masks, group, window)

    boxes = []
    for i, shape in enumerate(shapes):
        if i == 0:  # its own footprint is always allowed
            boxes.append(shape)
            continue
        if overlaps_any(shape, others):
            continue
        patch = backing[
            shape.y0 - window.y0 : shape.y1 - window.y0,
            shape.x0 - window.x0 : shape.x1 - window.x0,
        ]
        if patch.size and np.count_nonzero(patch) / patch.size >= ON_BACKING:
            boxes.append(shape)
    return boxes


def layout(draw, group: TextGroup, text: str, size, others, args, slot, masks):
    """Best ``(box, font, lines, spacing)`` for ``text``, or ``None``."""
    max_size = max(FONT_MIN, round(group.glyph_size * MAX_FONT_PER_GLYPH))
    candidates = bubble_boxes(group.region, slot, size) if group.region else []
    candidates = candidates or footprint_boxes(group, size, others, masks)

    best = None
    for box in candidates:
        fitted = fit_text(draw, text, box.w, box.h, args, max_size)
        if fitted and (best is None or fitted[0].size > best[1].size):
            best = (box, *fitted)
    return best


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------


def render_page(path, page, out_path, masks, args) -> int:
    """Write a copy of the page with each bubble's translation lettered in.

    The original lettering is erased and the translation set into the space it
    came out of, at the size the original was lettered at. Groups without a
    translation are left exactly as they were.
    """
    import numpy as np  # noqa: PLC0415
    from PIL import Image, ImageDraw, ImageOps  # noqa: PLC0415

    from .erase import erase_text  # noqa: PLC0415
    from .regions import page_masks  # noqa: PLC0415

    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    size = image.size

    lettered = [g for g in page.groups if g.translation.strip()]
    image = erase_text(image, lettered, masks, args.erase_pad)
    # Where the type may go is decided against the cleaned page, not the
    # original: the space a bubble offers is the space it has once emptied.
    cleaned = page_masks(np.array(image.convert("L")))
    draw = ImageDraw.Draw(image)

    sharing: dict[int, list[TextGroup]] = {}
    for group in lettered:
        if group.region is not None:
            sharing.setdefault(id(group.region), []).append(group)
    slots: dict[int, object] = {}
    for members in sharing.values():
        slots.update(region_slots(members[0].region, members))

    rendered = 0
    for group in lettered:
        others = [g.bbox for g in page.groups if g is not group]
        fitted = layout(
            draw,
            group,
            group.translation.strip(),
            size,
            others,
            args,
            slots.get(id(group)),
            cleaned,
        )
        if not fitted:
            continue
        box, font, lines, spacing = fitted
        block, width, height = _measure(draw, lines, font, spacing)

        left, top, _, _ = draw.multiline_textbbox(
            (0, 0), block, font=font, spacing=spacing, align="center"
        )
        x = box.cx - width / 2 - left
        y = box.cy - height / 2 - top

        # Only text left standing on artwork needs a halo to stay readable; on a
        # bubble the background is flat and an outline just looks furry.
        stroke = max(1, font.size // 14) if group.kind == ART else 0
        draw.multiline_text(
            (x, y),
            block,
            font=font,
            fill=args.text_colour,
            spacing=spacing,
            align="center",
            stroke_width=stroke,
            stroke_fill=args.halo_colour,
        )
        rendered += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(out_path, quality=92, subsampling=0)
    else:
        image.save(out_path)
    return rendered
