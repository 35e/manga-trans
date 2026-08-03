"""Taking the original lettering off the page.

The old version dilated each detected fragment into a rectangle and inpainted
the lot. Rectangles are much bigger than the strokes inside them, and Telea
inpainting fills a large hole by smearing its rim inwards, so a bubble came back
blotchy and text on artwork left a soft grey scar.

Two things fix that.

The first is to stop guessing wherever the answer is already there. Under a
bubble's lettering is the bubble, and what a bubble looks like without its text
can be *measured*: a greyscale closing with a brush wider than a character wipes
the strokes out of the picture and leaves the surface they were sitting on -
flat white for an opaque bubble, and the right shade of grey for one the artwork
shows through or that carries a wash. Painting that surface back over the text
restores the bubble rather than approximating it, and it does so without needing
the bubble to be one flat colour.

The second is that where a guess really is needed - text standing directly on
artwork, with lines running under it - only the *strokes* are handed to the
inpainter, not the boxes around them. That is a few percent of the area the old
mask covered, so what was a smudge becomes a repair.

Where the page is going to be re-lettered by hand rather than by ``--render``,
what is wanted instead is a flat patch of white sitting exactly over the
Japanese and nowhere else. ``TIGHT`` builds that. It does not trust the
detector's boxes to say where the ink is: inside a bubble the region is known,
so its outline can be excluded and *everything else on that paper* near the
group's text taken as lettering - which picks up the furigana that is far too
small to have been boxed in the first place. The result is closed up so a column
of type is covered by one solid patch instead of a constellation of
glyph-shaped holes, and clipped back inside the bubble so the white can never
cross the drawn outline.
"""

from __future__ import annotations

from .geometry import Box
from .pipeline import ART, TextGroup
from .regions import PageMasks

# Anti-aliasing leaves a rim of half-toned pixels around every glyph that is
# neither ink nor paper; without a bleed it survives as a grey ghost.
BLEED_GLYPHS = 0.12
MIN_BLEED_PX = 2
# How far past its own text a group is looked at, in glyph sizes. The detector
# boxes the body of an utterance and leaves the trimmings - furigana set beside
# a kanji, a leading ellipsis, a trailing "?" that got a column to itself - so
# the cover-up has to reach further than the boxes do. Inside a bubble that is
# nearly free: the reach only says where to look, and there is nothing there to
# find but the lettering.
WIPE_GLYPHS = 1.0
# The brush that lifts the lettering off the background, in glyph sizes. It has
# to be wider than a character or the character survives as a smear.
BACKGROUND_BRUSH = 1.4
MAX_BRUSH_PX = 80
# Telea reaches this far in from the edge of the hole. Small on purpose: a large
# radius visibly smears line art.
INPAINT_RADIUS = 3
# Keep the drawn outline of a bubble out of everything we paint.
OUTLINE_GUARD_PX = 2
# The gaps inside a character, and between the characters of a column, are
# closed so a line of type comes out as one solid patch rather than as separate
# glyph-shaped holes. In glyph sizes: a closing of radius r bridges a gap of
# about 2r, so this reaches half a character - across the counter of a kana and
# between two characters set in a column, and no further.
KNIT_GLYPHS = 0.25
# A dot screen is not paper either, so "everything that is not paper" picks the
# whole field of it up, and knitting then welds the dots into a slab that has
# nothing to do with the text. Size is what tells them apart: a tone dot is a
# speck a fraction of a character across and a stroke is not. In glyph areas.
SPECK_GLYPHS = 0.01

# How far the paint spreads, for the modes that do not measure it.
AUTO = "auto"  # what --render wants: the surface the text was sitting on
TIGHT = "text"  # a patch that hugs the lettering and stops there
WHOLE = "bubble"  # the whole inside of the bubble, for maximum room to re-letter
MODES = (AUTO, TIGHT, WHOLE)


def colour(name: str) -> tuple[int, int, int]:
    """``"white"``, ``"#ffe9d0"`` and friends as an RGB triple."""
    from PIL import ImageColor  # noqa: PLC0415

    try:
        return ImageColor.getrgb(name)[:3]
    except ValueError as exc:
        raise SystemExit(f"not a colour: {name}") from exc


def _blank(shape):
    import numpy as np  # noqa: PLC0415

    return np.zeros(shape, dtype=np.uint8)


def _disc(radius: int):
    import cv2  # noqa: PLC0415

    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)


def _stroke_mask(masks: PageMasks, group: TextGroup, bleed: int, shape):
    """Per-pixel mask of the lettering in ``group``.

    Within a fragment box, everything that is not the background tone is
    lettering. That is exact on a bubble and tight enough on artwork, because
    the detector's boxes hug the glyphs.
    """
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    height, width = shape
    backing = masks.pale if group.polarity == "light" else masks.ink
    mask = _blank(shape)
    for fragment in group.boxes:
        patch = fragment.padded(bleed, width, height)
        window = (slice(patch.y0, patch.y1), slice(patch.x0, patch.x1))
        mask[window][~backing[window]] = 255
    if bleed > 0:
        k = 2 * bleed + 1
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
    return mask


def _reach(group: TextGroup, shape, reach_glyphs: float = WIPE_GLYPHS) -> Box:
    """The group's footprint with a little room around it."""
    height, width = shape
    return group.bbox.padded(reach_glyphs * group.glyph_size, width, height)


def _interior_mask(region, shape):
    """The inside of ``region`` as a full-page mask, clear of its own outline."""
    interior = region.interior(OUTLINE_GUARD_PX)
    mask = _blank(shape)
    box = region.box
    mask[box.y0 : box.y1, box.x0 : box.x1][interior.astype(bool)] = 255
    return mask


def _bubble_mask(group: TextGroup, shape, reach_glyphs: float = WIPE_GLYPHS):
    """Inside of the group's bubble, around this group's text only.

    The bubble's own outline is left alone, and so is anything in the bubble far
    enough away to belong to another utterance.
    """
    mask = _interior_mask(group.region, shape)
    reach = _reach(group, shape, reach_glyphs)
    outside = _blank(shape)
    outside[reach.y0 : reach.y1, reach.x0 : reach.x1] = 255
    return mask & outside


def _drop_specks(mask, group: TextGroup):
    """Drop everything in ``mask`` too small to be part of a character.

    This is what keeps screentone out of it. Filtering by connected component
    rather than by an opening matters: an opening thins the strokes it keeps,
    and a stroke that has been thinned lets the ink it was covering back out.
    """
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    limit = SPECK_GLYPHS * group.glyph_size**2
    if limit < 2:  # nothing worth removing at this size
        return mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= limit
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def _knit(mask, group: TextGroup, knit_glyphs: float, bleed: int):
    """Close the gaps between strokes, then bleed over the anti-aliased rim.

    Closing first is what turns a column of characters into one patch: the type
    is meant to be covered, not traced. The bleed afterwards takes the half-toned
    pixels at the edge of every glyph, which are neither ink nor paper and
    survive as a grey ghost if they are left.
    """
    import cv2  # noqa: PLC0415

    knit = int(round(knit_glyphs * group.glyph_size))
    if knit > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _disc(knit))
    if bleed > 0:
        mask = cv2.dilate(mask, _disc(bleed))
    return mask


def _tight_mask(
    masks: PageMasks,
    group: TextGroup,
    shape,
    bleed: int,
    knit_glyphs: float,
    reach_glyphs: float = WIPE_GLYPHS,
):
    """Everything that has to be covered for this group's lettering to go.

    Inside a bubble the ink itself is the answer rather than the boxes the
    detector drew around it: the region says which pixels are that piece of
    paper, so whatever is left on it near the text is lettering - the furigana
    the detector never boxed included. Text with no bubble behind it has no such
    backing to appeal to and is read inside its own fragments, which is as far
    as it is safe to go with artwork underneath.
    """
    height, width = shape
    backing = masks.pale if group.polarity == "light" else masks.ink

    if group.region is None:
        keep = _blank(shape)
        for fragment in group.boxes:
            patch = fragment.padded(bleed, width, height)
            keep[patch.y0 : patch.y1, patch.x0 : patch.x1] = 255
    else:
        keep = _bubble_mask(group, shape, reach_glyphs)

    mask = _blank(shape)
    mask[keep.astype(bool) & ~backing] = 255
    mask = _knit(_drop_specks(mask, group), group, knit_glyphs, bleed)
    # Closing and bleeding both grow the patch, and neither knows what it is
    # growing over. Holding it back to where the lettering was looked for in the
    # first place is what keeps "exactly over the text" true: inside a bubble the
    # white can never cross the drawn outline, and elsewhere it can never leave
    # the fragments the detector actually found.
    return mask & keep


def _paint_background(pixels, group: TextGroup, mask, area: Box) -> None:
    """Replace ``mask`` with the surface the lettering was sitting on.

    A greyscale closing takes out everything darker than its surroundings and
    narrower than the brush - which is what lettering is - and leaves the
    background behind, shading and all. Light text on a dark plate is the same
    operation upside down.
    """
    import cv2  # noqa: PLC0415

    radius = min(MAX_BRUSH_PX, max(3, round(BACKGROUND_BRUSH * group.glyph_size)))
    height, width = pixels.shape[:2]
    window = area.padded(radius, width, height)
    patch = pixels[window.y0 : window.y1, window.x0 : window.x1]
    if patch.size == 0:
        return

    brush = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1,) * 2)
    operation = cv2.MORPH_CLOSE if group.polarity == "light" else cv2.MORPH_OPEN
    background = cv2.morphologyEx(patch, operation, brush)

    local = mask[window.y0 : window.y1, window.x0 : window.x1].astype(bool)
    patch[local] = background[local]


def plan_erase(
    masks: PageMasks,
    groups: list[TextGroup],
    shape,
    *,
    bleed_glyphs: float = BLEED_GLYPHS,
    knit_glyphs: float = KNIT_GLYPHS,
    reach_glyphs: float = WIPE_GLYPHS,
    mode: str = AUTO,
):
    """What to cover for ``groups``, and what has to be reconstructed instead.

    Returns ``(painted, guessed)``: a list of ``(group, mask, area)`` to paint
    over, and one mask of the strokes that only an inpainter can deal with.
    ``guessed`` is always empty outside ``AUTO`` - the other modes were asked for
    a flat patch, and reconstructing artwork under a sound effect is not that.
    """
    painted: list[tuple[TextGroup, object, Box]] = []
    guessed = _blank(shape)
    whole_done: set[int] = set()

    for group in groups:
        bleed = max(MIN_BLEED_PX, round(bleed_glyphs * group.glyph_size))

        if mode == AUTO:
            if group.kind == ART:
                # Line art runs underneath it; only a reconstruction will do.
                guessed |= _stroke_mask(masks, group, bleed, shape)
            elif group.region is not None:
                # The whole bubble around this utterance, so unboxed furigana and
                # specks go with it.
                painted.append(
                    (group, _bubble_mask(group, shape, reach_glyphs), group.region.box)
                )
            else:
                painted.append((
                    group,
                    _stroke_mask(masks, group, bleed, shape),
                    _reach(group, shape, reach_glyphs),
                ))
            continue

        if mode == WHOLE and group.region is not None:
            # Every utterance in the bubble is going anyway, so the interior is
            # cleared once rather than once per utterance.
            if id(group.region) in whole_done:
                continue
            whole_done.add(id(group.region))
            painted.append(
                (group, _interior_mask(group.region, shape), group.region.box)
            )
            continue

        mask = _tight_mask(masks, group, shape, bleed, knit_glyphs, reach_glyphs)
        painted.append((group, mask, _reach(group, shape, reach_glyphs)))

    return painted, guessed


def text_mask(
    masks: PageMasks,
    groups: list[TextGroup],
    shape,
    *,
    bleed_glyphs: float = BLEED_GLYPHS,
    knit_glyphs: float = KNIT_GLYPHS,
    reach_glyphs: float = WIPE_GLYPHS,
    mode: str = TIGHT,
):
    """One page-sized mask of everything ``groups`` covers: 255 on, 0 off.

    This is the picture to hand a hand-lettering step: paint it white and the
    Japanese is gone, with the space it used to occupy marked out for the
    English that replaces it.
    """
    painted, _ = plan_erase(
        masks,
        groups,
        shape,
        bleed_glyphs=bleed_glyphs,
        knit_glyphs=knit_glyphs,
        reach_glyphs=reach_glyphs,
        mode=mode,
    )
    combined = _blank(shape)
    for group, mask, _area in painted:
        combined |= mask
        group.mask_bbox = mask_bounds(mask)
    return combined


def mask_bounds(mask) -> Box | None:
    """Tight box around everything set in ``mask``, or ``None`` if it is empty."""
    import cv2  # noqa: PLC0415

    x, y, w, h = cv2.boundingRect(mask)
    return Box(int(x), int(y), int(x + w), int(y + h)) if w and h else None


def erase_text(
    image,
    groups: list[TextGroup],
    masks: PageMasks,
    bleed_glyphs: float,
    *,
    mode: str = AUTO,
    fill: tuple[int, int, int] | None = None,
    knit_glyphs: float = KNIT_GLYPHS,
    reach_glyphs: float = WIPE_GLYPHS,
):
    """Return ``image`` with the lettering of ``groups`` taken off.

    With ``fill`` unset the surface under each group is measured and painted
    back, which is what restores a bubble carrying a wash. Give a colour instead
    - white, normally - and that colour is laid down flat, which is what a page
    about to be re-lettered by hand wants.
    """
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    if not groups:
        return image

    pixels = np.asarray(image).copy()
    shape = pixels.shape[:2]
    painted, guessed = plan_erase(
        masks,
        groups,
        shape,
        bleed_glyphs=bleed_glyphs,
        knit_glyphs=knit_glyphs,
        reach_glyphs=reach_glyphs,
        mode=mode,
    )

    for group, mask, area in painted:
        if fill is None:
            _paint_background(pixels, group, mask, area)
        else:
            pixels[mask.astype(bool)] = fill
        group.mask_bbox = mask_bounds(mask)

    if guessed.any():
        pixels = cv2.inpaint(pixels, guessed, INPAINT_RADIUS, cv2.INPAINT_TELEA)
    return Image.fromarray(pixels)
