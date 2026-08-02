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
"""

from __future__ import annotations

from .geometry import Box
from .pipeline import ART, TextGroup
from .regions import PageMasks

# Anti-aliasing leaves a rim of half-toned pixels around every glyph that is
# neither ink nor paper; without a bleed it survives as a grey ghost.
BLEED_GLYPHS = 0.12
MIN_BLEED_PX = 2
# How far past its own text a bubble is wiped. Enough to take the furigana and
# specks the detector never boxed, short of the next utterance in the same
# bubble.
WIPE_GLYPHS = 0.45
# The brush that lifts the lettering off the background, in glyph sizes. It has
# to be wider than a character or the character survives as a smear.
BACKGROUND_BRUSH = 1.4
MAX_BRUSH_PX = 80
# Telea reaches this far in from the edge of the hole. Small on purpose: a large
# radius visibly smears line art.
INPAINT_RADIUS = 3
# Keep the drawn outline of a bubble out of everything we paint.
OUTLINE_GUARD_PX = 2


def _blank(shape):
    import numpy as np  # noqa: PLC0415

    return np.zeros(shape, dtype=np.uint8)


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


def _reach(group: TextGroup, shape) -> Box:
    """The group's footprint with a little room around it."""
    height, width = shape
    return group.bbox.padded(WIPE_GLYPHS * group.glyph_size, width, height)


def _bubble_mask(group: TextGroup, shape):
    """Inside of the group's bubble, around this group's text only.

    The bubble's own outline is left alone, and so is anything in the bubble far
    enough away to belong to another utterance.
    """
    region = group.region
    interior = region.interior(OUTLINE_GUARD_PX)

    mask = _blank(shape)
    box = region.box
    mask[box.y0 : box.y1, box.x0 : box.x1][interior.astype(bool)] = 255

    reach = _reach(group, shape)
    outside = _blank(shape)
    outside[reach.y0 : reach.y1, reach.x0 : reach.x1] = 255
    return mask & outside


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


def erase_text(image, groups: list[TextGroup], masks: PageMasks, bleed_glyphs: float):
    """Return ``image`` with the lettering of ``groups`` taken off.

    Text in a bubble, or standing on plain paper, is replaced by the background
    measured underneath it. Text on artwork is inpainted, and only over its own
    strokes.
    """
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    if not groups:
        return image

    pixels = np.asarray(image).copy()
    shape = pixels.shape[:2]
    guessed = _blank(shape)

    for group in groups:
        bleed = max(MIN_BLEED_PX, round(bleed_glyphs * group.glyph_size))
        if group.kind == ART:
            # Line art runs underneath it; only a reconstruction will do.
            guessed |= _stroke_mask(masks, group, bleed, shape)
            continue
        if group.region is not None:
            # The whole bubble around this utterance, so unboxed furigana and
            # specks go with it.
            _paint_background(pixels, group, _bubble_mask(group, shape), group.region.box)
        else:
            _paint_background(
                pixels, group, _stroke_mask(masks, group, bleed, shape), _reach(group, shape)
            )

    if guessed.any():
        pixels = cv2.inpaint(pixels, guessed, INPAINT_RADIUS, cv2.INPAINT_TELEA)
    return Image.fromarray(pixels)
