"""Hiding the old lettering. The new text is set into the page by the dashboard."""

from __future__ import annotations

from PIL import Image, ImageChops, ImageDraw

from . import inpaint
from .geometry import Box

WHITE = (255, 255, 255)

ART = "art"
TELEA = "telea"
WHITE_OUT = "white"
FILLS = (ART, TELEA, WHITE_OUT)


def marked(
    size: tuple[int, int], boxes: list[Box], mask: Image.Image | None = None
) -> Image.Image:
    """Everything to be hidden, as one greyscale page: boxes and mask together."""
    marks = Image.new("L", size, 0)
    draw = ImageDraw.Draw(marks)
    for box in boxes:
        if box.w > 0 and box.h > 0:
            draw.rectangle((box.x0, box.y0, box.x1 - 1, box.y1 - 1), fill=255)
    if mask is not None:
        marks = ImageChops.lighter(marks, mask.convert("L"))
    return marks


def cover_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    """A copy of the page with white laid over it wherever the mask is light."""
    out = image.convert("RGB")
    out.paste(WHITE, (0, 0, out.width, out.height), mask.convert("L"))
    return out


def hidden(
    image: Image.Image, marks: Image.Image, fill: str = ART, painter=None
) -> Image.Image:
    """A copy of the page with everything ``marks`` marks taken out of it.

    ``painter`` is the loaded LaMa, handed in rather than reached for. Without
    one, :data:`ART` is Telea.
    """
    if fill == WHITE_OUT:
        return cover_mask(image, marks)
    return inpaint.fill(image, marks, painter if fill == ART else None)
