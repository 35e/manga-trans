"""Hiding the old lettering, and setting new text in its place."""

from __future__ import annotations

import functools
from dataclasses import dataclass

from PIL import Image, ImageChops, ImageDraw, ImageFont

from . import inpaint
from .geometry import Box

FONTS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FONT_MIN = 8
LINE_SPACING = 0.15
INSET = 0.06

ART = "art"
TELEA = "telea"
WHITE_OUT = "white"
FILLS = (ART, TELEA, WHITE_OUT)


@dataclass(frozen=True)
class Region:
    """A box to hide, and the words to set in its place."""

    box: Box
    text: str = ""


@dataclass(frozen=True)
class Layout:
    """One block of text set at one size, and whether it landed in its box."""

    font: object
    block: str
    spacing: float
    bbox: tuple[float, float, float, float]
    fits: bool


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


@functools.lru_cache(maxsize=8)
def font_file(path: str | None = None) -> str | None:
    """The font file that will actually be used, or None for PIL's own."""
    for candidate in [path] if path else FONTS:
        try:
            ImageFont.truetype(candidate, 12)
        except OSError:
            continue
        return candidate
    return None


@functools.lru_cache(maxsize=128)
def load_font(path: str | None, size: int):
    chosen = font_file(path)
    if chosen is None:
        return ImageFont.load_default(size=size)
    return ImageFont.truetype(chosen, size)


def wrap(text: str, font, max_width: float) -> list[str]:
    """Greedy word wrap, measured with the font itself."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}" if current else word
        if current and font.getlength(candidate) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def set_at(draw, text: str, box: Box, font_path: str | None, size: int) -> Layout:
    """Wrap ``text`` to ``box`` at one type size and see whether it lands."""
    font = load_font(font_path, size)
    block = "\n".join(wrap(text, font, max(1, box.w)))
    spacing = size * LINE_SPACING
    bbox = draw.multiline_textbbox(
        (0, 0), block, font=font, spacing=spacing, align="center"
    )
    left, top, right, bottom = bbox
    fits = right - left <= box.w and bottom - top <= box.h
    return Layout(font, block, spacing, bbox, fits)


def fit(draw, text: str, box: Box, font_path: str | None) -> Layout:
    """The largest size that lands in ``box``, or the smallest if none does."""
    best = set_at(draw, text, box, font_path, FONT_MIN)
    lo, hi = FONT_MIN + 1, max(FONT_MIN, box.h)
    while lo <= hi:
        size = (lo + hi) // 2
        layout = set_at(draw, text, box, font_path, size)
        if layout.fits:
            best, lo = layout, size + 1
        else:
            hi = size - 1
    return best


def letter(draw, box: Box, text: str, font_path: str | None = None) -> None:
    """Set ``text`` centred in ``box``, as large as it will go."""
    inset = max(1, round(INSET * min(box.w, box.h)))
    area = Box(box.x0 + inset, box.y0 + inset, box.x1 - inset, box.y1 - inset)
    if area.w < FONT_MIN or area.h < FONT_MIN:
        area = box

    layout = fit(draw, text, area, font_path)
    left, top, right, bottom = layout.bbox
    draw.multiline_text(
        (area.cx - (right - left) / 2 - left, area.cy - (bottom - top) / 2 - top),
        layout.block,
        font=layout.font,
        fill=BLACK,
        spacing=layout.spacing,
        align="center",
    )


def overlay(
    image: Image.Image,
    regions: list[Region],
    font_path: str | None = None,
    fill: str = WHITE_OUT,
    painter=None,
) -> Image.Image:
    """A copy of the page with every region hidden and its text set in it."""
    marks = marked(image.size, [region.box for region in regions])
    out = hidden(image, marks, fill, painter)
    draw = ImageDraw.Draw(out)
    for region in regions:
        text = region.text.strip()
        if text and region.box.w > 0 and region.box.h > 0:
            letter(draw, region.box, text, font_path)
    return out
