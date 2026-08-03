"""Covering the original lettering and setting the translation in its place."""

from __future__ import annotations

import functools

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .geometry import Box

FONTS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
)

FONT_MIN = 8
LINE_SPACING = 0.15
INSET = 0.06
BLEED = 0.12
MIN_BLEED = 2
KNIT = 0.25
BRUSH = 1.4
MAX_BRUSH = 80
SPECK = 0.012
# Spread of tone left in a region once it is cleaned. A bubble comes out flat;
# anything above this still has artwork in it, and type needs a halo to stay
# readable on top of it.
BUSY = 12.0


@functools.lru_cache(maxsize=128)
def load_font(path: str | None, size: int):
    for candidate in [path] if path else FONTS:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def wrap(text: str, font, max_width: float) -> list[str]:
    """Greedy word wrap measured with the font itself."""
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


def fit(draw, text: str, box: Box, font_path: str | None):
    """Largest font size at which ``text`` wraps inside ``box``."""
    lo, hi, best = FONT_MIN, max(FONT_MIN, box.h), None
    while lo <= hi:
        size = (lo + hi) // 2
        font = load_font(font_path, size)
        lines = wrap(text, font, box.w)
        spacing = size * LINE_SPACING
        block = "\n".join(lines)
        left, top, right, bottom = draw.multiline_textbbox(
            (0, 0), block, font=font, spacing=spacing, align="center"
        )
        if right - left <= box.w and bottom - top <= box.h:
            best = (font, block, spacing)
            lo = size + 1
        else:
            hi = size - 1
    return best


def glyph_size(ink) -> float:
    """Character size of the lettering, from the components of its mask."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    sides = [
        min(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        for i in range(1, count)
    ]
    return float(np.median(sides)) if sides else 0.0


def ink_of(grey, mask, box: Box):
    """Boolean mask of the lettering inside ``box``, in box coordinates.

    The detector's own mask is used where it has one. A region drawn by hand may
    sit on text the detector never found, and there the tones decide: whatever is
    in the minority against the paper is the ink.
    """
    patch = grey[box.y0 : box.y1, box.x0 : box.x1]
    if mask is not None:
        ink = mask[box.y0 : box.y1, box.x0 : box.x1] > 0
        if ink.any():
            return ink
    if patch.size == 0:
        return np.zeros_like(patch, dtype=bool)
    _, dark = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = dark > 0
    return ink if ink.mean() <= 0.5 else ~ink


def _disc(radius: int):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)


def _drop_specks(ink, glyph: float):
    limit = SPECK * glyph**2
    if limit < 2:
        return ink
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= limit
    return keep[labels]


def cover(pixels, grey, mask, box: Box) -> bool:
    """Paint the surface the lettering sat on back over it.

    A greyscale closing removes everything narrower than the brush and darker
    than its surroundings, which is exactly what lettering is, and leaves the
    paper behind with its shading intact. Light text on a dark plate is the same
    operation the other way up. Returns whether that paper was the dark one.
    """
    ink = ink_of(grey, mask, box)
    if not ink.any():
        return False

    glyph = glyph_size(ink) or min(box.w, box.h)
    ink = _drop_specks(ink, glyph)
    if not ink.any():
        return False

    patch_grey = grey[box.y0 : box.y1, box.x0 : box.x1]
    paper = patch_grey[~ink]
    dark_paper = patch_grey[ink].mean() >= (paper.mean() if paper.size else 128)
    knit = round(KNIT * glyph)
    bleed = max(MIN_BLEED, round(BLEED * glyph))
    solid = ink.astype(np.uint8) * 255
    if knit > 0:
        solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, _disc(knit))
    solid = cv2.dilate(solid, _disc(bleed))

    radius = min(MAX_BRUSH, max(3, round(BRUSH * glyph)))
    height, width = grey.shape[:2]
    window = box.padded(radius, width, height)
    patch = pixels[window.y0 : window.y1, window.x0 : window.x1]
    operation = cv2.MORPH_OPEN if dark_paper else cv2.MORPH_CLOSE
    brush = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1,) * 2)
    background = cv2.morphologyEx(patch, operation, brush)

    # The cover never leaves the box the user approved.
    local = np.zeros(patch.shape[:2], dtype=bool)
    y, x = box.y0 - window.y0, box.x0 - window.x0
    local[y : y + box.h, x : x + box.w] = solid > 0
    patch[local] = background[local]
    return dark_paper


def letter(draw, box: Box, text: str, font_path, dark: bool, busy: bool = False):
    """Set ``text`` into ``box``, as large as it will go."""
    inset = max(1, round(INSET * min(box.w, box.h)))
    area = Box(box.x0 + inset, box.y0 + inset, box.x1 - inset, box.y1 - inset)
    if area.w < FONT_MIN or area.h < FONT_MIN:
        return False

    fitted = fit(draw, text, area, font_path)
    if fitted is None:
        return False

    font, block, spacing = fitted
    left, top, right, bottom = draw.multiline_textbbox(
        (0, 0), block, font=font, spacing=spacing, align="center"
    )
    ink, paper = ("white", "black") if dark else ("black", "white")
    draw.multiline_text(
        (area.cx - (right - left) / 2 - left, area.cy - (bottom - top) / 2 - top),
        block,
        font=font,
        fill=ink,
        spacing=spacing,
        align="center",
        stroke_width=max(1, font.size // 12) if busy else 0,
        stroke_fill=paper,
    )
    return True


def overlay(image, mask, regions: list[tuple[Box, str]], font_path: str | None = None):
    """Return a copy of ``image`` with each region covered and its text set in it."""
    pixels = np.array(image.convert("RGB"))
    grey = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    height, width = grey.shape[:2]

    boxes = [box.clipped(width, height) for box, _text in regions]
    dark = [box.area > 0 and cover(pixels, grey, mask, box) for box in boxes]

    out = Image.fromarray(pixels)
    draw = ImageDraw.Draw(out)
    # Measured on the cleaned page: what is left in a region decides whether the
    # type needs a halo to sit on it.
    cleaned = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    for box, (_box, text), dark_paper in zip(boxes, regions, dark):
        if not text.strip() or box.area <= 0:
            continue
        busy = float(cleaned[box.y0 : box.y1, box.x0 : box.x1].std()) > BUSY
        letter(draw, box, text.strip(), font_path, dark_paper, busy)
    return out
