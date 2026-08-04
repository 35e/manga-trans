"""Covering the original lettering and setting the translation in its place."""

from __future__ import annotations

import functools
from dataclasses import dataclass, field, replace

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
# Mean tone below which the cleaned paper counts as dark and the type goes white.
DARK = 128.0
# Characters that may not begin a line.
TAIL = ",.!?;:'\"’”)]}»"


@functools.lru_cache(maxsize=8)
def font_file(path: str | None = None) -> str | None:
    """The font file that will actually be used, or None for PIL's own.

    The browser is shown the same file, so the preview you drag around is set in
    the typeface that ends up on the page.
    """
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


@dataclass(frozen=True)
class Layout:
    """One block of text set at one size, and how well it took to its box."""

    font: object
    block: str
    spacing: float
    fits: bool
    whole: bool = True  # no word had to be hyphenated to make the lines


@dataclass(frozen=True)
class Region:
    """A box to clean, the words to set, and the box to set them in."""

    box: Box
    text: str = ""
    text_box: Box | None = None

    @property
    def where(self) -> Box:
        """Where the words go, which is over the region unless it was moved."""
        return self.box if self.text_box is None else self.text_box


@dataclass
class Rendered:
    """The overlaid page, with the regions that have something to answer for.

    Each list holds the indexes of regions the caller ought to look at again:
    ones covered with nothing set in them, ones whose words ran outside their
    box, and ones only fitted by hyphenating. All three are fixed the same way —
    give the region more room, or fewer words.
    """

    image: Image.Image
    blank: list[int] = field(default_factory=list)
    overflow: list[int] = field(default_factory=list)
    tight: list[int] = field(default_factory=list)


def split(word: str, font, max_width: float) -> list[str]:
    """Break a word wider than the line, hyphenating where it breaks."""
    pieces: list[str] = []
    current = ""
    for char in word:
        over = bool(current) and font.getlength(f"{current}{char}-") > max_width
        if over and char not in TAIL:
            pieces.append(f"{current}-")
            current = char
        else:
            # Punctuation is let over the edge rather than sent down alone: a
            # line opening with a comma reads worse than a line a hair too long.
            current += char
    pieces.append(current)
    return pieces


def wrap(text: str, font, max_width: float) -> list[str]:
    """Greedy word wrap measured with the font itself.

    A word too wide for the line is broken rather than left hanging over the
    edge. One over-wide word used to make a whole translation unfittable, and an
    unfittable translation was not drawn at all.
    """
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}" if current else word
        if current and font.getlength(candidate) > max_width:
            lines.append(current)
            candidate = word
        current = candidate
        if font.getlength(current) > max_width:
            *broken, current = split(current, font, max_width)
            lines.extend(broken)
    if current:
        lines.append(current)
    return lines


def measure(draw, layout: Layout):
    return draw.multiline_textbbox(
        (0, 0), layout.block, font=layout.font, spacing=layout.spacing, align="center"
    )


def set_at(draw, text: str, box: Box, font_path: str | None, size: int) -> Layout:
    """Wrap ``text`` to ``box`` at one font size and see whether it lands."""
    font = load_font(font_path, size)
    block = "\n".join(wrap(text, font, max(1, box.w)))
    layout = Layout(font, block, size * LINE_SPACING, fits=True)
    left, top, right, bottom = measure(draw, layout)
    return replace(layout, fits=right - left <= box.w and bottom - top <= box.h)


def largest(draw, text: str, box: Box, font_path: str | None, whole: bool):
    """Largest size that lands in ``box``; ``whole`` also forbids splitting a word."""
    lo, hi = FONT_MIN, max(FONT_MIN, box.h)
    best = None
    while lo <= hi:
        size = (lo + hi) // 2
        layout = set_at(draw, text, box, font_path, size)
        splits = whole and any(
            layout.font.getlength(word) > box.w for word in text.split()
        )
        if layout.fits and not splits:
            best, lo = layout, size + 1
        else:
            hi = size - 1
    return best


def fit(draw, text: str, box: Box, font_path: str | None) -> Layout:
    """The way to set ``text`` in ``box``: whole words, else broken, else too big.

    Nothing at all used to be drawn when even the smallest type overran the box,
    so a bubble the detector drew too tight came out cleaned and empty. Type
    that runs over its box can at least be read — and then moved.
    """
    kept = largest(draw, text, box, font_path, whole=True)
    if kept is not None:
        return kept
    broken = largest(draw, text, box, font_path, whole=False)
    return replace(broken or set_at(draw, text, box, font_path, FONT_MIN), whole=False)


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


def letter(draw, box: Box, text: str, font_path, dark: bool, busy: bool = False) -> Layout:
    """Set ``text`` into ``box``, as large as it will go, and say how it went.

    It is drawn whatever the answer: an unfittable line is still worth reading.
    """
    inset = max(1, round(INSET * min(box.w, box.h)))
    area = Box(box.x0 + inset, box.y0 + inset, box.x1 - inset, box.y1 - inset)
    if area.w < FONT_MIN or area.h < FONT_MIN:
        area = box  # too small to give any of it away to a margin

    layout = fit(draw, text, area, font_path)
    left, top, right, bottom = measure(draw, layout)
    ink, paper = ("white", "black") if dark else ("black", "white")
    draw.multiline_text(
        (area.cx - (right - left) / 2 - left, area.cy - (bottom - top) / 2 - top),
        layout.block,
        font=layout.font,
        fill=ink,
        spacing=layout.spacing,
        align="center",
        stroke_width=max(1, layout.font.size // 12) if busy else 0,
        stroke_fill=paper,
    )
    return layout


def overlay(
    image, mask, regions: list[Region], font_path: str | None = None
) -> Rendered:
    """Return a copy of ``image`` with each region covered and its text set in it."""
    pixels = np.array(image.convert("RGB"))
    grey = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    height, width = grey.shape[:2]

    for region in regions:
        box = region.box.clipped(width, height)
        if box.area > 0:
            cover(pixels, grey, mask, box)

    out = Image.fromarray(pixels)
    draw = ImageDraw.Draw(out)
    # Read off the cleaned page, so the colour of the type is decided by the
    # paper it is actually going onto rather than by a guess at what was there.
    cleaned = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)

    rendered = Rendered(out)
    for index, region in enumerate(regions):
        text = region.text.strip()
        box = region.where.clipped(width, height)
        if not text or box.area <= 0:
            rendered.blank.append(index)
            continue
        patch = cleaned[box.y0 : box.y1, box.x0 : box.x1]
        dark = float(patch.mean()) < DARK
        busy = float(patch.std()) > BUSY
        layout = letter(draw, box, text, font_path, dark, busy)
        if not layout.fits:
            rendered.overflow.append(index)
        elif not layout.whole:
            rendered.tight.append(index)
    return rendered
