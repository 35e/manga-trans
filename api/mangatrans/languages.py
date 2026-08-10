"""What a page can be written in, and what that means for reading it.

Finding the text is the same whatever the language: comic-text-detector was
trained on comics rather than on a script, and neither its blocks nor its
per-pixel mask know what alphabet they are looking at. Everything downstream of
it does. manga-ocr reads Japanese and only Japanese; a page of Korean is read
left to right where a page of Japanese is read right to left; and lines of
Chinese run together where lines of Korean are spaced.

So this is the one table both ends look those up in — the API for which reader
to stand up and which way to sort, the dashboard for what to offer and what to
tell the translator the page is in.
"""

from __future__ import annotations

from dataclasses import dataclass

# Which reader can read it. manga-ocr is a manga model and worth having for the
# language it was trained on; PP-OCR is a general printed-text recogniser and is
# what everything else goes through. See :mod:`mangatrans.read`.
MANGA_OCR = "manga-ocr"
PPOCR = "pp-ocr"


@dataclass(frozen=True)
class Language:
    """One language a page may be lettered in."""

    code: str
    name: str
    reader: str
    # PP-OCR's own name for the language, which is which weights to fetch. Empty
    # where the reader is not PP-OCR.
    recogniser: str = ""
    # Whether the page is read right to left, which is the order the blocks come
    # back in and so the order the page is translated in as one conversation.
    rtl: bool = False
    # Whether the script sets one character under another when it runs down the
    # page. A column of that is straightened out before it is read; a tall block
    # of a script that does not stack is a line of it turned on its side, and is
    # left to the reader to turn back.
    stacked: bool = True
    # Whether words are told apart by spaces, which is how the lines of one
    # balloon are joined back together.
    spaced: bool = False


# Traditional Chinese is the entry for a page set in columns and read right to
# left, since that is how the comics printed in it are set; simplified is the
# entry for the rows-and-left-to-right of a webcomic. Which way round a
# particular page is drawn is still the reader's to correct — blocks can be
# dragged into order in the dashboard.
LANGUAGES: tuple[Language, ...] = (
    Language("ja", "Japanese", MANGA_OCR, rtl=True),
    Language("zh", "Chinese (simplified)", PPOCR, "ch"),
    Language("zh-Hant", "Chinese (traditional)", PPOCR, "chinese_cht", rtl=True),
    Language("ko", "Korean", PPOCR, "korean", spaced=True),
    Language("en", "English", PPOCR, "en", stacked=False, spaced=True),
)

# What a caller that says nothing gets: this was written for manga first, and a
# request that predates any of this still means Japanese.
DEFAULT = LANGUAGES[0]

CODES = tuple(language.code for language in LANGUAGES)


def of(code: str | None) -> Language:
    """The language with that code, or the default where nothing was asked for."""
    if not code or not code.strip():
        return DEFAULT
    wanted = code.strip().lower()
    for language in LANGUAGES:
        if language.code.lower() == wanted:
            return language
    raise KeyError(code)
