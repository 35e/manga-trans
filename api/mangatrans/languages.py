"""What a page can be written in, and what that means for reading it."""

from __future__ import annotations

from dataclasses import dataclass

MANGA_OCR = "manga-ocr"
PPOCR = "pp-ocr"


@dataclass(frozen=True)
class Language:
    """One language a page may be lettered in."""

    code: str
    name: str
    reader: str
    recogniser: str = ""
    rtl: bool = False
    stacked: bool = True
    spaced: bool = False


LANGUAGES: tuple[Language, ...] = (
    Language("ja", "Japanese", MANGA_OCR, rtl=True),
    Language("zh", "Chinese (simplified)", PPOCR, "ch"),
    Language("zh-Hant", "Chinese (traditional)", PPOCR, "chinese_cht", rtl=True),
    Language("ko", "Korean", PPOCR, "korean", spaced=True),
    Language("en", "English", PPOCR, "en", stacked=False, spaced=True),
)

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
