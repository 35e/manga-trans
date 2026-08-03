"""Reading a cropped region with manga-ocr."""

from __future__ import annotations

import threading

from .geometry import Box

PAD = 0.04
MIN_PAD = 3

_reader = None
_lock = threading.Lock()


class OcrUnavailable(RuntimeError):
    pass


def reader():
    global _reader
    if _reader is None:
        try:
            from manga_ocr import MangaOcr
        except ImportError as exc:
            raise OcrUnavailable(f"manga-ocr is not usable ({exc})") from exc
        _reader = MangaOcr()
    return _reader


def read(image, box: Box) -> str:
    """Recognise the text inside ``box`` of a PIL image.

    One reader, one page at a time: the browser can ask for two crops at once
    when regions are resized in quick succession.
    """
    width, height = image.size
    pad = max(MIN_PAD, round(PAD * min(box.w, box.h)))
    crop = box.padded(pad, width, height)
    if crop.area <= 0:
        return ""
    with _lock:
        return reader()(image.crop((crop.x0, crop.y0, crop.x1, crop.y1))).strip()
