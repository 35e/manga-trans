"""Reading the lettering. The detector says where the text is; this says what it says.

Japanese goes to manga-ocr, which is trained on manga specifically — vertical
lines, stylised fonts, furigana — and is why it manages what general OCR does
not. Every other language goes to PP-OCR, a general printed-text recogniser with
a small set of weights per language; there is no manga-ocr for Korean, and using
the Japanese one on Chinese only gets Japanese back. Which reader a language
takes is in :mod:`mangatrans.languages`. Both run on the CPU.

**The only module here that needs torch**, and the import is deferred to the
first Japanese page read, so an API only ever asked to detect, to clean, or to
read Korean never pays for it. PP-OCR's onnxruntime is deferred the same way.
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
from PIL import Image

from . import languages
from .geometry import Box
from .languages import Language

MODEL_NAME = "kha-white/manga-ocr-base"
MODEL_ENV = "MANGA_TRANS_OCR_MODEL"

# PP-OCR's weights are a file per language, a few megabytes each, fetched on
# first use into here.
PPOCR_ENV = "MANGA_TRANS_OCR_MODELS"
PPOCR_DIR = "~/.cache/manga-trans/ppocr"

# The newest weights PP-OCR has, and the languages that have none that new and
# take the last set that did.
PPOCR_VERSION = "PP-OCRv5"
PPOCR_OLDER = {"chinese_cht": "PP-OCRv4"}

# The detector boxes lettering tightly. A little air around it reads better, but
# not so much that the next bubble is drawn in.
PAD = 0.03
PAD_MIN = 2

# Smaller than this in either direction is not lettering, whatever a model makes
# of it.
SMALLEST = 4

# A run of ink thinner than this is a speck — dirt, a fragment of a balloon's
# outline caught by the margin — rather than a line of text worth a model call.
SPECK = 3

# The gap left between two characters when a column is set out as a line, as a
# share of the character. Only so the model is not handed a column's worth of
# glyphs run solid together.
LOOSE = 0.06

# onnxruntime goes looking for a GPU as it loads and, in a container that has
# been shown a graphics card it cannot read the make of — which is every
# container on a Mac — says so at warning level on the way past. Nothing here
# wants a GPU and the reading is unaffected, but it is the loudest thing in the
# log and it reads like a failure. See :func:`quieted`.
GPU_HUNT = ("device_discovery.cc", "GetGpuDevices")


def model_name(explicit: str | None = None) -> str:
    """The model to read Japanese with: the one asked for, else the manga one."""
    return explicit or os.environ.get(MODEL_ENV) or MODEL_NAME


def ppocr_models(explicit: str | None = None) -> Path:
    """Where PP-OCR's weights are kept, which is where it downloads them to."""
    return Path(explicit or os.environ.get(PPOCR_ENV) or PPOCR_DIR).expanduser()


def ensure_model(explicit: str | None = None) -> str:
    """Fetch manga-ocr's weights (~450 MB) into the Hugging Face cache.

    Downloading is not loading: the image is built with this, which needs no
    torch, rather than by standing a whole model up.
    """
    from huggingface_hub import snapshot_download

    name = model_name(explicit)
    print(f"mangatrans: downloading {name}")
    path = snapshot_download(name)
    print(f"mangatrans: saved {path}")
    return name


def ensure_readers(explicit: str | None = None) -> None:
    """Fetch the weights for every language that can be read.

    PP-OCR has no download that is not a load: its weights come down as the
    engine is stood up, so each one is stood up here and thrown away. They are a
    few megabytes apiece, against manga-ocr's several hundred.
    """
    ensure_model(explicit)
    for language in languages.LANGUAGES:
        if language.reader == languages.PPOCR:
            print(f"mangatrans: downloading PP-OCR for {language.name}")
            Ppocr.load(language)


def padded(box: Box, width: int, height: int) -> Box:
    """The box with a margin, still on the page."""
    pad = max(PAD_MIN, round(PAD * min(box.w, box.h)))
    return Box(box.x0 - pad, box.y0 - pad, box.x1 + pad, box.y1 + pad).clipped(
        width, height
    )


# --- Handing a balloon to a reader that only knows lines --------------------
#
# manga-ocr takes a block whole, however it was set. PP-OCR takes one line at a
# time and reads across the page and nothing else, so a balloon has to be taken
# apart before it goes over: cut into its lines, and — where those lines are
# columns — each column set out as a line first.


def inked(crop: Image.Image) -> np.ndarray:
    """Where the ink is in one crop, pixel by pixel.

    Otsu rather than a fixed threshold: a scan's black is not 0 and its white is
    not 255. Which of the two sides it hands back is the ink is then decided at
    the edge of the crop, since a block comes with a margin around it and what
    runs round the outside of one is the ground it was set on — that is what
    tells white lettering on a dark balloon from black on a light one. Taking the
    rarer of the two would do for ordinary dialogue and get a heavy sound effect
    exactly backwards.
    """
    grey = np.array(crop.convert("L"))
    _, dark = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ink = dark > 0
    edge = np.concatenate([ink[0], ink[-1], ink[:, 0], ink[:, -1]])
    return ~ink if edge.mean() > 0.5 else ink


def runs(present: np.ndarray) -> list[tuple[int, int]]:
    """Every run of True, as (start, end) with the end exclusive."""
    edges = np.flatnonzero(np.diff(np.concatenate(([False], present, [False]))))
    return [
        (int(start), int(end))
        for start, end in zip(edges[::2].tolist(), edges[1::2].tolist())
        if end - start >= SPECK
    ]


def upright(ink: np.ndarray) -> bool:
    """Whether the lettering in a block runs down the page rather than across it.

    Measured off the ink rather than taken from the language, because a language
    that can be set in columns is rarely set only in columns: the same page
    carries a balloon of lines and a sound effect written down the side of it.

    Which it is, is the shape of what was set. A line is as long as the balloon
    is wide and a balloon holds a few of them, so a block of lines comes out
    wider than it is tall; a column is as long as the balloon is deep, and a
    block of columns comes out taller than it is wide. The gaps look like the
    better signal — line spacing against character spacing — but they are not:
    CJK is set solid both ways, and how much air is left between two columns is
    the letterer's taste rather than anything to measure against.
    """
    down = np.flatnonzero(ink.any(axis=1))
    across = np.flatnonzero(ink.any(axis=0))
    if not len(down) or not len(across):
        return False
    return down[-1] - down[0] > across[-1] - across[0]


def cells(ink: np.ndarray) -> list[tuple[int, int]]:
    """Where one character ends and the next begins down a column.

    The blanks between them, where there are any. A column set solid enough that
    there are none is cut into squares of its own width instead — CJK is set on a
    square em, so that is where the characters are whether or not the ink says
    so. Only where there is plainly more than one of them: a block one character
    tall is one character, not something to chop up.
    """
    height, width = ink.shape[:2]
    found = runs(ink.any(axis=1))
    if len(found) > 1 or height < width * 2:
        return found
    return [(at, min(at + width, height)) for at in range(0, height, width)]


def ground(crop: Image.Image, ink: np.ndarray) -> tuple[int, int, int]:
    """What the lettering is set on, so anything laid out afresh sits on the same."""
    behind = np.array(crop)[~ink]
    if not len(behind):
        return (255, 255, 255)
    return tuple(int(value) for value in np.median(behind, axis=0))


def unstacked(crop: Image.Image, ink: np.ndarray) -> Image.Image:
    """One column of characters set out as a line of them.

    PP-OCR reads across the page, and handed a column it reads the line that
    column would be if the page were turned — every glyph on its side. Cutting
    the column at the gaps between its characters and setting those out left to
    right hands the model what it was trained on.

    The whole column goes over as one line rather than a character at a time:
    the model reads a line as a line, and a character shown on its own has
    nothing either side of it to be read in the light of.
    """
    found = cells(ink)
    if len(found) < 2:
        return crop

    width = crop.width
    height = max(end - start for start, end in found)
    gap = max(1, round(LOOSE * width))

    line = Image.new(
        "RGB", (len(found) * (width + gap) - gap, height), ground(crop, ink)
    )
    for at, (start, end) in enumerate(found):
        cell = crop.crop((0, start, width, end))
        line.paste(cell, (at * (width + gap), (height - cell.height) // 2))
    return line


def pieces(crop: Image.Image, language: Language) -> list[Image.Image]:
    """One block cut into the lines PP-OCR wants, in the order they are read.

    A block whose ink says nothing — a crop that came out blank under Otsu, or
    one whose every mark is a speck — is handed over whole rather than dropped:
    the model may still make something of it, and there is nothing better to go
    on.
    """
    ink = inked(crop)
    if not ink.any():
        return [crop]

    if upright(ink):
        # A tall block of a script that does not stack is a line of it turned on
        # its side — a sound effect running up the page. PP-OCR turns that back
        # itself, and cutting it into rows would hand it a letter at a time.
        if not language.stacked:
            return [crop]
        # Columns, right to left: a script set in columns is read that way
        # whichever way round its pages are.
        cut = [
            unstacked(crop.crop((start, 0, end, crop.height)), ink[:, start:end])
            for start, end in reversed(runs(ink.any(axis=0)))
        ]
    else:
        cut = [
            crop.crop((0, start, crop.width, end))
            for start, end in runs(ink.any(axis=1))
        ]
    return cut or [crop]


# --- The readers ------------------------------------------------------------


class Unfetched(RuntimeError):
    """A reader's weights are not here and could not be had."""


@contextlib.contextmanager
def quieted():
    """stderr with onnxruntime's hunt for a GPU taken out of it.

    The hunt is written from C++ straight to the file descriptor, so there is no
    logger to turn down: the descriptor itself is caught and everything that is
    not the hunt is written back out afterwards, since a reader that cannot find
    its weights says so the same way. Held only for as long as one reader takes
    to load, that being the whole process's stderr.
    """
    caught = tempfile.TemporaryFile()
    sys.stderr.flush()
    kept = os.dup(2)
    try:
        os.dup2(caught.fileno(), 2)
        try:
            yield
        finally:
            sys.stderr.flush()
            os.dup2(kept, 2)
    finally:
        os.close(kept)
        caught.seek(0)
        said = caught.read().decode("utf-8", "replace")
        caught.close()
        for line in said.splitlines(keepends=True):
            if not all(mark in line for mark in GPU_HUNT):
                sys.stderr.write(line)
        sys.stderr.flush()


class Ppocr:
    """PP-OCR, for the languages manga-ocr was not trained on.

    One instance is one language: the weights are a file per language and the
    engine is built around one of them.
    """

    def __init__(self, language: Language) -> None:
        self.language = language
        self.engine = self.load(language)

    @staticmethod
    def load(language: Language):
        """The engine itself. Imported here, so onnxruntime loads on first use.

        Building it fetches whatever of its weights are not already down, which
        is the one thing here that wants the network at all — the image bakes
        them in, so a run that reaches for them was built without them. Saying so
        is worth the lines: what comes back otherwise is a modelscope URL.
        """
        with quieted():
            from rapidocr import EngineType, LangRec, ModelType, OCRVersion, RapidOCR

            version = PPOCR_OLDER.get(language.recogniser, PPOCR_VERSION)
            try:
                return RapidOCR(
                    params={
                        "Global.model_root_dir": str(ppocr_models()),
                        # It says which weights it is loading, on every engine,
                        # at INFO.
                        "Global.log_level": "warning",
                        "Rec.lang_type": LangRec(language.recogniser),
                        "Rec.ocr_version": OCRVersion(version),
                        "Rec.model_type": ModelType.MOBILE,
                        "Rec.engine_type": EngineType.ONNXRUNTIME,
                    }
                )
            except Exception as exc:
                raise Unfetched(
                    f"the {language.name} reader could not be stood up: {exc}. "
                    f"Its weights ({language.recogniser}, {version}) are looked "
                    f"for in {ppocr_models()} and fetched on first use if they "
                    "are not there."
                ) from exc

    def line(self, image: Image.Image) -> str:
        """What one line says.

        Detection and the up-the-right-way classifier are both off: the block is
        already known to be text, and it has already been cut into lines here.
        """
        if image.width < SMALLEST or image.height < SMALLEST:
            return ""
        # An array is taken as BGR, which is the order the models were trained on.
        pixels = np.ascontiguousarray(np.array(image.convert("RGB"))[:, :, ::-1])
        found = self.engine(pixels, use_det=False, use_cls=False, use_rec=True)
        return " ".join(said.strip() for said in (found.txts or ()) if said.strip())

    def __call__(self, image: Image.Image) -> str:
        """What one block says, its lines read in order and joined back up."""
        said = (self.line(piece) for piece in pieces(image, self.language))
        joint = " " if self.language.spaced else ""
        return joint.join(line for line in said if line)


class Reader:
    """The loaded OCR models, one per language, each stood up on first use.

    One box at a time: neither torch generation nor an ONNX session is
    reentrant, and the lock is shared across languages because the API works a
    page at a time anyway.
    """

    def __init__(self, model: str | None = None, ocr=None) -> None:
        # ``ocr`` is anything that turns one image into one string. Passing it runs
        # the cropping and the loop without half a gigabyte of weights, which is
        # what the tests do, and it stands in for every language.
        self.given = ocr
        self.model = model
        self.held: dict[str, Callable[[Image.Image], str]] = {}
        self._lock = threading.Lock()
        self._loading = threading.Lock()

    @staticmethod
    def load(model: str | None = None):
        """The manga-ocr model itself. Imported here, so torch loads on first use."""
        from manga_ocr import MangaOcr

        return MangaOcr(pretrained_model_name_or_path=model_name(model), force_cpu=True)

    def reads(self, language: Language) -> Callable[[Image.Image], str]:
        """The reader for one language, loaded once and kept.

        Under a lock of its own rather than the one held per crop: standing a
        model up takes seconds, and holding the reading lock through it would
        stop a language that is already loaded from being read in the meantime.
        """
        if self.given is not None:
            return self.given
        with self._loading:
            if language.code not in self.held:
                self.held[language.code] = (
                    self.load(self.model)
                    if language.reader == languages.MANGA_OCR
                    else Ppocr(language)
                )
            return self.held[language.code]

    def __call__(
        self,
        image: Image.Image,
        boxes: Sequence[Box],
        language: Language | None = None,
    ) -> list[str]:
        """What each box says, in the order the boxes were given."""
        ocr = self.reads(language or languages.DEFAULT)
        texts = []
        for box in boxes:
            if box.w < SMALLEST or box.h < SMALLEST:
                texts.append("")
                continue
            crop = padded(box, image.width, image.height)
            piece = image.crop((crop.x0, crop.y0, crop.x1, crop.y1))
            # Held for one crop rather than the whole page, so a second request is
            # only ever a box behind.
            with self._lock:
                texts.append(ocr(piece).strip())
        return texts
