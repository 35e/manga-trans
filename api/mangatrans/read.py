"""Reading the lettering: manga-ocr for Japanese, PP-OCR for everything else.

**The only module here that needs torch**, and the import is deferred to the
first Japanese page read. PP-OCR's onnxruntime is deferred the same way.
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

PPOCR_ENV = "MANGA_TRANS_OCR_MODELS"
PPOCR_DIR = "~/.cache/manga-trans/ppocr"

# The newest weights PP-OCR has, and the languages that take the last set that
# did.
PPOCR_VERSION = "PP-OCRv5"
PPOCR_OLDER = {"chinese_cht": "PP-OCRv4"}

PAD = 0.03
PAD_MIN = 2

SMALLEST = 4

# A run of ink thinner than this is a speck rather than a line of text.
SPECK = 3

# The gap left between two characters when a column is set out as a line.
LOOSE = 0.06

# The one line :func:`quieted` takes out.
GPU_HUNT = ("device_discovery.cc", "GetGpuDevices")


def model_name(explicit: str | None = None) -> str:
    """The model to read Japanese with: the one asked for, else the manga one."""
    return explicit or os.environ.get(MODEL_ENV) or MODEL_NAME


def ppocr_models(explicit: str | None = None) -> Path:
    """Where PP-OCR's weights are kept, which is where it downloads them to."""
    return Path(explicit or os.environ.get(PPOCR_ENV) or PPOCR_DIR).expanduser()


def ensure_model(explicit: str | None = None) -> str:
    """Fetch manga-ocr's weights (~450 MB) into the Hugging Face cache.

    Downloading is not loading: this needs no torch.
    """
    from huggingface_hub import snapshot_download

    name = model_name(explicit)
    print(f"mangatrans: downloading {name}")
    path = snapshot_download(name)
    print(f"mangatrans: saved {path}")
    return name


def ensure_readers(explicit: str | None = None) -> None:
    """Fetch the weights for every language that can be read.

    PP-OCR has no download that is not a load, so each engine is stood up here
    and thrown away.
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


def inked(crop: Image.Image) -> np.ndarray:
    """Where the ink is in one crop, pixel by pixel.

    Which side is the ink is decided at the *edge* of the crop, not by taking
    the rarer of the two — which gets a heavy sound effect exactly backwards.
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

    The shape of what was set, measured off the ink rather than taken from the
    language: the gaps look like the better signal and are not.
    """
    down = np.flatnonzero(ink.any(axis=1))
    across = np.flatnonzero(ink.any(axis=0))
    if not len(down) or not len(across):
        return False
    return down[-1] - down[0] > across[-1] - across[0]


def cells(ink: np.ndarray) -> list[tuple[int, int]]:
    """Where one character ends and the next begins down a column.

    A column set solid enough to have no blanks is cut into squares of its own
    width instead: CJK is set on a square em.
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

    Whole, rather than a character at a time: one shown on its own has nothing
    either side of it to be read in the light of.
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

    A block whose ink says nothing is handed over whole rather than dropped.
    """
    ink = inked(crop)
    if not ink.any():
        return [crop]

    if upright(ink):
        # A tall block of a script that does not stack is a line turned on its
        # side, which PP-OCR turns back itself.
        if not language.stacked:
            return [crop]
        # Columns, right to left, whichever way round the pages are.
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

    Written from C++ straight to the descriptor, so there is no logger to turn
    down. **Not a blanket silencer, and it must not become one**: everything
    that is not the hunt is written back out, since a reader that cannot find
    its weights says so the same way. Held only for one reader's load.
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
    """PP-OCR, for the languages manga-ocr was not trained on. One per language."""

    def __init__(self, language: Language) -> None:
        self.language = language
        self.engine = self.load(language)

    @staticmethod
    def load(language: Language):
        """The engine itself. Imported here, so onnxruntime loads on first use."""
        with quieted():
            from rapidocr import EngineType, LangRec, ModelType, OCRVersion, RapidOCR

            version = PPOCR_OLDER.get(language.recogniser, PPOCR_VERSION)
            try:
                return RapidOCR(
                    params={
                        "Global.model_root_dir": str(ppocr_models()),
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
        """What one line says."""
        if image.width < SMALLEST or image.height < SMALLEST:
            return ""
        # An array is taken as BGR, which is what the models were trained on.
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

    One box at a time: neither torch generation nor an ONNX session is reentrant.
    """

    def __init__(self, model: str | None = None, ocr=None) -> None:
        # ``ocr`` is anything turning one image into one string, and stands in
        # for every language — which is what the tests pass.
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

        Under a lock of its own rather than the one held per crop, or standing a
        model up would stop an already-loaded language being read meanwhile.
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
            # Held for one crop, not the page: a second request is only ever a
            # box behind.
            with self._lock:
                texts.append(ocr(piece).strip())
        return texts
