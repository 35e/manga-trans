"""Choosing how the page's text is found.

There are two ways to answer "where is the text, and which bits of it are one
utterance?", and they are not equally good.

The first is to detect text fragments with a general-purpose detector and then
*reconstruct* the rest with geometry: segment the page into flat islands, decide
which of them are bubbles, and cluster the fragments by how far apart they are.
That is what :mod:`.detect` and :mod:`.regions` do, and every step of it is a
hand-set threshold. It works, but each threshold is another gate a bubble has to
pass, the gates interact, and a bubble that fails one of them does not come back
wrong - it goes missing.

The second is to ask a model trained on comics, which answers all of it at once:
the text blocks, their language, and a per-pixel mask of the lettering. There is
nothing to tune, and a page that fails does so visibly, with a low confidence
attached, rather than silently.

So the pipeline takes either, behind one interface. What a detector must supply
is ``fragments``; if it can also supply ``blocks`` then the grouping stage steps
aside and uses them, and if it can supply a ``text_mask`` then the eraser uses
that instead of working the lettering out from the page's tones.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .geometry import Box

CRAFT = "craft"  # EasyOCR's general-purpose detector, plus geometry
COMIC = "comic"  # comic-text-detector, trained on manga and comics
AUTO = "auto"  # comic when its weights are installed, else craft
DETECTORS = (AUTO, COMIC, CRAFT)

# Where the comic detector's weights are looked for, in order. The container
# bakes them into /opt/models; a pip install lands them in the user's cache.
MODEL_ENV = "MANGA_TRANS_DETECTOR_MODEL"
MODEL_NAME = "comictextdetector.pt.onnx"
MODEL_DIRS = ("/opt/models", "~/.cache/manga-trans")
# Published with manga-image-translator, which is where comic-text-detector's
# own README points; huggingface hosts no copy of this file.
MODEL_URL = (
    "https://github.com/zyddnys/manga-image-translator/releases/download/"
    "beta-0.2.1/" + MODEL_NAME
)


@dataclass
class Detection:
    """One block of text the detector found, and how sure it is of it.

    ``confidence`` is what the heuristic pipeline never had. A bubble the model
    is unsure about is still reported, carrying the doubt with it, so it can be
    flagged rather than quietly dropped.

    ``fragments`` are the columns (or lines) of type inside the block. The
    pipeline measures the size of the lettering from them, so a block always
    carries at least one, even if it is only the block itself.
    """

    box: Box
    confidence: float = 1.0
    language: str = "unknown"  # "ja", "eng" or "unknown"
    fragments: list[Box] = field(default_factory=list)


@dataclass
class DetectionResult:
    """What a detector found on one page.

    ``fragments`` is the only required part: the columns (or lines) of type, as
    the grouping stage has always expected them. ``blocks`` and ``text_mask``
    are what a trained model can add, and each one retires a pile of geometry
    when it is present.
    """

    fragments: list[Box] = field(default_factory=list)
    blocks: list[Detection] | None = None
    text_mask: object | None = None  # np.ndarray, uint8, page-sized, 0-255

    @property
    def grouped(self) -> bool:
        """Did the detector decide the grouping itself?"""
        return self.blocks is not None


def model_path(explicit: str | None = None) -> Path | None:
    """Where the comic detector's weights are, or ``None`` if not installed."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    elif os.environ.get(MODEL_ENV):
        candidates.append(Path(os.environ[MODEL_ENV]).expanduser())
    else:
        candidates += [Path(d).expanduser() / MODEL_NAME for d in MODEL_DIRS]
    for path in candidates:
        if path.is_file():
            return path
    return None


def missing_model_message(explicit: str | None = None) -> str:
    looked = explicit or os.environ.get(MODEL_ENV) or ", ".join(
        str(Path(d).expanduser() / MODEL_NAME) for d in MODEL_DIRS
    )
    return (
        f"the comic detector needs {MODEL_NAME}, which was not found.\n"
        f"  looked in: {looked}\n"
        f"  fetch it with: python scripts/prefetch_models.py --detector-only\n"
        f"  or download {MODEL_URL}\n"
        f"  and point --detector-model (or ${MODEL_ENV}) at it.\n"
        f"  --detector craft falls back to the old EasyOCR path."
    )


def resolve(choice: str, explicit_model: str | None, log=lambda _m: None) -> str:
    """Turn ``--detector auto`` into a concrete backend name."""
    if choice != AUTO:
        return choice
    if model_path(explicit_model) is not None:
        return COMIC
    log(
        "  comic detector weights not installed, falling back to craft "
        "(see --help for how to install them)"
    )
    return CRAFT


def build(choice: str, args, log=lambda _m: None):
    """Construct the detector named by ``choice``.

    Loading is deferred to here rather than done at import time because each
    backend drags in a different heavy dependency - torch for craft, nothing but
    OpenCV for comic - and a run should only pay for the one it uses.
    """
    backend = resolve(choice, getattr(args, "detector_model", None), log)
    if backend == COMIC:
        from .comicdetect import ComicDetector  # noqa: PLC0415

        path = model_path(getattr(args, "detector_model", None))
        if path is None:
            raise SystemExit(missing_model_message(getattr(args, "detector_model", None)))
        return ComicDetector(
            path,
            conf_threshold=args.detector_conf,
            mask_threshold=args.detector_mask_threshold,
        )

    from .detect import CraftDetector  # noqa: PLC0415

    return CraftDetector(args, log=log)
