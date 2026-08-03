#!/usr/bin/env python3
"""Download the models into the cache the container reads from.

Run at image build time so a container never needs the network. Honours
``EASYOCR_MODULE_PATH`` / ``HF_HOME``, so the destination is whatever the
container is configured to read from.

Three models, and only the first is always needed:

* comic-text-detector, the text detector (~95 MB, ONNX). Downloaded by hand
  rather than through a library because it is published as a GitHub release
  asset and nothing on PyPI knows how to fetch it.
* ``manga-ocr``'s recognition weights (~450 MB), unless ``--detector-only``.
* EasyOCR's CRAFT detector (~80 MB), only with ``--craft``: it is the fallback
  detector now, and an image that has the comic detector does not need it.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

MODEL = os.environ.get("MANGA_OCR_MODEL", "kha-white/manga-ocr-base")
DETECTOR_DIR = Path(os.environ.get("MANGA_TRANS_MODEL_DIR", "/opt/models"))


def fetch_detector(destination: Path) -> None:
    from mangatrans.detectors import MODEL_NAME, MODEL_URL  # noqa: PLC0415

    target = destination / MODEL_NAME
    if target.exists():
        print(f"{target} is already there", file=sys.stderr)
        return
    destination.mkdir(parents=True, exist_ok=True)
    print(f"downloading {MODEL_URL}...", file=sys.stderr)
    # Written to a temporary name first so an interrupted build cannot leave a
    # truncated model behind that looks installed.
    partial = target.with_suffix(target.suffix + ".partial")
    with urllib.request.urlopen(MODEL_URL, timeout=300) as response:
        partial.write_bytes(response.read())
    partial.rename(target)
    print(f"wrote {target} ({target.stat().st_size / 1e6:.0f} MB)", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detector-only",
        action="store_true",
        help="only the text detector; skip the recognition model",
    )
    parser.add_argument(
        "--craft",
        action="store_true",
        help="also fetch EasyOCR's CRAFT weights, for --detector craft",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=DETECTOR_DIR,
        help="where the detector weights go",
    )
    args = parser.parse_args(argv)

    # The package lives beside this script's parent; importable without install.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    fetch_detector(args.dir)

    if args.craft:
        import easyocr  # noqa: PLC0415

        print("downloading CRAFT detector...", file=sys.stderr)
        kwargs = dict(gpu=False, detect_network="craft", verbose=False)
        try:
            easyocr.Reader(["ja", "en"], recognizer=False, **kwargs)
        except TypeError:
            easyocr.Reader(["ja", "en"], **kwargs)

    if not args.detector_only:
        from manga_ocr import MangaOcr  # noqa: PLC0415

        print(f"downloading {MODEL}...", file=sys.stderr)
        MangaOcr(pretrained_model_name_or_path=MODEL, force_cpu=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
