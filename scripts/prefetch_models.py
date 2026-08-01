#!/usr/bin/env python3
"""Download the CRAFT detector and manga-ocr weights into the model cache.

Run at image build time so a container never needs the network. Honours
``EASYOCR_MODULE_PATH`` / ``HF_HOME``, so the destination is whatever the
container is configured to read from.
"""

from __future__ import annotations

import os
import sys

MODEL = os.environ.get("MANGA_OCR_MODEL", "kha-white/manga-ocr-base")


def main() -> int:
    import easyocr

    print("downloading CRAFT detector...", file=sys.stderr)
    kwargs = dict(gpu=False, detect_network="craft", verbose=False)
    try:
        easyocr.Reader(["ja", "en"], recognizer=False, **kwargs)
    except TypeError:
        easyocr.Reader(["ja", "en"], **kwargs)

    from manga_ocr import MangaOcr

    print(f"downloading {MODEL}...", file=sys.stderr)
    MangaOcr(pretrained_model_name_or_path=MODEL, force_cpu=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
