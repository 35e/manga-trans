#!/usr/bin/env python3
"""Download the models before the first run.

    python scripts/fetch_models.py             # detector and manga-ocr
    python scripts/fetch_models.py --detector  # detector only
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mangatrans.detect import MODEL_DIRS, MODEL_NAME, MODEL_URL


def target(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    for directory in MODEL_DIRS:
        path = Path(directory).expanduser()
        try:
            path.mkdir(parents=True, exist_ok=True)
            return path / MODEL_NAME
        except OSError:
            continue
    raise SystemExit(f"nowhere writable to put {MODEL_NAME}")


def fetch(url: str, path: Path) -> None:
    if path.is_file():
        print(f"already there: {path}")
        return
    print(f"downloading {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(url, partial)
    partial.replace(path)
    print(f"saved {path} ({path.stat().st_size / 1e6:.0f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", action="store_true", help="skip manga-ocr")
    parser.add_argument("--to", help=f"where to put {MODEL_NAME}")
    args = parser.parse_args()

    fetch(MODEL_URL, target(args.to))
    if args.detector:
        return

    print("warming manga-ocr (~450 MB from huggingface)")
    from manga_ocr import MangaOcr

    MangaOcr()
    print("done")


if __name__ == "__main__":
    main()
