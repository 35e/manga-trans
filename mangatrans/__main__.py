"""Start the review GUI: python -m mangatrans"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from . import translate
from .server import Backend, Pages, create_app


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="mangatrans", description=__doc__)
    parser.add_argument(
        "--pages",
        type=Path,
        default=Path(os.environ.get("MANGA_TRANS_PAGES", "pages")),
        help="folder of manga pages (default: pages)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(os.environ.get("MANGA_TRANS_OUT", "out")),
        help="where overlaid pages are written (default: out)",
    )
    parser.add_argument("--host", default=os.environ.get("MANGA_TRANS_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("MANGA_TRANS_PORT", "8000"))
    )
    parser.add_argument("--model", help="path to comictextdetector.pt.onnx")
    parser.add_argument("--font", default=os.environ.get("MANGA_TRANS_FONT"))
    parser.add_argument("--ollama-url", default=translate.DEFAULT_URL)
    parser.add_argument("--ollama-model", default=translate.DEFAULT_MODEL)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    pages = Pages(args.pages, args.out)
    backend = Backend(
        pages,
        model=args.model,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        font=args.font,
    )
    app = create_app(backend)
    print(f"manga-trans: {pages.root} -> {pages.out}")
    print(f"open http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
