"""The review GUI: a small local server the browser talks to."""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory
from PIL import Image, ImageOps
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

from . import ocr, render, translate
from .detect import Detector
from .geometry import Box

SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
CACHED_PAGES = 3


class Pages:
    """The folder of manga pages, and where the overlaid copies go."""

    def __init__(self, root: Path, out: Path) -> None:
        self.root = Path(root).resolve()
        self.out = Path(out).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.out.mkdir(parents=True, exist_ok=True)

    def names(self) -> list[str]:
        return sorted(
            path.name
            for path in self.root.iterdir()
            if path.is_file() and path.suffix.lower() in SUFFIXES
        )

    def path(self, name: str) -> Path:
        """The page called ``name``, which can never be outside the folder."""
        path = (self.root / name).resolve()
        if path.parent != self.root or not path.is_file():
            raise FileNotFoundError(f"no such page: {name}")
        return path

    def output(self, name: str) -> Path:
        # A stem holds no separators, so this stays inside the output folder.
        return self.out / (Path(name).stem + ".png")

    def save(self, upload) -> str | None:
        """Store an uploaded image, keeping its suffix and its neighbours."""
        source = Path(upload.filename or "")
        suffix = source.suffix.lower()
        if suffix not in SUFFIXES:
            return None
        # secure_filename drops non-ASCII entirely, so a Japanese name would
        # come back empty rather than merely tidied.
        stem = secure_filename(source.stem) or "page"
        path = self.root / f"{stem}{suffix}"
        count = 1
        while path.exists():
            path = self.root / f"{stem}-{count}{suffix}"
            count += 1
        upload.save(path)
        return path.name

    def open(self, name: str) -> Image.Image:
        return ImageOps.exif_transpose(Image.open(self.path(name))).convert("RGB")


class Backend:
    """Models and page state, loaded on first use and shared between requests."""

    def __init__(
        self,
        pages: Pages,
        *,
        model: str | None = None,
        ollama_url: str = translate.DEFAULT_URL,
        ollama_model: str = translate.DEFAULT_MODEL,
        font: str | None = None,
    ) -> None:
        self.pages = pages
        self.model = model
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.font = font
        self._detector: Detector | None = None
        self._cache: OrderedDict[tuple, tuple] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def detector(self) -> Detector:
        if self._detector is None:
            self._detector = Detector(self.model)
        return self._detector

    def page(self, name: str):
        """The image and its detection, cached while the file is unchanged."""
        key = (name, self.pages.path(name).stat().st_mtime_ns)
        with self._lock:
            if key not in self._cache:
                image = self.pages.open(name)
                self._cache[key] = (image, self.detector(np.array(image)))
                while len(self._cache) > CACHED_PAGES:
                    self._cache.popitem(last=False)
            self._cache.move_to_end(key)
            return self._cache[key]


def create_app(backend: Backend) -> Flask:
    app = Flask(__name__, static_folder="web", static_url_path="")
    pages = backend.pages

    @app.errorhandler(Exception)
    def on_error(exc):
        if isinstance(exc, HTTPException):
            return exc
        status = 404 if isinstance(exc, FileNotFoundError) else 500
        return jsonify(error=str(exc) or type(exc).__name__), status

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/pages")
    def list_pages():
        return jsonify(pages=pages.names())

    @app.post("/api/pages")
    def upload():
        added = [pages.save(item) for item in request.files.getlist("file")]
        return jsonify(pages=pages.names(), added=[name for name in added if name])

    @app.get("/api/image/<name>")
    def image(name: str):
        return send_file(pages.path(name))

    @app.get("/api/output/<name>")
    def output(name: str):
        path = pages.output(name)
        if not path.is_file():
            raise FileNotFoundError(f"nothing rendered for {name}")
        return send_file(path)

    @app.post("/api/detect")
    def detect():
        name = request.json["page"]
        image, detection = backend.page(name)
        regions, warning = [], None
        for block in detection.blocks:
            text = ""
            if warning is None:
                try:
                    text = ocr.read(image, block.box)
                except ocr.OcrUnavailable as exc:
                    warning = str(exc)
            regions.append(
                {
                    "box": block.box.as_list(),
                    "text_box": block.box.as_list(),
                    "confidence": round(block.confidence, 3),
                    "text": text,
                    "translation": "",
                }
            )
        return jsonify(
            width=image.width, height=image.height, regions=regions, warning=warning
        )

    @app.post("/api/read")
    def read():
        image, _ = backend.page(request.json["page"])
        box = Box.from_list(request.json["box"]).clipped(image.width, image.height)
        return jsonify(text=ocr.read(image, box))

    @app.post("/api/translate")
    def translate_texts():
        translations = translate.translate(
            list(request.json["texts"]),
            url=backend.ollama_url,
            model=backend.ollama_model,
        )
        return jsonify(translations=translations)

    @app.get("/api/font")
    def font():
        """The very font the page is lettered with, for the browser to preview in."""
        path = render.font_file(backend.font)
        if path is None:
            raise FileNotFoundError("no font file: the built-in one is being used")
        return send_file(path, mimetype="font/ttf")

    @app.post("/api/render")
    def render_page():
        name = request.json["page"]
        image, detection = backend.page(name)
        regions = [
            render.Region(
                box=Box.from_list(region["box"]),
                text=str(region.get("text", "")),
                text_box=(
                    Box.from_list(region["text_box"]) if region.get("text_box") else None
                ),
            )
            for region in request.json["regions"]
        ]
        result = render.overlay(image, detection.mask, regions, backend.font)
        path = pages.output(name)
        result.image.save(path)
        return jsonify(
            output=path.name,
            url=f"/api/output/{name}",
            blank=result.blank,
            overflow=result.overflow,
            tight=result.tight,
        )

    return app
