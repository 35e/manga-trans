"""The HTTP API. An image goes up, boxes or a rendered page comes back.

Nothing is stored: every request carries the page it works on, as a multipart
``image`` field, and gets its answer back in the response.
"""

from __future__ import annotations

import io
import json
import os
import threading

import numpy as np
from flask import Flask, jsonify, request, send_file
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import BadRequest, HTTPException

from . import render
from .detect import Detector
from .geometry import Box

MAX_UPLOAD = 32 * 1024 * 1024
ORIGIN = os.environ.get("MANGA_TRANS_ORIGIN", "*")


def page() -> Image.Image:
    """The uploaded image, upright and in RGB."""
    upload = request.files.get("image")
    if upload is None:
        raise BadRequest("no image was uploaded (multipart form field 'image')")
    try:
        return ImageOps.exif_transpose(Image.open(upload.stream)).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise BadRequest(f"the upload is not a usable image: {exc}") from exc


def sent(field: str) -> list:
    """A JSON list sent alongside the image, in the form field ``field``."""
    raw = request.form.get(field)
    if raw is None:
        raise BadRequest(f"nothing was sent in the form field '{field}'")
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise BadRequest(f"'{field}' is not valid JSON: {exc}") from exc
    if not isinstance(value, list):
        raise BadRequest(f"'{field}' must be a JSON list")
    return value


def box_in(values, image: Image.Image) -> Box:
    """One [x0, y0, x1, y1] from a request, clipped to the page."""
    try:
        box = Box.from_list(values)
    except (TypeError, ValueError) as exc:
        raise BadRequest(f"not a box: {values!r}") from exc
    return box.clipped(image.width, image.height)


def png(image: Image.Image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    buffer.seek(0)
    return send_file(buffer, mimetype="image/png", download_name="page.png")


def create_app(font: str | None = None, model: str | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD
    font = font or os.environ.get("MANGA_TRANS_FONT")

    # The weights are ~95 MB and take a moment to load, so the first request
    # pays for them and the rest share the one detector.
    loading = threading.Lock()
    loaded: list[Detector] = []

    def detector() -> Detector:
        with loading:
            if not loaded:
                loaded.append(Detector(model))
            return loaded[0]

    @app.errorhandler(Exception)
    def on_error(exc):
        if isinstance(exc, HTTPException):
            return jsonify(error=exc.description), exc.code
        return jsonify(error=str(exc) or type(exc).__name__), 500

    @app.after_request
    def allow_origin(response):
        """The front end is served from somewhere else, so it needs letting in."""
        response.headers["Access-Control-Allow-Origin"] = ORIGIN
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        return response

    @app.post("/api/detect")
    def detect():
        """Every block of lettering on the page, boxed."""
        image = page()
        blocks = detector()(np.array(image))
        return jsonify(
            width=image.width,
            height=image.height,
            regions=[
                {"box": block.box.as_list(), "confidence": round(block.confidence, 3)}
                for block in blocks
            ],
        )

    @app.post("/api/clean")
    def clean():
        """The page back with white over every box: the lettering hidden."""
        image = page()
        boxes = [box_in(values, image) for values in sent("boxes")]
        return png(render.cover(image, boxes))

    @app.post("/api/render")
    def overlay():
        """The page back with every box hidden and its text set in its place."""
        image = page()
        regions = [
            render.Region(
                box=box_in(region.get("box"), image), text=str(region.get("text", ""))
            )
            for region in sent("regions")
        ]
        return png(render.overlay(image, regions, font))

    return app
