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
from werkzeug.exceptions import BadRequest, HTTPException, ServiceUnavailable

from . import ollama, render
from .detect import GROW, Detector
from .geometry import Box
from .read import Reader

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


def number(field: str, default: int, low: int, high: int) -> int:
    """A whole number sent beside the image, clamped to what makes sense."""
    raw = request.form.get(field)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise BadRequest(f"'{field}' must be a whole number") from exc
    return max(low, min(high, value))


def mask_in(image: Image.Image) -> Image.Image | None:
    """The mask sent beside the image, if one was: greyscale, the page's size."""
    upload = request.files.get("mask")
    if upload is None:
        return None
    try:
        sent_mask = Image.open(upload.stream)
        sent_mask.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise BadRequest(f"the mask is not a usable image: {exc}") from exc

    # A mask that carries transparency means it: white on clear is what
    # /api/letters hands out, and reading that by brightness alone would say
    # "hide the whole page".
    #
    # But an alpha channel is not the same as transparency. A browser canvas
    # always exports one whether or not anything was made see-through, and a
    # mask drawn white on black comes back opaque from edge to edge: going by
    # its alpha would paint out every pixel. So alpha is only believed when
    # some of it is actually clear.
    alpha = sent_mask.getchannel("A") if "A" in sent_mask.getbands() else None
    shaped = alpha is not None and alpha.getextrema()[0] < 255
    mask = alpha if shaped else sent_mask.convert("L")
    if mask.size != image.size:
        raise BadRequest(
            f"the mask is {mask.width}×{mask.height} "
            f"but the page is {image.width}×{image.height}"
        )
    return mask


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


def create_app(
    font: str | None = None, model: str | None = None, ocr_model: str | None = None
) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD
    font = font or os.environ.get("MANGA_TRANS_FONT")

    # Both models take a moment to load — the detector's ~95 MB, the reader's
    # ~450 MB and the torch behind it — so the first request that needs one pays
    # for it and the rest share it. An API only ever asked to detect never
    # stands the reader up at all.
    loading = threading.Lock()
    loaded: list[Detector] = []
    reading = threading.Lock()
    readers: list[Reader] = []

    def detector() -> Detector:
        with loading:
            if not loaded:
                loaded.append(Detector(model))
            return loaded[0]

    def reader() -> Reader:
        with reading:
            if not readers:
                readers.append(Reader(ocr_model))
            return readers[0]

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
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

    @app.get("/api/models")
    def models():
        """Every model Ollama has to translate with."""
        try:
            return jsonify(models=ollama.models())
        except ollama.Unreachable as exc:
            raise ServiceUnavailable(str(exc)) from exc

    @app.post("/api/translate")
    def translate():
        """One translation per text, in the order they were given.

        No image: this is the one thing here that works on words alone. Which
        model does it is the caller's choice out of /api/models.
        """
        texts = [str(text) for text in sent("texts")]
        model = request.form.get("model", "").strip()
        if not model:
            raise BadRequest("nothing to translate with (form field 'model')")
        target = request.form.get("target", "").strip() or ollama.TARGET_DEFAULT
        try:
            return jsonify(texts=ollama.translate(texts, model, target))
        except ollama.Unreachable as exc:
            raise ServiceUnavailable(str(exc)) from exc

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

    @app.post("/api/letters")
    def letters():
        """A mask of the lettering itself, pixel by pixel, as a PNG.

        White on clear: opaque where the ink is, transparent everywhere else, so
        it can be laid straight over the page or drawn into a canvas. Send it
        back to /api/clean to hide the letters and leave the art they sit on.
        """
        image = page()
        grow = number("grow", GROW, 0, 50)
        mask = Image.fromarray(detector().letters(np.array(image), grow), mode="L")
        out = Image.new("RGBA", mask.size, (255, 255, 255, 0))
        out.putalpha(mask)
        return png(out)

    @app.post("/api/read")
    def read():
        """What the lettering in each box says, in the order they were given."""
        image = page()
        boxes = [box_in(values, image) for values in sent("boxes")]
        return jsonify(texts=reader()(image, boxes))

    @app.post("/api/clean")
    def clean():
        """The page back with white over what was marked: the lettering hidden.

        What is marked can be boxes, a mask, or both. A mask is a greyscale page
        of the same size, and it is the only way to say "this bubble but not
        that corner of it".
        """
        image = page()
        mask = mask_in(image)
        boxes = (
            [box_in(values, image) for values in sent("boxes")]
            if "boxes" in request.form
            else []
        )
        if mask is None and not boxes:
            raise BadRequest("nothing to hide: send 'boxes', a 'mask', or both")

        out = render.cover(image, boxes)
        return png(out if mask is None else render.cover_mask(out, mask))

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
