"""The HTTP API. An image goes up, boxes or a rendered page comes back.

Nothing is stored: every request carries the page it works on, as a multipart
``image`` field, and gets its answer back in the response.
"""

from __future__ import annotations

import io
import json
import os
import threading
from typing import Callable, TypeVar

import numpy as np
from flask import Flask, jsonify, request, send_file
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import BadRequest, HTTPException, ServiceUnavailable

from . import bubble, ollama, render
from .detect import GROW, GROW_MAX, Detector
from .geometry import Box
from .read import Reader

MAX_UPLOAD = 32 * 1024 * 1024
ORIGIN = os.environ.get("MANGA_TRANS_ORIGIN", "*")

T = TypeVar("T")


def lazily(make: Callable[[], T]) -> Callable[[], T]:
    """One instance, built on first use and shared thereafter.

    Both models take a moment to load — the detector's ~95 MB, the reader's
    ~450 MB and the torch behind it — so the first request that needs one pays
    for it. An API only ever asked to detect never stands the reader up at all.
    """
    lock = threading.Lock()
    held: list[T] = []

    def get() -> T:
        with lock:
            if not held:
                held.append(make())
            return held[0]

    return get


# --- Reading the request --------------------------------------------------


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


def box_in(values, image: Image.Image) -> Box:
    """One [x0, y0, x1, y1] from a request, clipped to the page."""
    try:
        box = Box.from_list(values)
    except (TypeError, ValueError) as exc:
        raise BadRequest(f"not a box: {values!r}") from exc
    return box.clipped(image.width, image.height)


def boxes_in(image: Image.Image, field: str = "boxes") -> list[Box]:
    return [box_in(values, image) for values in sent(field)]


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

    # A mask carrying transparency means it: white on clear is what /api/letters
    # hands out, and reading that by brightness alone would say "hide the whole
    # page". But an alpha channel is not the same as transparency — a browser
    # canvas always exports one, and a mask drawn white on black comes back
    # opaque edge to edge. So alpha is only believed when some of it is clear.
    alpha = sent_mask.getchannel("A") if "A" in sent_mask.getbands() else None
    shaped = alpha is not None and alpha.getextrema()[0] < 255
    mask = alpha if shaped else sent_mask.convert("L")
    if mask.size != image.size:
        raise BadRequest(
            f"the mask is {mask.width}×{mask.height} "
            f"but the page is {image.width}×{image.height}"
        )
    return mask


def fill_in(default: str = render.ART) -> str:
    """What goes where the lettering was: the art around it, or flat white."""
    chosen = request.form.get("fill", default).strip().lower()
    if chosen not in render.FILLS:
        raise BadRequest(f"'fill' must be one of: {', '.join(render.FILLS)}")
    return chosen


def png(image: Image.Image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    buffer.seek(0)
    return send_file(buffer, mimetype="image/png", download_name="page.png")


# --- The app --------------------------------------------------------------


def create_app(
    font: str | None = None, model: str | None = None, ocr_model: str | None = None
) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD
    font = font or os.environ.get("MANGA_TRANS_FONT")

    detector = lazily(lambda: Detector(model))
    reader = lazily(lambda: Reader(ocr_model))

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

    @app.get("/api/prompt")
    def prompt():
        """What the model is told to do, unless a caller says otherwise.

        Handed out so a front end can show it, let it be edited, and send the
        edit back — nothing is kept here.
        """
        return jsonify(prompt=ollama.SYSTEM_DEFAULT)

    @app.post("/api/translate")
    def translate():
        """One translation per text, in the order they were given.

        No image: this is the one thing here that works on words alone. Which
        model does it is the caller's choice out of /api/models, and so is what
        the model is told — send `system` to say something other than the
        default, with `{target}` anywhere the language should go.
        """
        texts = [str(text) for text in sent("texts")]
        model = request.form.get("model", "").strip()
        if not model:
            raise BadRequest("nothing to translate with (form field 'model')")
        target = request.form.get("target", "").strip() or ollama.TARGET_DEFAULT
        system = request.form.get("system", "").strip() or None
        try:
            return jsonify(texts=ollama.translate(texts, model, target, system=system))
        except ollama.Unreachable as exc:
            raise ServiceUnavailable(str(exc)) from exc

    @app.post("/api/detect")
    def detect():
        """Every block of lettering on the page, boxed, with the balloon it is in.

        The box is where the words are; `bubble` is the room they were written
        in, which is what a translation wants — see /api/bubbles. It comes back
        here too because it is worked out from the page and the boxes alone, and
        both are already in hand.
        """
        image = page()
        pixels = np.array(image)
        blocks = detector()(pixels)
        balloons = bubble.bubbles(pixels, [block.box for block in blocks])
        return jsonify(
            width=image.width,
            height=image.height,
            regions=[
                {
                    "box": block.box.as_list(),
                    "confidence": round(block.confidence, 3),
                    "bubble": balloon.as_list() if balloon else None,
                }
                for block, balloon in zip(blocks, balloons)
            ],
        )

    @app.post("/api/bubbles")
    def balloons():
        """The balloon each box is written in, boxed, in the order they were given.

        Japanese runs down the page, so a block of it is a tall narrow column and
        a translation set in that column has nowhere to go. This answers with the
        room around it instead.

        `bubble` is null where no balloon could be made out — lettering over
        artwork is in none — and the caller should keep the box it has. No model
        is involved, so this is the one call on an image that never stands the
        detector up.
        """
        image = page()
        boxes = boxes_in(image)
        found = bubble.bubbles(np.array(image), boxes)
        return jsonify(
            regions=[
                {"box": box.as_list(), "bubble": balloon.as_list() if balloon else None}
                for box, balloon in zip(boxes, found)
            ]
        )

    @app.post("/api/letters")
    def letters():
        """A mask of the lettering itself, pixel by pixel, as a PNG.

        White on clear: opaque where the ink is, transparent everywhere else, so
        it can be laid over the page or drawn into a canvas. Send it back to
        /api/clean to hide the letters and leave the art they sit on.
        """
        image = page()
        grow = number("grow", GROW, 0, GROW_MAX)
        mask = Image.fromarray(detector().letters(np.array(image), grow), mode="L")
        out = Image.new("RGBA", mask.size, (255, 255, 255, 0))
        out.putalpha(mask)
        return png(out)

    @app.post("/api/read")
    def read():
        """What the lettering in each box says, in the order they were given."""
        image = page()
        return jsonify(texts=reader()(image, boxes_in(image)))

    @app.post("/api/clean")
    def clean():
        """The page back with what was marked taken out of it: the lettering hidden.

        What is marked can be boxes, a mask, or both. A mask is a greyscale page
        of the same size, and it is the only way to say "this bubble but not that
        corner of it".

        What goes in its place is `fill`: by default the art around the mark,
        carried inwards; send `fill=white` to paint it flat instead.
        """
        image = page()
        mask = mask_in(image)
        boxes = boxes_in(image) if "boxes" in request.form else []
        if mask is None and not boxes:
            raise BadRequest("nothing to hide: send 'boxes', a 'mask', or both")

        return png(render.hidden(image, render.marked(image.size, boxes, mask), fill_in()))

    @app.post("/api/render")
    def overlay():
        """The page back with every box hidden and its text set in its place.

        `fill` says what a box is hidden under, as it does for /api/clean, but
        white is the default here: a box is a rectangle and the text set in it is
        black, and black lettering wants a ground that is clear.
        """
        image = page()
        regions = [
            render.Region(
                box=box_in(region.get("box"), image), text=str(region.get("text", ""))
            )
            for region in sent("regions")
        ]
        return png(render.overlay(image, regions, font, fill_in(render.WHITE_OUT)))

    return app
