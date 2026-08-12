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

from . import bubble, inpaint, languages, ollama, render
from .detect import GROW, GROW_MAX, Letters, Regions
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


def optionally(make: Callable[[], T]) -> Callable[[], T | None]:
    """:func:`lazily`, for something the API can do without.

    LaMa is the best fill there is and it is not the only one, so an image built
    without its weights cleans with Telea rather than answering 500. Tried once:
    a miss is remembered, since it is a missing file rather than anything that
    might be there next time.
    """
    held: list[T | None] = []
    get = lazily(make)

    def maybe() -> T | None:
        if not held:
            try:
                held.append(get())
            except Exception as exc:  # noqa: BLE001 — any failure means "do without"
                print(f"mangatrans: cleaning with telea instead of lama: {exc}")
                held.append(None)
        return held[0]

    return maybe


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


def language_in() -> languages.Language:
    """What the page is lettered in, which decides who reads it and which way round.

    Nothing sent means Japanese: this was written for manga first, and a caller
    that predates any of the rest still means that.
    """
    asked = request.form.get("language")
    try:
        return languages.of(asked)
    except KeyError as exc:
        raise BadRequest(
            f"'language' is not one this reads: {asked!r} "
            f"(try one of {', '.join(languages.CODES)})"
        ) from exc


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
    font: str | None = None,
    model: str | None = None,
    ocr_model: str | None = None,
    regions_model: str | None = None,
    lama_model: str | None = None,
) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD
    font = font or os.environ.get("MANGA_TRANS_FONT")

    regions_of = lazily(lambda: Regions(regions_model))
    letters_of = lazily(lambda: Letters(model))
    reader = lazily(lambda: Reader(ocr_model))
    painter = optionally(lambda: inpaint.Lama(lama_model))

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

    @app.get("/api/languages")
    def written():
        """Every language a page can be read in, and which way round it is read.

        Handed out so a front end can offer them rather than hold its own copy of
        the list: which reader exists for what is the API's business.
        """
        return jsonify(
            languages=[
                {"code": language.code, "name": language.name, "rtl": language.rtl}
                for language in languages.LANGUAGES
            ]
        )

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
        default, with `{target}` and `{source}` anywhere the languages should go.

        `source` and `target` are language names rather than codes: they are only
        ever words in a prompt, and a caller may well be translating something
        this API has no reader for.
        """
        texts = [str(text) for text in sent("texts")]
        model = request.form.get("model", "").strip()
        if not model:
            raise BadRequest("nothing to translate with (form field 'model')")
        target = request.form.get("target", "").strip() or ollama.TARGET_DEFAULT
        source = request.form.get("source", "").strip() or ollama.SOURCE_DEFAULT
        system = request.form.get("system", "").strip() or None
        try:
            said = ollama.translate(texts, model, target, system=system, source=source)
            return jsonify(texts=said)
        except ollama.Unreachable as exc:
            raise ServiceUnavailable(str(exc)) from exc

    @app.post("/api/detect")
    def detect():
        """Every block of lettering on the page, boxed, with the balloon it is in.

        The box is where the words are; `bubble` is the room they were written
        in, which is what a translation wants — see /api/bubbles. It comes back
        here too because it is worked out from the page and the boxes alone, and
        both are already in hand.

        Finding the text needs no model of the language, but the order they come
        back in does: `language` says which way across the page it is read.
        """
        image = page()
        pixels = np.array(image)
        blocks, balloons = regions_of()(pixels, language_in().rtl)
        rooms = bubble.rooms(pixels, [block.box for block in blocks], balloons)
        return jsonify(
            width=image.width,
            height=image.height,
            regions=[
                {
                    "box": block.box.as_list(),
                    "confidence": round(block.confidence, 3),
                    "bubble": room.as_list() if room else None,
                }
                for block, room in zip(blocks, rooms)
            ],
        )

    @app.post("/api/bubbles")
    def balloons():
        """The balloon each box is written in, boxed, in the order they were given.

        Japanese and much Chinese run down the page, so a block of either is a
        tall narrow column and a translation set in that column has nowhere to
        go. This answers with the room around it instead. No language is named:
        a balloon is a shape on the page, and finding one reads nothing.

        `bubble` is null where no balloon could be made out — lettering over
        artwork is in none — and the caller should keep the box it has.

        The answer depends on which *other* boxes were asked about, because two
        blocks in one balloon are cut a side each rather than both handed the
        whole of it. So anything whose answer will be lettered with has to send
        every box on the page, not only the ones that changed.
        """
        image = page()
        pixels = np.array(image)
        boxes = boxes_in(image)
        _, balloons = regions_of()(pixels)
        found = bubble.rooms(pixels, boxes, balloons)
        return jsonify(
            regions=[
                {"box": box.as_list(), "bubble": room.as_list() if room else None}
                for box, room in zip(boxes, found)
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
        mask = Image.fromarray(letters_of()(np.array(image), grow), mode="L")
        out = Image.new("RGBA", mask.size, (255, 255, 255, 0))
        out.putalpha(mask)
        return png(out)

    @app.post("/api/read")
    def read():
        """What the lettering in each box says, in the order they were given.

        `language` is what it is lettered in, out of /api/languages, and decides
        which reader is stood up: manga-ocr for Japanese, PP-OCR for the rest.
        Only the reader asked for is ever loaded.
        """
        image = page()
        return jsonify(texts=reader()(image, boxes_in(image), language_in()))

    @app.post("/api/clean")
    def clean():
        """The page back with what was marked taken out of it: the lettering hidden.

        What is marked can be boxes, a mask, or both. A mask is a greyscale page
        of the same size, and it is the only way to say "this bubble but not that
        corner of it".

        What goes in its place is `fill`: by default the art around the mark,
        carried inwards by a LaMa trained on manga; `fill=telea` for the same
        without a model, and `fill=white` to paint it flat instead.
        """
        image = page()
        mask = mask_in(image)
        boxes = boxes_in(image) if "boxes" in request.form else []
        if mask is None and not boxes:
            raise BadRequest("nothing to hide: send 'boxes', a 'mask', or both")

        marks = render.marked(image.size, boxes, mask)
        return png(render.hidden(image, marks, fill_in(), painter()))

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
        return png(
            render.overlay(
                image, regions, font, fill_in(render.WHITE_OUT), painter()
            )
        )

    return app
