"""The HTTP API. An image goes up, boxes or a rendered page comes back.

Nothing is stored. Every endpoint's fields and behaviour are in README.md.
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
from .detect import GROW, GROW_MAX, KINDS, Letters, Regions
from .geometry import Box
from .read import Reader

MAX_UPLOAD = 32 * 1024 * 1024
ORIGIN = os.environ.get("MANGA_TRANS_ORIGIN", "*")

T = TypeVar("T")


def lazily(make: Callable[[], T]) -> Callable[[], T]:
    """One instance, built on first use and shared thereafter."""
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

    Tried once: a miss is remembered, being a missing file rather than anything
    that might be there next time.
    """

    def make_or_none() -> T | None:
        try:
            return make()
        except Exception as exc:  # noqa: BLE001
            print(f"mangatrans: cleaning with telea instead of lama: {exc}")
            return None

    return lazily(make_or_none)


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


def maybe_sent(field: str) -> list | None:
    """The same, for a list a caller may simply not have."""
    if request.form.get(field) is None:
        return None
    return sent(field)


def terms_in(glossary: list | None) -> list[dict] | None:
    """A glossary as it arrives, refused rather than half-read if it is malformed.

    Strict where `ollama.noted` is lenient: that reads a model's answer, this
    reads a caller's request.
    """
    if glossary is None:
        return None
    terms = []
    for term in glossary:
        if not isinstance(term, dict):
            raise BadRequest("'glossary' must be a list of {source, target} objects")
        source, target = term.get("source"), term.get("target")
        if not isinstance(source, str) or not isinstance(target, str):
            raise BadRequest("every term in 'glossary' needs a source and a target")
        if not (source.strip() and target.strip()):
            continue
        note = term.get("note")
        settled = {"source": source.strip(), "target": target.strip()}
        if isinstance(note, str) and note.strip():
            settled["note"] = note.strip()[: ollama.NOTE_LIMIT]
        terms.append(settled)
    return terms


def object_in(field: str, shape: str) -> dict | None:
    """A JSON object sent in the form field ``field``, or None if none was."""
    raw = request.form.get(field)
    if raw is None:
        return None
    try:
        said = json.loads(raw)
    except ValueError as exc:
        raise BadRequest(f"'{field}' is not valid JSON: {exc}") from exc
    if not isinstance(said, dict):
        raise BadRequest(f"'{field}' must be a JSON object of {shape}")
    return said


def people_in(cast, field: str) -> list[dict]:
    """A cast as it arrives, refused rather than half-read if it is malformed.

    Shared by the story a caller carries page to page and the one a survey
    worked out: they are the same shape on purpose.
    """
    if not isinstance(cast, list):
        raise BadRequest(f"'{field}.cast' must be a list")
    people = []
    for person in cast:
        if not isinstance(person, dict):
            raise BadRequest(f"every one of '{field}.cast' must be an object")
        name = str(person.get("name") or "").strip()
        if not name:
            raise BadRequest(f"every one of '{field}.cast' needs a name")
        gender = person.get("gender", ollama.UNKNOWN)
        if gender not in ollama.GENDERS:
            raise BadRequest(
                f"a cast gender must be one of: {', '.join(ollama.GENDERS)}"
            )
        settled = person.get("settled") or []
        if not isinstance(settled, list) or any(
            fact not in ollama.FACTS for fact in settled
        ):
            raise BadRequest(
                f"'settled' must be a list of: {', '.join(ollama.FACTS)}"
            )
        people.append(
            {
                "name": name,
                "gender": gender,
                "note": str(person.get("note") or "").strip()[
                    : ollama.CAST_NOTE_LIMIT
                ],
                "settled": settled,
            }
        )
    return people[: ollama.CAST_LIMIT]


def story_in():
    """Where the chapter had got to, as the caller has it.

    `settled` is the caller's own — a fact set by hand — and is carried through
    so the prompt can say so.
    """
    said = object_in("previously", "{scene, cast}")
    if said is None:
        return None
    return {
        "scene": str(said.get("scene") or "").strip()[: ollama.SCENE_LIMIT],
        "cast": people_in(said.get("cast") or [], "previously"),
    }


def chapter_in():
    """What a survey made of the whole chapter, as the caller has it.

    `beats` is one line per page and positional with them, so it is carried
    whole and cut only at the far end.
    """
    said = object_in("chapter", "{synopsis, register, beats, cast, terms}")
    if said is None:
        return None
    beats = said.get("beats") or []
    if not isinstance(beats, list):
        raise BadRequest("'chapter.beats' must be a list, one line per page")
    return {
        "synopsis": str(said.get("synopsis") or "").strip()[: ollama.SYNOPSIS_LIMIT],
        "register": str(said.get("register") or "").strip()[: ollama.REGISTER_LIMIT],
        "beats": [
            str(beat or "").strip()[: ollama.BEAT_LIMIT]
            for beat in beats[: ollama.BEATS_LIMIT]
        ],
        "cast": people_in(said.get("cast") or [], "chapter"),
        "terms": terms_in(said.get("terms") or []) or [],
    }


def pages_in() -> list[list[str]]:
    """A chapter's lettering to survey: one list of lines per page, in order.

    A page with nothing on it is kept rather than dropped: the answer is one
    beat per page given.
    """
    pages = sent("pages")
    read = []
    for page in pages:
        if not isinstance(page, list):
            raise BadRequest("'pages' must be a list of pages, each a list of lines")
        read.append([str(line) for line in page])
    return read


def whole_in(field: str) -> int:
    """A whole number sent in a form field, or 0 where none was."""
    raw = request.form.get(field, "").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError as exc:
        raise BadRequest(f"'{field}' must be a whole number") from exc


def beside(field: str, texts: list[str]) -> list | None:
    """A list sent alongside `texts` and lined up with it, or None if none was.

    Refused rather than padded when the counts differ: one that has slipped by a
    place describes the wrong line, and nothing downstream can tell.
    """
    values = maybe_sent(field)
    if values is None:
        return None
    if len(values) != len(texts):
        raise BadRequest(
            f"'{field}' has {len(values)} for {len(texts)} texts — "
            "one per text, in the same order"
        )
    return values


def kinds_in(texts: list[str]) -> list[str] | None:
    """What each line is, in the words /api/detect answers with."""
    values = beside("kinds", texts)
    if values is None:
        return None
    kinds = []
    for kind in values:
        if kind in (None, ""):
            kinds.append("")
        elif kind in KINDS.values():
            kinds.append(str(kind))
        else:
            raise BadRequest(
                f"'kinds' must be empty or one of: {', '.join(KINDS.values())}"
            )
    return kinds


def budgets_in(texts: list[str]) -> list[int] | None:
    """How many characters each line has room for where it will be lettered."""
    values = beside("budgets", texts)
    if values is None:
        return None
    budgets = []
    for budget in values:
        try:
            budgets.append(max(0, int(budget)))
        except (TypeError, ValueError) as exc:
            raise BadRequest("every budget must be a whole number") from exc
    return budgets


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
    """What the page is lettered in. Nothing sent means Japanese."""
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
    except (TypeError, ValueError, OverflowError) as exc:
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
        if sent_mask.size != image.size:
            raise BadRequest(
                f"the mask is {sent_mask.width}×{sent_mask.height} "
                f"but the page is {image.width}×{image.height}"
            )
        sent_mask.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise BadRequest(f"the mask is not a usable image: {exc}") from exc

    alpha = sent_mask.getchannel("A") if "A" in sent_mask.getbands() else None
    shaped = alpha is not None and alpha.getextrema()[0] < 255
    return alpha if shaped else sent_mask.convert("L")


def fill_in(default: str = render.ART) -> str:
    """What goes where the lettering was: the art around it, or flat white."""
    chosen = request.form.get("fill", default).strip().lower()
    if chosen not in render.FILLS:
        raise BadRequest(f"'fill' must be one of: {', '.join(render.FILLS)}")
    return chosen


def png(image: Image.Image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=1)
    buffer.seek(0)
    return send_file(buffer, mimetype="image/png", download_name="page.png")


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

    @app.errorhandler(HTTPException)
    def on_http_error(exc):
        response = exc.get_response()
        response.data = json.dumps({"error": exc.description})
        response.content_type = "application/json"
        return response

    @app.errorhandler(Exception)
    def on_error(exc):
        app.logger.exception("Unhandled exception")
        return jsonify(error="internal server error"), 500

    @app.after_request
    def allow_origin(response):
        """The front end is served from somewhere else, so it needs letting in."""
        response.headers["Access-Control-Allow-Origin"] = ORIGIN
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

    @app.get("/api/languages")
    def written():
        """Every language a page can be read in, and which way round it is read."""
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
        """What the model is told to do, unless a caller says otherwise."""
        return jsonify(prompt=ollama.SYSTEM_DEFAULT, survey=ollama.SURVEY_DEFAULT)

    @app.post("/api/translate")
    def translate():
        """One translation per text, in the order they were given."""
        texts = [str(text) for text in sent("texts")]
        model = request.form.get("model", "").strip()
        if not model:
            raise BadRequest("nothing to translate with (form field 'model')")
        target = request.form.get("target", "").strip() or ollama.TARGET_DEFAULT
        source = request.form.get("source", "").strip() or ollama.SOURCE_DEFAULT
        system = request.form.get("system", "").strip() or None
        glossary = terms_in(maybe_sent("glossary"))
        kinds, budgets = kinds_in(texts), budgets_in(texts)
        story = story_in()
        chapter, page = chapter_in(), whole_in("page")
        try:
            done = ollama.translate(
                texts,
                model,
                target,
                system=system,
                source=source,
                glossary=glossary,
                kinds=kinds,
                budgets=budgets,
                story=story,
                chapter=chapter,
                page=page,
            )
            return jsonify(texts=done.texts, terms=done.terms, story=done.story)
        except ollama.Unreachable as exc:
            raise ServiceUnavailable(str(exc)) from exc

    @app.post("/api/survey")
    def survey():
        """What a chapter is, read a windowful at a time before any is translated."""
        pages = pages_in()
        model = request.form.get("model", "").strip()
        if not model:
            raise BadRequest("nothing to survey with (form field 'model')")
        target = request.form.get("target", "").strip() or ollama.TARGET_DEFAULT
        source = request.form.get("source", "").strip() or ollama.SOURCE_DEFAULT
        system = request.form.get("system", "").strip() or None
        try:
            found = ollama.survey(
                pages,
                model,
                target,
                system=system,
                source=source,
                chapter=chapter_in(),
                first=whole_in("first"),
            )
            return jsonify(
                chapter={
                    "synopsis": found.synopsis,
                    "register": found.register,
                    "beats": found.beats,
                    "cast": found.cast,
                    "terms": found.terms,
                }
            )
        except ollama.Unreachable as exc:
            raise ServiceUnavailable(str(exc)) from exc

    @app.post("/api/detect")
    def detect():
        """Every block of lettering on the page, boxed, with the balloon it is in."""
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
                    "kind": block.kind,
                }
                for block, room in zip(blocks, rooms)
            ],
        )

    @app.post("/api/bubbles")
    def balloons():
        """The balloon each box is written in, boxed, in the order they were given.

        The answer depends on which *other* boxes were asked about, so a caller
        that will letter with it must send **every** box on the page.
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
        """A mask of the lettering itself, pixel by pixel, as a PNG: white on clear."""
        image = page()
        grow = number("grow", GROW, 0, GROW_MAX)
        mask = Image.fromarray(letters_of()(np.array(image), grow), mode="L")
        out = Image.new("RGBA", mask.size, (255, 255, 255, 0))
        out.putalpha(mask)
        return png(out)

    @app.post("/api/read")
    def read():
        """What the lettering in each box says, in the order they were given."""
        image = page()
        return jsonify(texts=reader()(image, boxes_in(image), language_in()))

    @app.post("/api/clean")
    def clean():
        """The page back with what was marked taken out of it: the lettering hidden."""
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

        `fill` defaults to white here, the other way round from /api/clean.
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
