"""Translating the lettering with a model running under Ollama.

The whole page goes over in one request, held to a JSON schema so the answers
come back countable; :func:`survey` reads a chapter first, a window at a time,
so a page can be translated against all of it. See DOCS.md.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace

OLLAMA_ENV = "MANGA_TRANS_OLLAMA"

OLLAMA_HOSTS = (
    "http://localhost:11434",
    "http://host.docker.internal:11434",
    "http://host.containers.internal:11434",
)
OLLAMA_DEFAULT = OLLAMA_HOSTS[0]

TARGET_DEFAULT = "English"
SOURCE_DEFAULT = "Japanese"
TIMEOUT = 600
LISTING_TIMEOUT = 15
FINDING_TIMEOUT = 5

CONTEXT = 12288

PREDICT = 4096

TERM_SCHEMA = {
    "type": "object",
    "properties": {
        "source": {"type": "string"},
        "target": {"type": "string"},
        "note": {"type": "string"},
    },
    "required": ["source", "target"],
}

CAST_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "gender": {"type": "string", "enum": ["male", "female", "unknown"]},
        "note": {"type": "string"},
    },
    "required": ["name", "gender"],
}

SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {"type": "array", "items": {"type": "string"}},
        "terms": {"type": "array", "items": TERM_SCHEMA},
        "story": {
            "type": "object",
            "properties": {
                "scene": {"type": "string"},
                "cast": {"type": "array", "items": CAST_SCHEMA},
            },
        },
    },
    "required": ["translations"],
}

SURVEY_SCHEMA = {
    "type": "object",
    "properties": {
        "beats": {"type": "array", "items": {"type": "string"}},
        "synopsis": {"type": "string"},
        "register": {"type": "string"},
        "terms": {"type": "array", "items": TERM_SCHEMA},
        "cast": {"type": "array", "items": CAST_SCHEMA},
    },
    "required": ["beats"],
}

GLOSSARY_LIMIT = 40
NOTE_LIMIT = 80
SCENE_LIMIT = 600
CAST_LIMIT = 12

CAST_NOTE_LIMIT = 200

SURVEY_CONTEXT = 16384

SYNOPSIS_LIMIT = 1200
REGISTER_LIMIT = 200
BEAT_LIMIT = 160
BEATS_LIMIT = 400

BEATS_BEFORE, BEATS_AFTER = 6, 2

MALE, FEMALE, UNKNOWN = "male", "female", "unknown"
GENDERS = (MALE, FEMALE, UNKNOWN)

FACTS = ("gender", "note")

NOT_A_NAME = {"unknown", "unnamed", "none", "n/a", "?", "-", "—"}

SYSTEM_DEFAULT = (
    "You translate {source} manga dialogue into {target}. You are given the lines "
    "of one page, in order, and they are one conversation: read them together. Reply "
    "with a JSON object holding one translation per line, in the same order, the "
    "same number of them. Keep it short enough to letter back into a speech "
    "bubble. Translate only: no notes, no romaji, no quotation marks around the "
    "line."
)

TERMS_NOTE = (
    "Also list, under `terms`, any name, place, honorific or invented word on this "
    "page that a later page would have to render the same way, each with the "
    "wording you just used for it. Where who or what it is decides that wording, "
    "add a few words of `note` saying which — 'the younger brother', 'how a pupil "
    "addresses a teacher' — since no later page can see this one. Recurring ones "
    "only: nothing that is ordinary vocabulary, and nothing already settled below."
)

STORY_NOTE = (
    "Also give, under `story`, where the chapter has got to. `scene` is one or two "
    "sentences: who is present, what is going on between them, anything a page not "
    "yet read would need. `cast` is everyone who speaks or is spoken about, with "
    "their `gender` and a `note` saying who they are: their part in the story, "
    "who they are to the others, and how they speak. Give the note at that "
    "length whatever you were handed — a fuller one shortened is a character "
    "lost for every page after this one. Name each of them "
    "as the pages do, in the {source} they are written in — 先輩, 田中先生 — "
    "and "
    "keep that name exactly from page to page, since it is what says who is who. "
    "Somebody the pages never name at all is named by what they are — 'the boy on "
    "the bicycle' — and never as `unknown`, which is not a name. Answer "
    "`unknown` for anyone the chapter has not actually shown to be one or the "
    "other, and that is most people at first: do not guess it from a name, from a "
    "manner of speaking, or from how someone is addressed. An `unknown` costs "
    "nothing and the page that settles it will fill it in; a guess is read as "
    "fact by every page after this one. What you are given below was written from "
    "earlier pages and may be wrong: where this page shows otherwise, correct it. "
    "Anything marked settled is known and is not yours to change. Give the cast "
    "whole each time, not only what is new."
)

GLOSSARY_HEADING = "Terms already used in this chapter. Translate them the same way:"
PREVIOUSLY_HEADING = "The story so far, from the pages before this one:"
CAST_HEADING = "Who is in it:"

KINDS_NOTE = (
    "Each line is marked [speech] where the lettering is inside a balloon and "
    "[free] where it is not — a sound effect, a caption, a sign, a shout across the "
    "art. A [free] line is not someone talking: render a sound effect as a sound "
    "effect and a caption as narration, not as dialogue. Answer for every line, "
    "[free] ones included, and do not repeat the markers in your answer."
)

BUDGET_NOTE = (
    "A line marked <=N has room for about N characters where it will be lettered. "
    "Past that it has to be set too small to read, so say it in fewer words rather "
    "than running over. It is a ceiling and not a target: short is fine."
)

SURVEY_DEFAULT = (
    "You are reading a chapter of a {source} comic before it is translated into "
    "{target}. You are given the lettering of several pages, in the order it is "
    "read, one page after another. There are no pictures: a page is what is said "
    "on it and nothing more, so say what can be told from that and do not invent "
    "the rest. You are not translating here — this is what the translator will be "
    "handed before they start."
)

SURVEY_NOTE = (
    "Reply with a JSON object. `beats` is one line for each page you are given, in "
    "the same order and the same number of them: what happens on that page, "
    "plainly, and a page where nothing does still needs one. `synopsis` is the "
    "chapter as far as you have read it, in a few sentences — what it is about, who "
    "it is between, where it is going. `register` is how it is written: how formal, "
    "whose voice it is told in, when and where it is set, anything that decides "
    "wording. `terms` is every name, place, honorific or invented word that will "
    "have to be rendered the same way each time, each with the wording you would "
    "use for it in {target} and a few words of `note` on who or what it is where "
    "that is what decides the wording. `cast` is everyone who speaks or is spoken "
    "about, named as the pages name them, in the {source} they are written in — "
    "先輩, 田中先生 — since that name is what says who is who. Somebody the pages "
    "never name is named by what they are — 'the boy on the bicycle' — and never as "
    "`unknown`, which is not a name. Give each of them a `note` saying who they "
    "are: their part in the chapter, who they are to the others, and how they "
    "speak — how formal, how blunt, what they call people. That note is all a "
    "page being translated will know about them, so write it for someone who has "
    "not read the chapter. Answer `unknown` for the gender of anyone the "
    "chapter has not actually shown to be one or the other: do not guess it from a "
    "name, from a manner of speaking, or from how someone is addressed. A later "
    "page settles it, and a guess made here is read as fact by every page of the "
    "translation. Where you are given what the earlier pages came to, write the "
    "synopsis, the register, the cast and the terms again with these pages in them "
    "rather than starting over; the beats are only for the pages here."
)

CHAPTER_NOTE = (
    "You are shown the whole chapter, the pages after this one included, so that a "
    "word can be chosen knowing where it leads — which pronoun someone takes, which "
    "of two readings a name has, how much weight a line is carrying. Translate only "
    "the lines you are given, and say no more than they say: a line must not be made "
    "to hint at anything the reader has not reached. Where the {source} is vague and "
    "the chapter settles what it meant, render it as the chapter settles it. Where "
    "the {source} is vague on purpose, leave it vague."
)

SYNOPSIS_HEADING = "What this chapter is, whole — the pages after this one included:"
SO_FAR_HEADING = "What the pages up to here came to. Write it again with these in it:"
REGISTER_HEADING = "How it is written:"
BEATS_HEADING = "The chapter page by page. The page you are translating is marked →:"

MISCOUNTED = (
    "That was {got} translations for {wanted} lines. Answer again with exactly "
    "{wanted}, one for each numbered line, in the same order. A line you would "
    "leave as it is still needs one: give it back as it stands rather than "
    "dropping it."
)
MISBEATEN = (
    "That was {got} beats for {wanted} pages. Answer again with exactly {wanted}, "
    "one for each page, in the same order. A page where nothing happens still needs "
    "one: say so rather than dropping it."
)


@dataclass(frozen=True)
class Line:
    """One block on its way over: what it says, what it is, what room it has."""

    text: str
    kind: str = ""
    budget: int | None = None


@dataclass(frozen=True)
class Chapter:
    """What a survey made of a chapter, and what its pages are translated against.

    `beats` is one line per page and positional with them, which is why a window
    that miscounts hands back none rather than storing them a page out.
    """

    beats: list[str] = field(default_factory=list)
    synopsis: str = ""
    register: str = ""
    cast: list[dict] = field(default_factory=list)
    terms: list[dict] = field(default_factory=list)


class Unreachable(RuntimeError):
    """Ollama is not answering where it was expected to be."""


_answering: str | None = None
_finding = threading.Lock()


def base(explicit: str | None = None) -> str:
    """Where Ollama is: the one asked for, the one set, else wherever it answers."""
    said = explicit or os.environ.get(OLLAMA_ENV)
    return said.rstrip("/") if said else answering()


def answering() -> str:
    """The first of the usual places Ollama answers at, kept once one does.

    Only a hit is remembered: Ollama is as often started after the dashboard as
    before it, so a miss must not settle the question for the life of the process.
    """
    global _answering
    with _finding:
        if _answering:
            return _answering
        for host in OLLAMA_HOSTS:
            try:
                ask("/api/tags", timeout=FINDING_TIMEOUT, host=host)
            except Unreachable:
                continue
            _answering = host
            return host
    raise Unreachable(
        f"no ollama answering at any of {', '.join(OLLAMA_HOSTS)} — "
        f"start it, or set {OLLAMA_ENV} to say where it is"
    )


def ask(path: str, body: dict | None = None, timeout: int = TIMEOUT, host=None) -> dict:
    """One call to Ollama, GET when there is nothing to send."""
    where = base(host)
    request = urllib.request.Request(
        f"{where}{path}",
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            return json.load(answer)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:200]
        raise Unreachable(f"ollama at {where} answered {exc.code}: {detail}") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise Unreachable(f"no ollama answering at {where}: {exc}") from exc


def models(host=None) -> list[str]:
    """Every model pulled on that Ollama, by name."""
    listing = ask("/api/tags", timeout=LISTING_TIMEOUT, host=host)
    return sorted(
        model["name"] for model in listing.get("models", []) if model.get("name")
    )


def as_json(text: str):
    """The JSON in a model's answer, fenced or prefaced or neither."""
    text = text.strip()
    if not text:
        return None
    if text.startswith("```"):
        text = text.strip("`")
        _, _, text = text.partition("\n")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except ValueError:
        return None


def answered(message: dict, key: str = "translations") -> dict | None:
    """The whole answer out of one reply, translations and terms together.

    Some Ollama builds put the whole answer under `thinking` rather than
    `content`, so both are read.
    """
    for where in ("content", "thinking"):
        found = as_json(message.get(where) or "")
        if isinstance(found, dict) and isinstance(found.get(key), list):
            return found
    return None


def text_of(message: dict) -> str:
    """Whatever a reply said, wherever it filed it — see :func:`answered`."""
    return (message.get("content") or message.get("thinking") or "").strip()


def noted(reply: dict | None) -> list[dict]:
    """The terms out of an answer, keeping only the ones that are actually a pair."""
    if not reply:
        return []
    found = reply.get("terms")
    if not isinstance(found, list):
        return []
    terms = []
    for term in found:
        if not isinstance(term, dict):
            continue
        source = str(term.get("source") or "").strip()
        target = str(term.get("target") or "").strip()
        if not (source and target):
            continue
        note = str(term.get("note") or "").strip()[:NOTE_LIMIT]
        terms.append(
            {"source": source, "target": target, "note": note}
            if note
            else {"source": source, "target": target}
        )
    return terms[:GLOSSARY_LIMIT]


def peopled(cast) -> list[dict]:
    """The cast out of an answer, everyone who is at least a name.

    A gender this does not know becomes `unknown`: the belt to the schema's
    braces, and the honest reading of a word nobody here recognises.
    """
    if not isinstance(cast, list):
        return []
    people = []
    for person in cast:
        if not isinstance(person, dict):
            continue
        name = str(person.get("name") or "").strip()
        if not name or name.casefold() in NOT_A_NAME:
            continue
        gender = person.get("gender")
        note = str(person.get("note") or "").strip()[:CAST_NOTE_LIMIT]
        people.append(
            {
                "name": name,
                "gender": gender if gender in GENDERS else UNKNOWN,
                "note": note,
            }
        )
    return people[:CAST_LIMIT]


def storied(reply: dict | None) -> dict:
    """The story so far out of an answer: the scene, and who is in it.

    Lenient the way :func:`noted` is: this reads a model's answer, not a
    caller's request.
    """
    said = reply.get("story") if reply else None
    if not isinstance(said, dict):
        return {}
    scene = str(said.get("scene") or "").strip()[:SCENE_LIMIT]
    cast = peopled(said.get("cast"))
    return {"scene": scene, "cast": cast} if (scene or cast) else {}


def surveyed(reply: dict | None) -> Chapter:
    """What a survey window made of the chapter, its beats aside.

    The beats go through :func:`beaten` instead: they are the one part of this
    with a count to hold.
    """
    if not reply:
        return Chapter()
    return Chapter(
        synopsis=str(reply.get("synopsis") or "").strip()[:SYNOPSIS_LIMIT],
        register=str(reply.get("register") or "").strip()[:REGISTER_LIMIT],
        cast=peopled(reply.get("cast")),
        terms=noted(reply),
    )


def beaten(reply: dict | None, wanted: int) -> list[str] | None:
    """One beat for each page sent, or None where the window lost count."""
    got = counted(reply, wanted, "beats")
    if got is None:
        return None
    return [beat.strip()[:BEAT_LIMIT] for beat in got]


def briefing(
    target: str, system: str | None = None, source: str = SOURCE_DEFAULT
) -> str:
    """What the model is told, with the languages filled in.

    Replaced rather than formatted: a hand-written prompt may have braces of its
    own, and str.format would choke on them.
    """
    return filled(system or SYSTEM_DEFAULT, target, source)


def filled(said: str, target: str, source: str) -> str:
    """The two languages put into whatever says `{target}` or `{source}`."""
    return said.replace("{target}", target).replace("{source}", source)


def settled(glossary: list[dict] | None) -> str:
    """The terms already decided, as lines to put in front of the page."""
    if not glossary:
        return ""
    lines = "\n".join(
        f"{term['source']} = {term['target']}"
        + (f"  ({term['note']})" if term.get("note") else "")
        for term in glossary[:GLOSSARY_LIMIT]
    )
    return f"{GLOSSARY_HEADING}\n{lines}"


def described(person: dict) -> str:
    """One of the cast as a line to put in front of the page.

    `settled` is the load-bearing word: it marks what was set by hand, and
    STORY_NOTE tells the model those are not its to change.
    """
    settled = person.get("settled") or ()
    gender = person.get("gender") if person.get("gender") in GENDERS else UNKNOWN
    if "gender" in settled:
        gender = f"{gender} (settled)"
    note = str(person.get("note") or "").strip()
    if note and "note" in settled:
        note = f"{note} (settled)"
    return f"{person['name']} — {gender}" + (f", {note}" if note else "")


def asking(story: dict | None) -> str:
    """Who is still unknown, asked at the foot of the page rather than in the brief.

    Measured, and the reason this exists: a standing instruction is read as
    describing the job, where a question under the page is read as being about
    the page. Do not move this into the briefing.
    """
    if not isinstance(story, dict):
        return ""
    waiting = [
        person["name"]
        for person in (story.get("cast") or [])[:CAST_LIMIT]
        if isinstance(person, dict)
        and str(person.get("name") or "").strip()
        and person.get("gender", UNKNOWN) == UNKNOWN
        and "gender" not in (person.get("settled") or ())
    ]
    if not waiting:
        return ""
    return (
        f"Still unknown: {', '.join(waiting)}. Does anything on this page settle "
        "who one of them is — a pronoun, a word for a brother, a sister, a son or "
        "a daughter, the way someone is addressed? If it does, say so in `story`. "
        "If it does not, leave them unknown."
    )


def previously(story: dict | None) -> str:
    """Where the chapter had got to, as the page before this one left it."""
    if not isinstance(story, dict):
        return ""
    scene = str(story.get("scene") or "").strip()[:SCENE_LIMIT]
    cast = described_cast(story)
    parts = []
    if scene:
        parts.append(f"{PREVIOUSLY_HEADING}\n{scene}")
    if cast:
        parts.append(CAST_HEADING + "\n" + "\n".join(cast))
    return "\n\n".join(parts)


def described_cast(chapter: dict) -> list[str]:
    """The cast of a chapter as lines, whoever is at least a name."""
    return [
        described(person)
        for person in (chapter.get("cast") or [])[:CAST_LIMIT]
        if isinstance(person, dict) and str(person.get("name") or "").strip()
    ]


def chaptered(chapter: dict | None, page: int = 0) -> str:
    """The chapter in front of one of its pages, its beats windowed around it."""
    if not isinstance(chapter, dict):
        return ""
    parts = []
    synopsis = str(chapter.get("synopsis") or "").strip()[:SYNOPSIS_LIMIT]
    if synopsis:
        parts.append(f"{SYNOPSIS_HEADING}\n{synopsis}")
    register = str(chapter.get("register") or "").strip()[:REGISTER_LIMIT]
    if register:
        parts.append(f"{REGISTER_HEADING}\n{register}")

    beats = chapter.get("beats")
    if isinstance(beats, list):
        said = [str(beat or "").strip()[:BEAT_LIMIT] for beat in beats[:BEATS_LIMIT]]
        lines = "\n".join(
            f"{'→' if at == page else ' '} {at + 1}. {said[at]}"
            for at in range(max(0, page - BEATS_BEFORE), min(len(said), page + BEATS_AFTER + 1))
            if said[at]
        )
        if lines:
            parts.append(f"{BEATS_HEADING}\n{lines}")
    return "\n\n".join(parts)


def placed(chapter: dict | None, page: int = 0) -> str:
    """Where in the chapter this page is, said under it rather than above it.

    The same measured finding :func:`asking` rests on.
    """
    if not isinstance(chapter, dict):
        return ""
    beats = chapter.get("beats")
    if not isinstance(beats, list) or not any(str(beat or "").strip() for beat in beats):
        return ""
    return (
        f"This is page {page + 1} of {len(beats)}. Translate it as the reader "
        "reaches it: they have read this far and no further, whatever you know "
        "from the pages after it."
    )


def gathered(chapter: dict | None) -> str:
    """What the pages read so far came to, in front of the next window of them.

    The beats already written are not put back: this window writes beats for its
    own pages only.
    """
    if not isinstance(chapter, dict):
        return ""
    parts = []
    synopsis = str(chapter.get("synopsis") or "").strip()[:SYNOPSIS_LIMIT]
    if synopsis:
        parts.append(f"{SO_FAR_HEADING}\n{synopsis}")
    register = str(chapter.get("register") or "").strip()[:REGISTER_LIMIT]
    if register:
        parts.append(f"{REGISTER_HEADING}\n{register}")
    cast = described_cast(chapter)
    if cast:
        parts.append(CAST_HEADING + "\n" + "\n".join(cast))
    terms = settled(chapter.get("terms"))
    if terms:
        parts.append(terms)
    return "\n\n".join(parts)


def told(
    target: str,
    system: str | None,
    source: str,
    glossary: list[dict] | None = None,
    kinds: bool = False,
    budgets: bool = False,
    story: dict | None = None,
    chapter: dict | None = None,
    page: int = 0,
) -> str:
    """The whole system message: the prompt, the notes that apply, what is known.

    Order matters — what to do, then what has happened, then what is on the page.
    """
    surveyed = chaptered(chapter, page)
    return "\n\n".join(
        part
        for part in (
            briefing(target, system, source),
            TERMS_NOTE,
            filled(STORY_NOTE, target, source),
            KINDS_NOTE if kinds else "",
            BUDGET_NOTE if budgets else "",
            filled(CHAPTER_NOTE, target, source) if surveyed else "",
            surveyed,
            previously(story),
            settled(glossary),
        )
        if part
    )


def marked(number: int, line: Line) -> str:
    """One line as it goes over: its number, what it is, how much room it has."""
    notes = []
    if line.kind:
        notes.append(f"[{line.kind}]")
    if line.budget:
        notes.append(f"<={line.budget}")
    return f"{number}. {' '.join(notes + [line.text])}"


def request_for(
    lines: list[Line],
    model: str,
    target: str,
    system: str | None = None,
    source: str = SOURCE_DEFAULT,
    glossary: list[dict] | None = None,
    story: dict | None = None,
    chapter: dict | None = None,
    page: int = 0,
) -> dict:
    return {
        "model": model,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": CONTEXT,
            "repeat_penalty": 1.0,
            "num_predict": PREDICT,
        },
        "format": SCHEMA,
        "messages": [
            {
                "role": "system",
                "content": told(
                    target,
                    system,
                    source,
                    glossary,
                    any(line.kind for line in lines),
                    any(line.budget for line in lines),
                    story,
                    chapter,
                    page,
                ),
            },
            {
                "role": "user",
                "content": "\n\n".join(
                    part
                    for part in (
                        "\n".join(
                            marked(number, line)
                            for number, line in enumerate(lines, 1)
                        ),
                        placed(chapter, page),
                        asking(story),
                    )
                    if part
                ),
            },
        ],
    }


def counted(
    reply: dict | None, wanted: int, key: str = "translations"
) -> list[str] | None:
    """One list out of an answer, if it came back with one entry for every line.

    One a place out is worse than none at all: nothing downstream can tell.
    """
    got = reply.get(key) if reply else None
    if not isinstance(got, list) or len(got) != wanted:
        return None
    return [str(line) for line in got]


def corrected(body: dict, said: dict, complaint: str) -> dict:
    """The same request again, with the miscounted answer and what was wrong with it.

    Shown its own reply rather than only asked again: the count is what it
    cannot see from the request alone.
    """
    return {
        **body,
        "messages": [
            *body["messages"],
            {"role": "assistant", "content": text_of(said)},
            {"role": "user", "content": complaint},
        ],
    }


def one(
    line: Line,
    model: str,
    target: str,
    host=None,
    system: str | None = None,
    source: str = SOURCE_DEFAULT,
    glossary: list[dict] | None = None,
    story: dict | None = None,
    chapter: dict | None = None,
    page: int = 0,
) -> str:
    """One line on its own, for a page that came back miscounted twice."""
    body = request_for(
        [line], model, target, system, source, glossary, story, chapter, page
    )
    sent = ask("/api/chat", body, host=host)
    message = sent["message"]
    reply = answered(message)
    got = reply["translations"] if reply else None
    if got:
        return str(got[0]).strip()
    return (message.get("content") or "").strip()


@dataclass(frozen=True)
class Translation:
    """A page translated: the lines, and what the page said about the chapter."""

    texts: list[str]
    terms: list[dict]
    story: dict = field(default_factory=dict)


def asked_once(wanted: list[tuple[int, Line]]) -> tuple[list[Line], list[int]]:
    """The lines to send, and which sent line answers each block.

    The same words in the same kind of lettering are one question, however many
    balloons they fill. The **tightest** budget of the identical blocks is the
    one sent, since the one answer has to fit all of them.
    """
    lines: list[Line] = []
    at_line: dict[tuple[str, str], int] = {}
    where: list[int] = []
    for _, line in wanted:
        same = (line.text, line.kind)
        seen = at_line.get(same)
        if seen is None:
            at_line[same] = len(lines)
            lines.append(line)
        elif line.budget and (
            lines[seen].budget is None or line.budget < lines[seen].budget
        ):
            lines[seen] = replace(lines[seen], budget=line.budget)
        where.append(at_line[same])
    return lines, where


def translate(
    texts: list[str],
    model: str,
    target: str = TARGET_DEFAULT,
    host=None,
    system: str | None = None,
    source: str = SOURCE_DEFAULT,
    glossary: list[dict] | None = None,
    kinds: list[str] | None = None,
    budgets: list[int] | None = None,
    story: dict | None = None,
    chapter: dict | None = None,
    page: int = 0,
) -> Translation:
    """One translation per text in the order given, and what the page named.

    `kinds` and `budgets` are positional with `texts`. An empty text stays empty
    and is never sent, so both must be carried along with the renumbering.
    """
    wanted = [
        (
            at,
            Line(
                text,
                kinds[at] if kinds else "",
                budgets[at] if budgets else None,
            ),
        )
        for at, text in enumerate(texts)
        if text.strip()
    ]
    done = [""] * len(texts)
    if not wanted:
        return Translation(done, [])

    lines, where = asked_once(wanted)
    body = request_for(
        lines, model, target, system, source, glossary, story, chapter, page
    )
    said = ask("/api/chat", body, host=host)["message"]
    reply = answered(said)
    got = counted(reply, len(lines))
    terms, after = noted(reply), storied(reply)

    if got is None:
        gave = len(reply["translations"]) if reply else 0
        asked_again = corrected(
            body, said, MISCOUNTED.format(got=gave, wanted=len(lines))
        )
        again = answered(ask("/api/chat", asked_again, host=host)["message"])
        got = counted(again, len(lines))
        terms = terms or noted(again)
        after = after or storied(again)

    if got is None:
        got = [
            one(line, model, target, host, system, source, glossary, story, chapter, page)
            for line in lines
        ]

    for (at, _), which in zip(wanted, where):
        done[at] = str(got[which]).strip()
    return Translation(done, terms, after)


def paged(pages: list[list[str]], first: int = 0) -> str:
    """A windowful as it goes over: each page numbered, its lettering under it.

    A page with nothing on it is still named, and still wants a beat.
    """
    return "\n\n".join(
        "\n".join(
            [f"Page {first + at + 1}:"]
            + [line.strip() for line in page if line.strip()]
        )
        for at, page in enumerate(pages)
    )


def survey_request_for(
    pages: list[list[str]],
    model: str,
    target: str,
    system: str | None = None,
    source: str = SOURCE_DEFAULT,
    chapter: dict | None = None,
    first: int = 0,
) -> dict:
    return {
        "model": model,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": SURVEY_CONTEXT,
            "repeat_penalty": 1.0,
            "num_predict": PREDICT,
        },
        "format": SURVEY_SCHEMA,
        "messages": [
            {
                "role": "system",
                "content": "\n\n".join(
                    part
                    for part in (
                        filled(system or SURVEY_DEFAULT, target, source),
                        filled(SURVEY_NOTE, target, source),
                        gathered(chapter),
                    )
                    if part
                ),
            },
            {"role": "user", "content": paged(pages, first)},
        ],
    }


def survey(
    pages: list[list[str]],
    model: str,
    target: str = TARGET_DEFAULT,
    host=None,
    system: str | None = None,
    source: str = SOURCE_DEFAULT,
    chapter: dict | None = None,
    first: int = 0,
) -> Chapter:
    """One windowful of a chapter, read before any of it is translated.

    The beats are counted the way the translations are: a second miscount hands
    back **no** beats rather than beats a page out.
    """
    if not pages:
        return Chapter()

    body = survey_request_for(pages, model, target, system, source, chapter, first)
    said = ask("/api/chat", body, host=host)["message"]
    reply = answered(said, "beats")
    beats = beaten(reply, len(pages))
    found = surveyed(reply)

    if beats is None:
        gave = len(reply["beats"]) if reply else 0
        asked_again = corrected(
            body, said, MISBEATEN.format(got=gave, wanted=len(pages))
        )
        again = answered(ask("/api/chat", asked_again, host=host)["message"], "beats")
        beats = beaten(again, len(pages))
        if not (found.synopsis or found.register or found.cast or found.terms):
            found = surveyed(again)

    return replace(found, beats=beats or [])
