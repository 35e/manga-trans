"""Translating the lettering with a model running under Ollama.

Nothing leaves the machine Ollama is on. Which model does the work is the
caller's choice out of whatever has been pulled there.

The page goes over in one request rather than one per line: a line of manga read
on its own often cannot be translated at all, having no idea who is speaking or
about what. The model is held to a JSON schema so the answers come back
countable; a page that comes back miscounted is shown what it did and asked
again, and only a second miscount falls back to one line at a time — which always
works and is the worst translation this can produce, having thrown away the page
each line was to be read against.

Each line carries what the detector made of it and how much room it has to be
lettered into. Both are things a translator has and a model reading a bare list
of strings does not: whether a line is spoken at all, and how long it may be.

The same answer carries back the names and terms the page introduced and where the
chapter has got to, and the caller sends both back with the next page. That is
what keeps a chapter consistent across pages the model never sees together — and
it rides on the one request rather than a second, which over a folder run would
double the calls. The terms are collected; the story is rewritten each page, so
neither grows with the chapter.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace

OLLAMA_ENV = "MANGA_TRANS_OLLAMA"

# Where to look when nothing says. Ollama runs on the machine, not in the
# container, and what that machine is called from inside one depends on what is
# running it: Docker answers to host.docker.internal, Podman to
# host.containers.internal. Naming either in the image leaves the other with a
# name that does not resolve — and a page that reads perfectly and then cannot be
# translated at all — so each is tried and whichever answers is the one used.
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
# Long enough to answer, short enough that all three misses are a wait rather
# than a hang: nothing is listening is a refused connection or an unknown name,
# and both come back at once.
FINDING_TIMEOUT = 5

# How much context to ask for. Ollama's own default is 4096 tokens and it drops
# what will not fit rather than saying so, front first — which is the briefing and
# the glossary, in a request that is otherwise a page of dialogue. Japanese runs
# to a token or two a character, so a busy page with a chapter's names in front of
# it is not far off it. What that costs is invisible: the terms quietly stop being
# honoured, and a briefing half gone is a page that comes back miscounted and is
# then translated a line at a time with nothing around it. The KV cache this asks
# for is a few hundred MB at most, and only while a page is being translated.
CONTEXT = 8192

# As much as an answer could honestly need: a page of forty balloons, its terms
# and two sentences of summary come to well under a thousand tokens. It is here as
# the backstop for a model that has begun looping, which is the only thing the
# repetition penalty turned off in `request_for` was doing for us — bounded, it is
# a page that comes back unreadable JSON and is asked again, rather than one that
# holds the request open until the ten-minute timeout.
PREDICT = 4096

# `terms` is deliberately not required. The count — one translation per line, in
# order — is what this whole module protects, and it must not start failing over
# a model that saw nothing worth naming on a page of "...".
SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {"type": "array", "items": {"type": "string"}},
        "terms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    # Who or what it is, where that is what decides the wording:
                    # a name is rendered one way for a boy and another for a
                    # teacher, and the page that named him is the only one that
                    # ever sees which he was.
                    "note": {"type": "string"},
                },
                "required": ["source", "target"],
            },
        },
        # The story so far, rewritten with this page in it. Not required, for the
        # same reason `terms` is not.
        #
        # `gender` is an enum with `unknown` in it rather than a free string, and
        # that is the whole point of the shape: asked for prose a model has to
        # pick a pronoun, so the first page guesses, the guess is read back as
        # established fact by every page after it, and the page that finally shows
        # otherwise is rewriting a text that already disagrees with it. Here
        # saying nothing is a value.
        "story": {
            "type": "object",
            "properties": {
                "scene": {"type": "string"},
                "cast": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "gender": {
                                "type": "string",
                                "enum": ["male", "female", "unknown"],
                            },
                            "note": {"type": "string"},
                        },
                        "required": ["name", "gender"],
                    },
                },
            },
        },
    },
    "required": ["translations"],
}

# How many terms are worth putting in front of a page. A chapter's recurring cast
# is a couple of dozen names; past that it is vocabulary, and a prompt long enough
# to hold it crowds out the page it is meant to help.
GLOSSARY_LIMIT = 40

# A few words on who someone is, not a biography: it is said again on every page
# of the chapter, forty terms at a time.
NOTE_LIMIT = 80

# The scene is rewritten each page rather than added to, so it does not grow —
# but a model asked for two sentences may write ten, and this rides on every page
# of the chapter. Cut rather than refused: a scene that stops mid sentence still
# says who is in it, and there is no second chance to ask.
SCENE_LIMIT = 600

# How many people a chapter is carried with. A scene has a handful of speakers and
# this is said again on every page of a folder run; past a dozen it is a crowd
# nobody is tracking, and the names that matter came first.
CAST_LIMIT = 12

# What is known about someone, and what is not.
MALE, FEMALE, UNKNOWN = "male", "female", "unknown"
GENDERS = (MALE, FEMALE, UNKNOWN)

# What can be known about one of the cast, and so settled by a caller who already
# knows it. Per fact rather than per person: someone whose gender was set by hand
# is still someone the chapter can say new things about.
FACTS = ("gender", "note")

# Names that are not names. A model asked for someone the page never names will
# sometimes file them under "unknown", which is worse than useless: the name is
# the key the chapter is carried on, so it collides with the next unnamed
# character, and the question at the foot of the page comes out reading "Still
# unknown: 先輩, unknown". Measured — and it appears to be what stopped the model
# answering that question at all.
NOT_A_NAME = {"unknown", "unnamed", "none", "n/a", "?", "-", "—"}

# ``{target}`` is replaced by the language being translated into and ``{source}``
# by the one the page was lettered in. Saying the source is worth the words: the
# same characters are Japanese or Chinese depending on nothing the model can see
# from a line of dialogue, and asked to guess it will translate a Chinese page as
# though it were Japanese. The schema above is what makes the answer JSON,
# whatever this says; what the wording holds up is the count and the order, and
# losing those only costs a slower pass line by line. So this can be rewritten
# freely.
SYSTEM_DEFAULT = (
    "You translate {source} manga dialogue into {target}. You are given the lines "
    "of one page, in order, and they are one conversation: read them together. Reply "
    "with a JSON object holding one translation per line, in the same order, the "
    "same number of them. Keep it short enough to letter back into a speech "
    "bubble. Translate only: no notes, no romaji, no quotation marks around the "
    "line."
)

# Said on every request, whatever the prompt is. It belongs here rather than in
# SYSTEM_DEFAULT because the prompt above is the caller's to replace: a
# hand-written one would not mention terms, and the glossary would quietly stop
# working the moment anyone edited it. This is about the shape of the answer, the
# way the schema is, rather than about how to translate.
TERMS_NOTE = (
    "Also list, under `terms`, any name, place, honorific or invented word on this "
    "page that a later page would have to render the same way, each with the "
    "wording you just used for it. Where who or what it is decides that wording, "
    "add a few words of `note` saying which — 'the younger brother', 'how a pupil "
    "addresses a teacher' — since no later page can see this one. Recurring ones "
    "only: nothing that is ordinary vocabulary, and nothing already settled below."
)

# Said on every request, the same as TERMS_NOTE and for the same reason: a page
# read on its own has no idea who is speaking, and a chapter is read one page at a
# time by something that never sees two of them together. `scene` is rewritten
# each page rather than added to, so forty pages carry two sentences rather than
# eighty.
#
# The two sentences about guessing are what this note is really for. A chapter
# tells you who someone is when it is ready to, and a translation of page one made
# by guessing is wrong on every page until the guess is corrected — which, being
# handed back its own summary each time, it rarely is. `unknown` costs nothing and
# is filled in by the page that settles it.
STORY_NOTE = (
    "Also give, under `story`, where the chapter has got to. `scene` is one or two "
    "sentences: who is present, what is going on between them, anything a page not "
    "yet read would need. `cast` is everyone who speaks or is spoken about, with "
    "their `gender` and a few words of `note` on who they are. Name each of them "
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

# Said only when the lines actually carry a kind, since a note about markers that
# are not there is a note about nothing. The last sentence is doing real work: a
# model told a line is a sound effect is a model that may decide it needs no
# translation and answer for the other nineteen, which is the one failure this
# module cannot let through.
KINDS_NOTE = (
    "Each line is marked [speech] where the lettering is inside a balloon and "
    "[free] where it is not — a sound effect, a caption, a sign, a shout across the "
    "art. A [free] line is not someone talking: render a sound effect as a sound "
    "effect and a caption as narration, not as dialogue. Answer for every line, "
    "[free] ones included, and do not repeat the markers in your answer."
)

# Likewise. A translation longer than its balloon is not refused anywhere — it is
# lettered smaller until it fits, which is how a page ends up set in type nobody
# can read — so the length is worth saying while the words are still being chosen.
BUDGET_NOTE = (
    "A line marked <=N has room for about N characters where it will be lettered. "
    "Past that it has to be set too small to read, so say it in fewer words rather "
    "than running over. It is a ceiling and not a target: short is fine."
)


@dataclass(frozen=True)
class Line:
    """One block on its way over: what it says, what it is, what room it has.

    `kind` is the detector's word for it — speech or free lettering — and empty
    where nothing said. `budget` is roughly how many characters fit where the
    translation will be set, and None where the caller did not work it out.
    """

    text: str
    kind: str = ""
    budget: int | None = None


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

    Only a host that answered is remembered: a page may well be worked on before
    Ollama has been started, and a miss then must not settle the question for the
    life of the process.
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


def answered(message: dict) -> dict | None:
    """The whole answer out of one reply, translations and terms together.

    Ollama files a thinking model's reasoning under `thinking` and its answer
    under `content`, but a model held to a schema may not think at all — and then
    some builds put the whole answer under `thinking`. The answer is whichever
    field holds the JSON.

    What makes a reply an answer is still the translations: a page that came back
    with terms and nothing to letter is not one.
    """
    for field in ("content", "thinking"):
        found = as_json(message.get(field) or "")
        if isinstance(found, dict) and isinstance(found.get("translations"), list):
            return found
    return None


def text_of(message: dict) -> str:
    """Whatever a reply said, wherever it filed it — see :func:`answered`."""
    return (message.get("content") or message.get("thinking") or "").strip()


def noted(reply: dict | None) -> list[dict]:
    """The terms out of an answer, keeping only the ones that are actually a pair.

    `terms` is optional and unpoliced by the schema beyond its shape, so a model
    that files something else there costs nothing rather than reaching the caller.
    """
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
        # Only carried when there is one: a term is a pair, and the note is a
        # convenience on top of it that a model may simply not offer.
        note = str(term.get("note") or "").strip()[:NOTE_LIMIT]
        terms.append(
            {"source": source, "target": target, "note": note}
            if note
            else {"source": source, "target": target}
        )
    return terms[:GLOSSARY_LIMIT]


def peopled(cast) -> list[dict]:
    """The cast out of an answer, everyone who is at least a name.

    A gender this does not know becomes `unknown` rather than being refused or
    passed on: the enum in the schema is what is meant to keep it to the three,
    and this is the belt to that pair of braces. `unknown` is the honest reading
    of a word nobody here recognises.
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
        note = str(person.get("note") or "").strip()[:NOTE_LIMIT]
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

    Lenient in the same way :func:`noted` is, and for the same reason — this reads
    a model's answer rather than a caller's request. A story that came back as
    something else costs nothing rather than reaching the caller.
    """
    said = reply.get("story") if reply else None
    if not isinstance(said, dict):
        return {}
    scene = str(said.get("scene") or "").strip()[:SCENE_LIMIT]
    cast = peopled(said.get("cast"))
    return {"scene": scene, "cast": cast} if (scene or cast) else {}


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

    `settled` is the load-bearing word: it marks what was decided outside this
    conversation — set by hand — and STORY_NOTE tells the model those are not its
    to change. Without it a page that reads ambiguously talks the chapter back out
    of something someone already knew. It follows the fact it is about, since a
    caller may know someone's gender and nothing else about them.
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

    Measured, and the reason this exists: told in the system message to correct
    what it is given, a model handed a page that says 「先輩は僕の兄です」 —
    senpai is my older brother — translates that line correctly and hands the cast
    straight back with senpai still unknown. Standing instructions are read as
    describing the job; a question under the page is read as being about the page.
    So the names are put where the evidence is, and what counts as evidence is
    named rather than left as "correct it".
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
    cast = [
        described(person)
        for person in (story.get("cast") or [])[:CAST_LIMIT]
        if isinstance(person, dict) and str(person.get("name") or "").strip()
    ]
    parts = []
    if scene:
        parts.append(f"{PREVIOUSLY_HEADING}\n{scene}")
    if cast:
        parts.append(CAST_HEADING + "\n" + "\n".join(cast))
    return "\n\n".join(parts)


def told(
    target: str,
    system: str | None,
    source: str,
    glossary: list[dict] | None = None,
    kinds: bool = False,
    budgets: bool = False,
    story: dict | None = None,
) -> str:
    """The whole system message: the prompt, the notes that apply, what is known.

    The notes come before the chapter, and the chapter before the page: what to
    do, then what has happened, then what is on the page.
    """
    return "\n\n".join(
        part
        for part in (
            briefing(target, system, source),
            TERMS_NOTE,
            # The one note with a language in it: the cast is named in the page's
            # own script, so the note has to say which that is.
            filled(STORY_NOTE, target, source),
            KINDS_NOTE if kinds else "",
            BUDGET_NOTE if budgets else "",
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
) -> dict:
    return {
        "model": model,
        "stream": False,
        # Thinking off where the model allows it: on a page of twenty lines it is
        # the difference between seconds and minutes, and none of it is wanted.
        "think": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": CONTEXT,
            # Off, against a default of 1.1. A page repeats itself on purpose —
            # the same shout in two balloons, a catchphrase, a row of "……" — and
            # a penalty on saying a thing twice is a push to render the second
            # one differently from the first for no reason but that it came
            # second. What it is nominally there to stop is a model looping,
            # which num_predict below bounds without touching the wording.
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
                        asking(story),
                    )
                    if part
                ),
            },
        ],
    }


def counted(reply: dict | None, wanted: int) -> list[str] | None:
    """The translations out of an answer, if it came back with one for every line."""
    got = reply.get("translations") if reply else None
    if not isinstance(got, list) or len(got) != wanted:
        return None
    return [str(line) for line in got]


def corrected(body: dict, said: dict, got: int, wanted: int) -> dict:
    """The same page again, with the miscounted answer and what was wrong with it.

    Shown its own reply rather than only asked again: what it got wrong was the
    count, which it cannot see from the request alone. The page stays in front of
    it either way, which is the whole difference between this and :func:`one`.
    """
    return {
        **body,
        "messages": [
            *body["messages"],
            {"role": "assistant", "content": text_of(said)},
            {
                "role": "user",
                "content": (
                    f"That was {got} translations for {wanted} lines. Answer again "
                    f"with exactly {wanted}, one for each numbered line, in the same "
                    "order. A line you would leave as it is still needs one: give "
                    "it back as it stands rather than dropping it."
                ),
            },
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
) -> str:
    """One line on its own, for a page that came back miscounted twice."""
    body = request_for([line], model, target, system, source, glossary, story)
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
    # Where the chapter has got to with this page in it, to hand back on the next:
    # `{"scene": ..., "cast": [...]}`, or empty where the page settled nothing.
    story: dict = field(default_factory=dict)


def asked_once(wanted: list[tuple[int, Line]]) -> tuple[list[Line], list[int]]:
    """The lines to send, and which sent line answers each block.

    The same words in the same kind of lettering are one question, however many
    balloons they fill: a page of "……" is asked about once. Two identical lines
    coming back different is the commonest way a page reads as though nobody
    checked it, and the model has no reason to render the second one differently
    beyond its having come second. Fewer lines is also less to miscount, which is
    what everything else here is defending.

    The tightest room of the identical blocks is the one asked for, since the one
    answer has to be lettered into all of them.
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
) -> Translation:
    """One translation per text in the order given, and what the page named.

    `kinds` and `budgets` are positional with `texts` where they are sent at all:
    what the detector made of each block, and how much room its translation has.
    `story` is where the chapter had got to before this page, and comes back
    rewritten with this page in it.

    An empty text stays empty and is never sent: there is nothing to translate,
    and a blank line only gives the model something to miscount.
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
    body = request_for(lines, model, target, system, source, glossary, story)
    said = ask("/api/chat", body, host=host)["message"]
    reply = answered(said)
    got = counted(reply, len(lines))
    # Kept even when the count was wrong: naming a character or saying what is
    # going on does not depend on counting the lines, and neither pass below has a
    # page left to find either in.
    terms, after = noted(reply), storied(reply)

    if got is None:
        # It lost count. Shown its own answer and asked again it usually holds,
        # and a page asked again is still a page — every line still read against
        # the ones around it, which is what the fallback below gives up.
        gave = len(reply["translations"]) if reply else 0
        asked_again = corrected(body, said, gave, len(lines))
        again = answered(ask("/api/chat", asked_again, host=host)["message"])
        got = counted(again, len(lines))
        terms = terms or noted(again)
        after = after or storied(again)

    if got is None:
        # Twice. Asked one line at a time it cannot miscount, though every line
        # loses the rest of the page it was to be read against.
        got = [
            one(line, model, target, host, system, source, glossary, story)
            for line in lines
        ]

    for (at, _), which in zip(wanted, where):
        done[at] = str(got[which]).strip()
    return Translation(done, terms, after)
