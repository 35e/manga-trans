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

The same answer carries back the names and terms the page introduced, and the
caller sends back what it has collected so far. That is what keeps a chapter
consistent across pages the model never sees together — and it rides on the one
request rather than a second, which over a folder run would double the calls.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass

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
                },
                "required": ["source", "target"],
            },
        },
    },
    "required": ["translations"],
}

# How many terms are worth putting in front of a page. A chapter's recurring cast
# is a couple of dozen names; past that it is vocabulary, and a prompt long enough
# to hold it crowds out the page it is meant to help.
GLOSSARY_LIMIT = 40

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
    "wording you just used for it. Recurring ones only: nothing that is ordinary "
    "vocabulary, and nothing already listed as settled below."
)

GLOSSARY_HEADING = "Terms already used in this chapter. Translate them the same way:"

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
        if source and target:
            terms.append({"source": source, "target": target})
    return terms[:GLOSSARY_LIMIT]


def briefing(
    target: str, system: str | None = None, source: str = SOURCE_DEFAULT
) -> str:
    """What the model is told, with the languages filled in.

    Replaced rather than formatted: a hand-written prompt may have braces of its
    own, and str.format would choke on them.
    """
    said = system or SYSTEM_DEFAULT
    return said.replace("{target}", target).replace("{source}", source)


def settled(glossary: list[dict] | None) -> str:
    """The terms already decided, as lines to put in front of the page."""
    if not glossary:
        return ""
    lines = "\n".join(
        f"{term['source']} = {term['target']}" for term in glossary[:GLOSSARY_LIMIT]
    )
    return f"{GLOSSARY_HEADING}\n{lines}"


def told(
    target: str,
    system: str | None,
    source: str,
    glossary: list[dict] | None = None,
    kinds: bool = False,
    budgets: bool = False,
) -> str:
    """The whole system message: the prompt, the notes that apply, the glossary."""
    return "\n\n".join(
        part
        for part in (
            briefing(target, system, source),
            TERMS_NOTE,
            KINDS_NOTE if kinds else "",
            BUDGET_NOTE if budgets else "",
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
) -> dict:
    return {
        "model": model,
        "stream": False,
        # Thinking off where the model allows it: on a page of twenty lines it is
        # the difference between seconds and minutes, and none of it is wanted.
        "think": False,
        "options": {"temperature": 0.2, "num_ctx": CONTEXT},
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
                ),
            },
            {
                "role": "user",
                "content": "\n".join(
                    marked(number, line) for number, line in enumerate(lines, 1)
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
) -> str:
    """One line on its own, for a page that came back miscounted twice."""
    body = request_for([line], model, target, system, source, glossary)
    sent = ask("/api/chat", body, host=host)
    message = sent["message"]
    reply = answered(message)
    got = reply["translations"] if reply else None
    if got:
        return str(got[0]).strip()
    return (message.get("content") or "").strip()


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
) -> tuple[list[str], list[dict]]:
    """One translation per text in the order given, and the terms the page named.

    `kinds` and `budgets` are positional with `texts` where they are sent at all:
    what the detector made of each block, and how much room its translation has.

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
        return done, []

    lines = [line for _, line in wanted]
    body = request_for(lines, model, target, system, source, glossary)
    said = ask("/api/chat", body, host=host)["message"]
    reply = answered(said)
    got = counted(reply, len(lines))
    # Kept even when the count was wrong: naming a character does not depend on
    # counting the lines, and neither pass below has a page left to find one in.
    terms = noted(reply)

    if got is None:
        # It lost count. Shown its own answer and asked again it usually holds,
        # and a page asked again is still a page — every line still read against
        # the ones around it, which is what the fallback below gives up.
        gave = len(reply["translations"]) if reply else 0
        asked_again = corrected(body, said, gave, len(lines))
        again = answered(ask("/api/chat", asked_again, host=host)["message"])
        got = counted(again, len(lines))
        terms = terms or noted(again)

    if got is None:
        # Twice. Asked one line at a time it cannot miscount, though every line
        # loses the rest of the page it was to be read against.
        got = [
            one(line, model, target, host, system, source, glossary) for line in lines
        ]

    for (at, _), answer in zip(wanted, got):
        done[at] = str(answer).strip()
    return done, terms
