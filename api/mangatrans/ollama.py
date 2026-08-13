"""Translating the lettering with a model running under Ollama.

Nothing leaves the machine Ollama is on. Which model does the work is the
caller's choice out of whatever has been pulled there.

The page goes over in one request rather than one per line: a line of manga read
on its own often cannot be translated at all, having no idea who is speaking or
about what. The model is held to a JSON schema so the answers come back
countable, and if it loses count anyway the lines are asked about one at a time.

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
) -> str:
    """The whole system message: the prompt, the terms note, the glossary."""
    return "\n\n".join(
        part
        for part in (briefing(target, system, source), TERMS_NOTE, settled(glossary))
        if part
    )


def request_for(
    lines: list[str],
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
        "options": {"temperature": 0.2},
        "format": SCHEMA,
        "messages": [
            {"role": "system", "content": told(target, system, source, glossary)},
            {
                "role": "user",
                "content": "\n".join(
                    f"{number}. {line}" for number, line in enumerate(lines, 1)
                ),
            },
        ],
    }


def one(
    text: str,
    model: str,
    target: str,
    host=None,
    system: str | None = None,
    source: str = SOURCE_DEFAULT,
    glossary: list[dict] | None = None,
) -> str:
    """One line on its own, for when a whole page came back miscounted."""
    body = request_for([text], model, target, system, source, glossary)
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
) -> tuple[list[str], list[dict]]:
    """One translation per text in the order given, and the terms the page named.

    An empty text stays empty and is never sent: there is nothing to translate,
    and a blank line only gives the model something to miscount.
    """
    wanted = [(at, text) for at, text in enumerate(texts) if text.strip()]
    done = [""] * len(texts)
    if not wanted:
        return done, []

    lines = [text for _, text in wanted]
    body = request_for(lines, model, target, system, source, glossary)
    reply = answered(ask("/api/chat", body, host=host)["message"])
    got = reply["translations"] if reply else None
    # Kept even when the count was wrong: naming a character does not depend on
    # counting the lines, and the pass below has no page left to find one in.
    terms = noted(reply)

    if got is None or len(got) != len(lines):
        # It lost count. Asked one line at a time it cannot, though it loses the
        # rest of the page as context.
        got = [
            one(line, model, target, host, system, source, glossary) for line in lines
        ]

    for (at, _), answer in zip(wanted, got):
        done[at] = str(answer).strip()
    return done, terms
