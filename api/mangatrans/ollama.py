"""Translating the lettering with a model running under Ollama.

The whole page goes over in one request, held to a JSON schema so the answers
come back countable.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, replace

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

SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["translations"],
}

SYSTEM_DEFAULT = (
    "You translate {source} manga dialogue into {target}. You are given the lines "
    "of one page, in order, and they are one conversation: read them together. Reply "
    "with a JSON object holding one translation per line, in the same order, the "
    "same number of them. Keep it short enough to letter back into a speech "
    "bubble. Translate only: no notes, no romaji, no quotation marks around the "
    "line."
)

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

MISCOUNTED = (
    "That was {got} translations for {wanted} lines. Answer again with exactly "
    "{wanted}, one for each numbered line, in the same order. A line you would "
    "leave as it is still needs one: give it back as it stands rather than "
    "dropping it."
)


@dataclass(frozen=True)
class Line:
    """One block on its way over: what it says, what it is, what room it has."""

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


def answered(message: dict) -> dict | None:
    """The answer out of one reply.

    Some Ollama builds put the whole answer under `thinking` rather than
    `content`, so both are read.
    """
    for where in ("content", "thinking"):
        found = as_json(message.get(where) or "")
        if isinstance(found, dict) and isinstance(found.get("translations"), list):
            return found
    return None


def text_of(message: dict) -> str:
    """Whatever a reply said, wherever it filed it — see :func:`answered`."""
    return (message.get("content") or message.get("thinking") or "").strip()


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


def told(
    target: str,
    system: str | None,
    source: str,
    kinds: bool = False,
    budgets: bool = False,
) -> str:
    """The whole system message: the prompt, and the notes that apply to the page."""
    return "\n\n".join(
        part
        for part in (
            briefing(target, system, source),
            KINDS_NOTE if kinds else "",
            BUDGET_NOTE if budgets else "",
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
    """The translations out of an answer, if there is one for every line.

    One a place out is worse than none at all: nothing downstream can tell.
    """
    got = reply.get("translations") if reply else None
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
) -> str:
    """One line on its own, for a page that came back miscounted twice."""
    body = request_for([line], model, target, system, source)
    sent = ask("/api/chat", body, host=host)
    message = sent["message"]
    reply = answered(message)
    got = reply["translations"] if reply else None
    if got:
        return str(got[0]).strip()
    return (message.get("content") or "").strip()


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
    kinds: list[str] | None = None,
    budgets: list[int] | None = None,
) -> list[str]:
    """One translation per text in the order given.

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
        return done

    lines, where = asked_once(wanted)
    body = request_for(lines, model, target, system, source)
    said = ask("/api/chat", body, host=host)["message"]
    reply = answered(said)
    got = counted(reply, len(lines))

    if got is None:
        gave = len(reply["translations"]) if reply else 0
        asked_again = corrected(
            body, said, MISCOUNTED.format(got=gave, wanted=len(lines))
        )
        again = answered(ask("/api/chat", asked_again, host=host)["message"])
        got = counted(again, len(lines))

    if got is None:
        got = [one(line, model, target, host, system, source) for line in lines]

    for (at, _), which in zip(wanted, where):
        done[at] = str(got[which]).strip()
    return done
