"""Translating the lettering with a model running under Ollama.

Nothing leaves the machine Ollama is on. Which model does the work is the
caller's choice out of whatever has been pulled there.

The page goes over in one request rather than one per line: a line of manga read
on its own often cannot be translated at all, having no idea who is speaking or
about what. The model is held to a JSON schema so the answers come back
countable, and if it loses count anyway the lines are asked about one at a time.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

OLLAMA_ENV = "MANGA_TRANS_OLLAMA"
OLLAMA_DEFAULT = "http://localhost:11434"

TARGET_DEFAULT = "English"
SOURCE_DEFAULT = "Japanese"
TIMEOUT = 600
LISTING_TIMEOUT = 15

SCHEMA = {
    "type": "object",
    "properties": {"translations": {"type": "array", "items": {"type": "string"}}},
    "required": ["translations"],
}

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


class Unreachable(RuntimeError):
    """Ollama is not answering where it was expected to be."""


def base(explicit: str | None = None) -> str:
    """Where Ollama is: the one asked for, the one set, else this machine."""
    return (explicit or os.environ.get(OLLAMA_ENV) or OLLAMA_DEFAULT).rstrip("/")


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


def answered(message: dict) -> list | None:
    """The translations out of one reply.

    Ollama files a thinking model's reasoning under `thinking` and its answer
    under `content`, but a model held to a schema may not think at all — and then
    some builds put the whole answer under `thinking`. The answer is whichever
    field holds the JSON.
    """
    for field in ("content", "thinking"):
        found = as_json(message.get(field) or "")
        if isinstance(found, dict) and isinstance(found.get("translations"), list):
            return found["translations"]
    return None


def briefing(
    target: str, system: str | None = None, source: str = SOURCE_DEFAULT
) -> str:
    """What the model is told, with the languages filled in.

    Replaced rather than formatted: a hand-written prompt may have braces of its
    own, and str.format would choke on them.
    """
    said = system or SYSTEM_DEFAULT
    return said.replace("{target}", target).replace("{source}", source)


def request_for(
    lines: list[str],
    model: str,
    target: str,
    system: str | None = None,
    source: str = SOURCE_DEFAULT,
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
            {"role": "system", "content": briefing(target, system, source)},
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
) -> str:
    """One line on its own, for when a whole page came back miscounted."""
    body = request_for([text], model, target, system, source)
    sent = ask("/api/chat", body, host=host)
    message = sent["message"]
    got = answered(message)
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
) -> list[str]:
    """One translation per text, in the order they were given.

    An empty text stays empty and is never sent: there is nothing to translate,
    and a blank line only gives the model something to miscount.
    """
    wanted = [(at, text) for at, text in enumerate(texts) if text.strip()]
    done = [""] * len(texts)
    if not wanted:
        return done

    lines = [text for _, text in wanted]
    body = request_for(lines, model, target, system, source)
    got = answered(ask("/api/chat", body, host=host)["message"])

    if got is None or len(got) != len(lines):
        # It lost count. Asked one line at a time it cannot, though it loses the
        # rest of the page as context.
        got = [one(line, model, target, host, system, source) for line in lines]

    for (at, _), answer in zip(wanted, got):
        done[at] = str(answer).strip()
    return done
