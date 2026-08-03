"""Translating a page's text with a local ollama model."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

DEFAULT_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:12b")
TIMEOUT = 120.0

SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["id", "text"],
            },
        }
    },
    "required": ["translations"],
}

SYSTEM_PROMPT = (
    "You are a professional manga translator. You are given the text of every "
    "speech bubble on one manga page, in reading order, each with an id. "
    "Translate each one into {target}.\n"
    "- Return exactly one translation per id, reusing the same ids.\n"
    "- Translate each bubble on its own; never merge or split bubbles.\n"
    "- Keep the tone of the original and render sound effects as {target} "
    "onomatopoeia.\n"
    "- Keep it short: it has to fit back into the same bubble.\n"
    "- Output the translation only, with no notes, romaji or explanations."
)


class OllamaError(RuntimeError):
    pass


def extract_json(text: str) -> dict:
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    candidates = [text]
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise OllamaError(f"ollama did not return JSON: {text[:200]!r}")


def chat(prompt: str, *, url: str, model: str, target: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.format(target=target)},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "think": False,
        "format": SCHEMA,
        "options": {"temperature": 0},
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            body = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace").strip()
        raise OllamaError(f"ollama returned HTTP {exc.code}: {detail[:200]}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise OllamaError(f"cannot reach ollama at {url}: {exc}") from exc

    message = body.get("message") or {}
    return (message.get("content") or message.get("thinking") or "").strip()


def clean(text: str) -> str:
    return re.sub(r"^\s*\d+[.)]\s*", "", text).strip()


def translate(
    texts: list[str],
    *,
    url: str = DEFAULT_URL,
    model: str = DEFAULT_MODEL,
    target: str = "English",
) -> list[str]:
    """Translate one page of bubbles, keeping the results aligned with the input."""
    wanted = [i for i, text in enumerate(texts) if text.strip()]
    if not wanted:
        return ["" for _ in texts]

    prompt = "\n".join(f"{i + 1}. {texts[i]}" for i in wanted)
    reply = extract_json(chat(prompt, url=url, model=model, target=target))
    by_id = {}
    for item in reply.get("translations", []):
        try:
            by_id[int(item["id"])] = clean(str(item["text"]))
        except (KeyError, TypeError, ValueError):
            continue

    out = ["" for _ in texts]
    for i in wanted:
        translated = by_id.get(i + 1, "")
        if not translated:
            single = extract_json(
                chat(f"1. {texts[i]}", url=url, model=model, target=target)
            ).get("translations", [])
            translated = clean(str(single[0]["text"])) if single else ""
        out[i] = translated
    return out
