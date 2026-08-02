"""Translating a page's bubbles with a local ollama model."""

from __future__ import annotations

import json
import re

TRANSLATION_SCHEMA = {
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
    "Rules:\n"
    "- Return exactly one translation per id, reusing the same ids.\n"
    "- Translate each bubble on its own; never merge or split bubbles.\n"
    "- Keep the tone of the original (casual speech stays casual, shouting stays "
    "shouting) and render sound effects as {target} onomatopoeia.\n"
    "- Use the other bubbles only as context for pronouns and politeness.\n"
    "- Keep it short: it has to fit back into the same bubble.\n"
    "- Output the translation only, with no notes, romaji or explanations.\n"
    "- If a line is unreadable OCR noise, return it unchanged."
)


class OllamaError(RuntimeError):
    """Ollama could not be reached or returned something unusable."""


def extract_json(text: str) -> dict:
    """Parse a JSON object out of a model response.

    Thinking models like to wrap the answer in prose or ``<think>`` tags even
    when a schema is set, so fall back to the outermost ``{...}``.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise OllamaError(f"could not parse ollama's response as JSON: {text[:200]!r}")


def ollama_chat(
    messages: list[dict],
    *,
    url: str,
    model: str,
    schema: dict | None = None,
    timeout: float = 120.0,
    think: bool = False,
) -> str:
    """POST to Ollama's /api/chat and return the assistant's answer.

    ``think=False`` matters: on a thinking model like qwen3, reasoning about a
    handful of speech bubbles costs a minute or more per page and adds nothing
    to a translation.
    """
    import urllib.error  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": False,
        # Translation should not be creative.
        "options": {"temperature": 0},
    }
    if schema is not None:
        payload["format"] = schema
    if think is not None:
        payload["think"] = think

    def post(body: dict) -> dict:
        request = urllib.request.Request(
            url.rstrip("/") + "/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    try:
        try:
            body = post(payload)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace").strip()
            if "think" not in detail.lower() or "think" not in payload:
                raise OllamaError(f"ollama returned HTTP {exc.code}: {detail}") from exc
            # Model has no thinking mode - ask again without the field.
            payload.pop("think")
            body = post(payload)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace").strip()
        raise OllamaError(f"ollama returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise OllamaError(
            f"cannot reach ollama at {url} ({exc.reason}).\n"
            "  - is `ollama serve` running?\n"
            "  - from a container the host is http://host.containers.internal:11434\n"
            "  - override with --ollama-url or $OLLAMA_URL"
        ) from exc
    except TimeoutError as exc:
        raise OllamaError(
            f"ollama timed out after {timeout}s; raise --ollama-timeout"
        ) from exc

    message = body.get("message") or {}
    content = (message.get("content") or "").strip()
    if not content:
        # Some models (qwen3-vl) put the whole answer in `thinking` and leave
        # `content` empty when thinking is disabled.
        content = (message.get("thinking") or "").strip()
    return content


def clean_translation(text: str) -> str:
    """Drop the list numbering models sometimes echo back ("1. Hello" -> "Hello")."""
    return re.sub(r"^\s*\d+[.)]\s*", "", text).strip()


def translate_texts(
    texts: list[str],
    *,
    url: str,
    model: str,
    target_lang: str = "English",
    timeout: float = 120.0,
    log=lambda _msg: None,
) -> list[str]:
    """Translate one page worth of bubbles, keeping them aligned with the input.

    The whole page goes in one request so the model has the surrounding bubbles
    as context. A JSON schema keeps the ids intact; anything the model still
    drops is retried on its own.
    """
    if not any(t.strip() for t in texts):
        return ["" for _ in texts]

    numbered = "\n".join(
        f"{i}. {text}" for i, text in enumerate(texts, start=1) if text.strip()
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(target=target_lang)},
        {"role": "user", "content": numbered},
    ]
    content = ollama_chat(
        messages, url=url, model=model, schema=TRANSLATION_SCHEMA, timeout=timeout
    )

    by_id: dict[int, str] = {}
    try:
        for item in extract_json(content).get("translations", []):
            by_id[int(item["id"])] = clean_translation(str(item["text"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise OllamaError(f"unexpected response shape: {content[:200]!r}") from exc

    out: list[str] = []
    for i, text in enumerate(texts, start=1):
        if not text.strip():
            out.append("")
            continue
        translated = by_id.get(i, "")
        if not translated:
            # The model skipped this bubble - ask again for just this one.
            log(f"  retrying bubble {i}...")
            single = ollama_chat(
                [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT.format(target=target_lang),
                    },
                    {"role": "user", "content": f"1. {text}"},
                ],
                url=url,
                model=model,
                schema=TRANSLATION_SCHEMA,
                timeout=timeout,
            )
            try:
                items = extract_json(single).get("translations", [])
                translated = clean_translation(str(items[0]["text"])) if items else ""
            except (OllamaError, KeyError, IndexError, TypeError, ValueError):
                translated = ""
        out.append(translated)
    return out
