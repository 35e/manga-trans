#!/usr/bin/env python3
"""OCR a manga page and return its text grouped per text box.

Pipeline
--------
1. **Detect**  - EasyOCR's CRAFT detector finds the individual text fragments
   (roughly one box per character run). Its own line-merging is disabled so we
   keep the raw fragments.
2. **Group**   - Fragments are clustered with single-linkage on the *edge-to-edge*
   gap between boxes. The gap threshold is relative to the glyph size of the two
   boxes being compared, so a small furigana cluster and a big SFX cluster both
   group sensibly on the same page. Fragments further apart than the threshold
   end up in different groups, i.e. different text boxes / speech bubbles.
3. **Recognise** - Each group is cropped as a whole and passed to `manga-ocr`,
   which is trained on complete manga text blocks and handles vertical text,
   multiple lines and furigana on its own.

Usage
-----
    python manga_ocr_groups.py page.jpg
    python manga_ocr_groups.py page.jpg --json out.json --viz boxes.png
    python manga_ocr_groups.py page.jpg --gap 1.6      # merge more aggressively
    python manga_ocr_groups.py page.jpg --no-ocr --viz boxes.png   # tune grouping fast

Arguments may also be folders - or left out entirely, in which case the current
folder is scanned. With ``--out-dir`` every page gets its own ``<name>.json``:

    python manga_ocr_groups.py pages --out-dir pages/out
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}

# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


@dataclass
class Box:
    """Axis-aligned box in pixel coordinates (x1/y1 exclusive-ish, inclusive is fine)."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def glyph_size(self) -> int:
        """Short side of the box ~ font size (char height for a horizontal run,
        char width for a vertical column)."""
        return max(1, min(self.w, self.h))

    def padded(self, px: int, width: int, height: int) -> "Box":
        return Box(
            max(0, self.x0 - px),
            max(0, self.y0 - px),
            min(width, self.x1 + px),
            min(height, self.y1 + px),
        )

    def as_list(self) -> list[int]:
        return [self.x0, self.y0, self.x1, self.y1]


def union_box(boxes: list[Box]) -> Box:
    return Box(
        min(b.x0 for b in boxes),
        min(b.y0 for b in boxes),
        max(b.x1 for b in boxes),
        max(b.y1 for b in boxes),
    )


def box_gap(a: Box, b: Box) -> float:
    """Edge-to-edge distance between two boxes (0 when they touch or overlap)."""
    dx = max(0, max(a.x0, b.x0) - min(a.x1, b.x1))
    dy = max(0, max(a.y0, b.y0) - min(a.y1, b.y1))
    return math.hypot(dx, dy)


# ---------------------------------------------------------------------------
# grouping
# ---------------------------------------------------------------------------


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[rj] = ri


def group_boxes(
    boxes: list[Box],
    gap_factor: float = 1.0,
    min_gap_px: float = 0.0,
    max_gap_px: float | None = None,
) -> list[list[int]]:
    """Cluster boxes by proximity; returns lists of indices into ``boxes``.

    Two boxes join the same group when the gap between them is at most
    ``gap_factor`` times the glyph size of the smaller one (clamped to
    [``min_gap_px``, ``max_gap_px``]). Grouping is single-linkage, so a chain of
    close fragments forms one text box.
    """
    uf = _UnionFind(len(boxes))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            threshold = gap_factor * min(a.glyph_size, b.glyph_size)
            threshold = max(threshold, min_gap_px)
            if max_gap_px is not None:
                threshold = min(threshold, max_gap_px)
            if box_gap(a, b) <= threshold:
                uf.union(i, j)

    clusters: dict[int, list[int]] = {}
    for idx in range(len(boxes)):
        clusters.setdefault(uf.find(idx), []).append(idx)
    return list(clusters.values())


@dataclass
class TextGroup:
    bbox: Box
    boxes: list[Box] = field(default_factory=list)
    text: str = ""
    translation: str = ""

    def to_dict(self) -> dict:
        return {
            "bbox": self.bbox.as_list(),
            "text": self.text,
            "translation": self.translation,
            "fragments": len(self.boxes),
            "fragment_boxes": [b.as_list() for b in self.boxes],
        }


def sort_reading_order(groups: list[TextGroup], order: str = "rtl") -> list[TextGroup]:
    """Best-effort reading order.

    ``rtl`` (default, normal for Japanese manga): top to bottom, right to left.
    Groups whose vertical centres are within a row tolerance are treated as
    being on the same row.
    """
    if order == "none" or not groups:
        return groups

    heights = [g.bbox.h for g in groups]
    row_tol = max(1.0, 0.6 * statistics.median(heights))
    sign = 1 if order == "ltr" else -1
    return sorted(groups, key=lambda g: (round(g.bbox.y0 / row_tol), sign * g.bbox.cx))


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------


def build_detector(gpu: bool, detect_network: str, verbose: bool):
    try:
        import easyocr  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise SystemExit(
            "easyocr is required for text detection: pip install easyocr"
        ) from exc

    kwargs = dict(gpu=gpu, detect_network=detect_network, verbose=verbose)
    try:
        # The recognition model is never used - manga-ocr does the reading.
        return easyocr.Reader(["ja", "en"], recognizer=False, **kwargs)
    except TypeError:
        return easyocr.Reader(["ja", "en"], **kwargs)


def detect_fragments(reader, image_grey, args) -> list[Box]:
    """Return raw text fragment boxes, with EasyOCR's own line merging disabled."""
    horizontal_agg, free_agg = reader.detect(
        image_grey,
        min_size=args.min_size,
        text_threshold=args.text_threshold,
        low_text=args.low_text,
        link_threshold=args.link_threshold,
        canvas_size=args.canvas_size,
        mag_ratio=args.mag_ratio,
        # Zeroed so EasyOCR does not merge fragments into "lines" using
        # horizontal-text assumptions - our own grouping handles that.
        slope_ths=0.0,
        ycenter_ths=0.0,
        height_ths=0.0,
        width_ths=0.0,
        add_margin=0.0,
    )

    boxes: list[Box] = []
    for x_min, x_max, y_min, y_max in horizontal_agg[0]:
        boxes.append(Box(int(x_min), int(y_min), int(x_max), int(y_max)))
    for poly in free_agg[0]:
        xs = [int(p[0]) for p in poly]
        ys = [int(p[1]) for p in poly]
        boxes.append(Box(min(xs), min(ys), max(xs), max(ys)))

    return [b for b in boxes if b.w > 0 and b.h > 0]


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------


@dataclass
class Page:
    path: Path
    width: int
    height: int
    fragments_detected: int
    groups: list[TextGroup]

    def to_dict(self) -> dict:
        return {
            "image": str(self.path),
            "width": self.width,
            "height": self.height,
            "fragments_detected": self.fragments_detected,
            "groups": [g.to_dict() for g in self.groups],
        }


def process_image(path: Path, reader, mocr, args) -> Page:
    from PIL import Image, ImageOps  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size

    grey = np.array(image.convert("L"))
    fragments = detect_fragments(reader, grey, args)

    groups: list[TextGroup] = []
    for indices in group_boxes(
        fragments,
        gap_factor=args.gap,
        min_gap_px=args.min_gap_px,
        max_gap_px=args.max_gap_px,
    ):
        members = [fragments[i] for i in indices]
        if len(members) < args.min_fragments:
            continue
        bbox = union_box(members)
        if bbox.w < args.min_group_px and bbox.h < args.min_group_px:
            continue
        groups.append(TextGroup(bbox=bbox, boxes=members))

    groups = sort_reading_order(groups, args.order)

    if mocr is not None:
        for group in groups:
            pad = max(2, int(round(args.pad * group.bbox.glyph_size)))
            crop_box = group.bbox.padded(pad, width, height)
            crop = image.crop((crop_box.x0, crop_box.y0, crop_box.x1, crop_box.y1))
            group.text = mocr(crop).strip()
        if args.drop_empty:
            groups = [g for g in groups if g.text]

    return Page(
        path=path,
        width=width,
        height=height,
        fragments_detected=len(fragments),
        groups=groups,
    )


def draw_visualisation(path: Path, page: Page, out_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont, ImageOps  # noqa: PLC0415

    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    draw = ImageDraw.Draw(image)
    scale = max(1, min(image.size) // 400)
    try:
        font = ImageFont.load_default(size=14 * scale)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()

    for group in page.groups:
        for frag in group.boxes:
            draw.rectangle(frag.as_list(), outline=(120, 180, 255), width=scale)

    for i, group in enumerate(page.groups, start=1):
        box = group.bbox
        draw.rectangle(box.as_list(), outline=(255, 40, 40), width=2 * scale)
        label = str(i)
        tx, ty = box.x0 + 2 * scale, max(0, box.y0 - 18 * scale)
        draw.rectangle(
            [tx - 2 * scale, ty, tx + 14 * scale, ty + 18 * scale], fill=(255, 40, 40)
        )
        draw.text((tx, ty), label, fill=(255, 255, 255), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


# ---------------------------------------------------------------------------
# translation (ollama)
# ---------------------------------------------------------------------------

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
                raise OllamaError(
                    f"ollama returned HTTP {exc.code}: {detail}"
                ) from exc
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


# ---------------------------------------------------------------------------
# text output
# ---------------------------------------------------------------------------


def page_lines(page: "Page", fmt: str = "both") -> list[str]:
    """The page's groups as text lines, in reading order."""
    lines = [f"# {page.path.name}"]
    for i, group in enumerate(page.groups, start=1):
        if fmt == "translation":
            if group.translation:
                lines.append(f"[{i}] {group.translation}")
        elif fmt == "original":
            if group.text:
                lines.append(f"[{i}] {group.text}")
        else:
            lines.append(f"[{i}] {group.text}")
            if group.translation:
                lines.append(f"    -> {group.translation}")
    return lines


def write_txt(pages: list["Page"], out_path: Path, fmt: str = "both") -> None:
    blocks = ["\n".join(page_lines(page, fmt)) for page in pages]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# input discovery
# ---------------------------------------------------------------------------


def natural_key(path: Path) -> tuple:
    """Sort key that orders page2 before page10.

    Each chunk is a same-shaped tuple so that a numeric chunk never has to be
    compared against a text one ("001.webp" next to "image.png").
    """
    chunks = [
        (0, int(c), "") if c.isdigit() else (1, 0, c)
        for c in re.split(r"(\d+)", path.name.lower())
        if c
    ]
    return (path.parent.as_posix(), chunks)


def collect_images(
    paths: list[Path], recursive: bool = False, skip_dir: Path | None = None
) -> list[Path]:
    """Expand folders into the image files inside them.

    Files given explicitly are always kept, whatever their extension. Folders
    contribute the images they contain (``recursive`` also walks sub-folders);
    anything under ``skip_dir`` - the output folder - is left out so results are
    never fed back in as input.
    """
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            entries = path.rglob("*") if recursive else path.glob("*")
            found.extend(
                p for p in entries if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            )
        elif path.is_file():
            found.append(path)
        else:
            raise SystemExit(f"no such file or folder: {path}")

    if skip_dir is not None and skip_dir.exists():
        skip = skip_dir.resolve()
        found = [p for p in found if skip not in p.resolve().parents]

    images: list[Path] = []
    seen: set[Path] = set()
    for path in sorted(found, key=natural_key):
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            images.append(path)
    return images


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OCR a manga page, grouping text by text box / speech bubble.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_input = Path(os.environ.get("MANGA_TRANS_INPUT", "."))
    default_out_dir = os.environ.get("MANGA_TRANS_OUT_DIR")

    parser.add_argument(
        "images",
        nargs="*",
        type=Path,
        default=[default_input],
        help="image file(s) or folder(s) to process; a folder is scanned for images",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="also scan sub-folders of the folders given",
    )

    grouping = parser.add_argument_group("grouping")
    grouping.add_argument(
        "--gap",
        type=float,
        default=1.0,
        help="max gap between two fragments of the same text box, "
        "in multiples of the glyph size (higher = merge more)",
    )
    grouping.add_argument("--min-gap-px", type=float, default=0.0, help="gap floor in px")
    grouping.add_argument(
        "--max-gap-px", type=float, default=None, help="gap ceiling in px"
    )
    grouping.add_argument(
        "--min-fragments",
        type=int,
        default=1,
        help="drop groups made of fewer detected fragments than this",
    )
    grouping.add_argument(
        "--min-group-px",
        type=int,
        default=12,
        help="drop groups smaller than this in both dimensions",
    )
    grouping.add_argument(
        "--order",
        choices=["rtl", "ltr", "none"],
        default="rtl",
        help="reading order of the returned groups (rtl = manga)",
    )

    detection = parser.add_argument_group("detection (EasyOCR/CRAFT)")
    detection.add_argument("--min-size", type=int, default=10)
    detection.add_argument("--text-threshold", type=float, default=0.7)
    detection.add_argument("--low-text", type=float, default=0.4)
    detection.add_argument("--link-threshold", type=float, default=0.4)
    detection.add_argument("--canvas-size", type=int, default=2560)
    detection.add_argument("--mag-ratio", type=float, default=1.0)
    detection.add_argument("--detect-network", default="craft", choices=["craft", "dbnet18"])

    recognition = parser.add_argument_group("recognition (manga-ocr)")
    recognition.add_argument(
        "--model",
        default="kha-white/manga-ocr-base",
        help="HuggingFace model id or local path for manga-ocr",
    )
    recognition.add_argument(
        "--pad",
        type=float,
        default=0.15,
        help="padding added around a group before OCR, in multiples of glyph size",
    )
    recognition.add_argument(
        "--no-ocr",
        action="store_true",
        help="only detect and group (useful to tune --gap without loading the OCR model)",
    )
    recognition.add_argument(
        "--keep-empty",
        dest="drop_empty",
        action="store_false",
        help="keep groups whose OCR result is empty",
    )
    recognition.add_argument("--cpu", action="store_true", help="force CPU inference")

    translation = parser.add_argument_group("translation (ollama)")
    translation.add_argument(
        "--translate",
        action="store_true",
        help="translate every text group with a local ollama model",
    )
    translation.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        help="ollama server; inside a container use http://host.containers.internal:11434",
    )
    translation.add_argument(
        "--ollama-model",
        default=os.environ.get("OLLAMA_MODEL", "qwen3-vl:8b"),
        help="model to translate with (`ollama list` shows what you have)",
    )
    translation.add_argument(
        "--target-lang", default="English", help="language to translate into"
    )
    translation.add_argument(
        "--ollama-timeout",
        type=float,
        default=180.0,
        help="seconds to wait for one ollama response",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--out-dir",
        type=Path,
        default=Path(default_out_dir) if default_out_dir else None,
        help="write one <name>.json per image into this folder "
        "(files already in it are never used as input)",
    )
    output.add_argument(
        "--save-viz",
        action="store_true",
        help="with --out-dir, also write an annotated <name>.boxes.png per image",
    )
    output.add_argument(
        "--txt",
        type=Path,
        help="write every page's text to one .txt file, in reading order "
        "(default with --out-dir: <out-dir>/pages.txt)",
    )
    output.add_argument(
        "--txt-format",
        choices=["both", "translation", "original"],
        default="both",
        help="what the .txt files contain",
    )
    output.add_argument(
        "--json", type=Path, help="write all pages to one JSON file ('-' for stdout)"
    )
    output.add_argument(
        "--viz", type=Path, help="write an annotated copy of the image (single image only)"
    )
    output.add_argument("--quiet", action="store_true", help="suppress progress logging")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    images = collect_images(args.images, args.recursive, skip_dir=args.out_dir)
    if not images:
        where = ", ".join(str(p) for p in args.images)
        raise SystemExit(f"no images found in {where}")
    if args.viz and len(images) > 1:
        raise SystemExit("--viz works with a single image at a time; use --save-viz")
    if args.translate and args.no_ocr:
        raise SystemExit("--translate needs the OCR text; drop --no-ocr")

    def log(msg: str) -> None:
        if not args.quiet:
            print(msg, file=sys.stderr)

    log(f"{len(images)} image(s) to process")

    log("loading text detector...")
    reader = build_detector(
        gpu=not args.cpu, detect_network=args.detect_network, verbose=not args.quiet
    )

    mocr = None
    if not args.no_ocr:
        try:
            from manga_ocr import MangaOcr  # noqa: PLC0415
        except ImportError as exc:
            raise SystemExit(
                "manga-ocr is required for recognition: pip install manga-ocr"
            ) from exc
        log("loading manga-ocr...")
        mocr = MangaOcr(pretrained_model_name_or_path=args.model, force_cpu=args.cpu)

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    used_stems: set[str] = set()

    def out_stem(path: Path) -> str:
        """Unique output name, so same-named pages in different folders (--recursive)
        do not overwrite each other."""
        stem = path.stem
        if stem in used_stems:
            stem = f"{path.parent.name}_{stem}" if path.parent.name else stem
            suffix = 2
            while stem in used_stems:
                stem, suffix = f"{path.stem}_{suffix}", suffix + 1
        used_stems.add(stem)
        return stem

    pages = []
    for path in images:
        log(f"processing {path}...")
        page = process_image(path, reader, mocr, args)

        if args.translate and page.groups:
            log(f"translating {len(page.groups)} group(s) with {args.ollama_model}...")
            try:
                translations = translate_texts(
                    [g.text for g in page.groups],
                    url=args.ollama_url,
                    model=args.ollama_model,
                    target_lang=args.target_lang,
                    timeout=args.ollama_timeout,
                    log=log,
                )
            except OllamaError as exc:
                raise SystemExit(f"translation failed: {exc}") from exc
            for group, translated in zip(page.groups, translations):
                group.translation = translated

        pages.append(page)

        print(f"=== {path} - {len(page.groups)} text group(s) ===")
        for i, group in enumerate(page.groups, start=1):
            box = group.bbox
            print(
                f"[{i}] bbox=({box.x0},{box.y0})-({box.x1},{box.y1}) "
                f"fragments={len(group.boxes)}"
            )
            if group.text:
                print(f"    {group.text}")
            if group.translation:
                print(f"    -> {group.translation}")
        print()

        if args.out_dir:
            stem = out_stem(path)
            page_json = args.out_dir / f"{stem}.json"
            page_json.write_text(
                json.dumps(page.to_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            log(f"wrote {page_json}")

            page_txt = args.out_dir / f"{stem}.txt"
            write_txt([page], page_txt, args.txt_format)
            log(f"wrote {page_txt}")

            if args.save_viz:
                viz_path = args.out_dir / f"{stem}.boxes.png"
                draw_visualisation(path, page, viz_path)
                log(f"wrote {viz_path}")

    if args.viz:
        draw_visualisation(images[0], pages[0], args.viz)
        log(f"wrote {args.viz}")

    # All pages in one file, in the order they were processed.
    txt_path = args.txt or (args.out_dir / "pages.txt" if args.out_dir else None)
    if txt_path:
        write_txt(pages, txt_path, args.txt_format)
        log(f"wrote {txt_path}")

    if args.json:
        payload = {"pages": [p.to_dict() for p in pages]}
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if str(args.json) == "-":
            print(text)
        else:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(text + "\n", encoding="utf-8")
            log(f"wrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
