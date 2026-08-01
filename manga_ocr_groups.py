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
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

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

    def to_dict(self) -> dict:
        return {
            "bbox": self.bbox.as_list(),
            "text": self.text,
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
# cli
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OCR a manga page, grouping text by text box / speech bubble.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("images", nargs="+", type=Path, help="image file(s) to process")

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

    output = parser.add_argument_group("output")
    output.add_argument("--json", type=Path, help="write results as JSON ('-' for stdout)")
    output.add_argument(
        "--viz", type=Path, help="write an annotated copy of the image (single image only)"
    )
    output.add_argument("--quiet", action="store_true", help="suppress progress logging")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    for path in args.images:
        if not path.is_file():
            raise SystemExit(f"no such file: {path}")
    if args.viz and len(args.images) > 1:
        raise SystemExit("--viz works with a single image at a time")

    def log(msg: str) -> None:
        if not args.quiet:
            print(msg, file=sys.stderr)

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

    pages = []
    for path in args.images:
        log(f"processing {path}...")
        page = process_image(path, reader, mocr, args)
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
        print()

    if args.viz:
        draw_visualisation(args.images[0], pages[0], args.viz)
        log(f"wrote {args.viz}")

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
