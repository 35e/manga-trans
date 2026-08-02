"""Turning one page into text groups: detect, segment, match, group, recognise."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path

from .detect import auto_canvas_size, available_memory_bytes, detect_fragments
from .geometry import Box, group_boxes, sort_reading_order, union_box
from .regions import Region, assign_regions, backing_of, find_regions, page_masks

# What a group is sitting on, in descending order of "this is certainly text".
BUBBLE = "bubble"  # inside a speech bubble, caption box or sign
PLATE = "plate"  # free-standing, but on plain paper: narration, titles
ART = "art"  # free-standing over artwork: sound effects, scribbles

KEEP = {
    "bubbles": {BUBBLE},
    "page": {BUBBLE, PLATE},
    "all": {BUBBLE, PLATE, ART},
}


@dataclass
class TextGroup:
    """One utterance: the fragments of a single bubble or block of text."""

    bbox: Box
    boxes: list[Box] = field(default_factory=list)
    kind: str = ART
    plainness: float = 0.0
    polarity: str = "light"
    region: Region | None = None
    text: str = ""
    translation: str = ""

    @property
    def glyph_size(self) -> float:
        """Character size of the original lettering.

        Fragments are single columns or lines, so their short side is one
        character; the median across a group ignores a stray furigana run.
        """
        return statistics.median([f.glyph_size for f in self.boxes] or [self.bbox.w])

    def to_dict(self) -> dict:
        return {
            "bbox": self.bbox.as_list(),
            "text": self.text,
            "translation": self.translation,
            "kind": self.kind,
            "plainness": round(self.plainness, 3),
            "region": self.region.to_dict() if self.region else None,
            "fragments": len(self.boxes),
            "fragment_boxes": [b.as_list() for b in self.boxes],
        }


@dataclass
class Page:
    path: Path
    width: int
    height: int
    fragments_detected: int
    groups: list[TextGroup]
    regions: list[Region] = field(default_factory=list)
    dropped: list[TextGroup] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "image": str(self.path),
            "width": self.width,
            "height": self.height,
            "fragments_detected": self.fragments_detected,
            "groups": [g.to_dict() for g in self.groups],
        }


def _cluster(fragments: list[Box], indices: list[int], args) -> list[list[Box]]:
    """Split a set of fragments into blocks of text by proximity and alignment."""
    members = [fragments[i] for i in indices]
    return [
        [members[i] for i in cluster]
        for cluster in group_boxes(
            members,
            gap_factor=args.gap,
            stack_factor=args.stack_gap,
            min_gap_px=args.min_gap_px,
            max_gap_px=args.max_gap_px,
        )
    ]


def build_groups(masks, fragments: list[Box], regions: list[Region], args) -> list[TextGroup]:
    """Assign every fragment to the shape it sits on and gather them into groups.

    Grouping happens *inside* a region rather than across the whole page, so two
    bubbles that nearly touch can never be merged into one utterance however
    generous ``--gap`` is. Only text that belongs to no region falls back to
    grouping by distance alone.
    """
    held, free = assign_regions(regions, fragments, contains=args.contains)

    groups: list[TextGroup] = []
    for region, indices in held:
        # A bubble is drawn *around* its lettering, so it stays within a few
        # times the size of it. A region far bigger than the text on it is a
        # blank patch of a panel that happens to be flat and bounded - a scrap
        # of sky, a sleeve - and the handwriting on it is not dialogue. Letting
        # those through would also hand the eraser licence to repaint a chunk of
        # artwork in flat white.
        if region.area > args.max_bubble_ratio * union_box(
            [fragments[i] for i in indices]
        ).area:
            free.extend(indices)
            continue
        # One region can still hold several blocks: bubbles drawn overlapping
        # share a single blob of paper, and a caption box may carry two
        # paragraphs. The same alignment rule separates them.
        for members in _cluster(fragments, indices, args):
            groups.append(
                TextGroup(
                    bbox=union_box(members),
                    boxes=members,
                    kind=BUBBLE,
                    plainness=1.0,
                    polarity=region.polarity,
                    region=region,
                )
            )

    for members in _cluster(fragments, sorted(free), args):
        bbox = union_box(members)
        glyph = statistics.median([f.glyph_size for f in members])
        polarity, plainness = backing_of(masks, bbox, int(glyph))
        groups.append(
            TextGroup(
                bbox=bbox,
                boxes=members,
                kind=PLATE if plainness >= args.plain_threshold else ART,
                plainness=plainness,
                polarity=polarity,
            )
        )
    return groups


def process_image(path: Path, reader, mocr, args, log=lambda _msg: None) -> Page:
    import numpy as np  # noqa: PLC0415
    from PIL import Image, ImageOps  # noqa: PLC0415

    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    width, height = image.size

    canvas_size = args.canvas_size
    if canvas_size is None:  # --canvas-size auto
        canvas_size = auto_canvas_size(width, height, available_memory_bytes())
        if canvas_size < max(width, height):
            log(f"  {width}x{height} page, detecting at canvas {canvas_size}")

    grey = np.array(image.convert("L"))
    masks = page_masks(grey)
    fragments = detect_fragments(reader, grey, args, canvas_size, log=log)

    regions = find_regions(
        masks,
        seal=args.seal,
        min_solidity=args.min_solidity,
        max_midtone=args.max_midtone,
        inverted=not args.no_inverted,
    )
    groups = build_groups(masks, fragments, regions, args)

    kept, dropped = [], []
    allowed = KEEP[args.text]
    for group in groups:
        too_few = len(group.boxes) < args.min_fragments
        too_small = (
            group.bbox.w < args.min_group_px and group.bbox.h < args.min_group_px
        )
        if group.kind in allowed and not too_few and not too_small:
            kept.append(group)
        else:
            dropped.append(group)

    for group in dropped:
        log(
            f"  skipping {group.kind} text at {group.bbox.as_list()} "
            f"(plain {group.plainness:.2f}, {len(group.boxes)} fragment(s))"
        )

    kept = sort_reading_order(kept, args.order)

    if mocr is not None:
        for group in kept:
            pad = max(2, round(args.pad * group.glyph_size))
            crop = group.bbox.padded(pad, width, height)
            group.text = mocr(
                image.crop((crop.x0, crop.y0, crop.x1, crop.y1))
            ).strip()
        if args.drop_empty:
            kept = [g for g in kept if g.text]

    return Page(
        path=path,
        width=width,
        height=height,
        fragments_detected=len(fragments),
        groups=kept,
        regions=regions,
        dropped=dropped,
    )
