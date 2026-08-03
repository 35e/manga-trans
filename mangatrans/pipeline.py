"""Turning one page into text groups: detect, segment, match, group, recognise."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path

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
# The default, resolved per page against the detector that found the text.
AUTO_TEXT = "auto"
TEXT_CHOICES = (AUTO_TEXT, "bubbles", "page", "all")


def keep_kinds(choice: str, grouped: bool) -> set[str]:
    """Which kinds of text to keep.

    ``kind`` answers two different questions, and only one of them is a good
    reason to throw text away. For the eraser it says what is *under* the
    lettering, which decides whether the surface can be measured or has to be
    inpainted - always worth knowing. As a filter it stands in for "is this a
    sound effect?", which was a fair guess when the alternative was feeding
    CRAFT's noise to the translator.

    It is no longer a fair guess when a model has already said "this is a block
    of text, and here is how sure I am". Dropping its answer because a ring of
    pixels around the box did not look plain enough is the same compounding of
    hand-set gates that made a page's output depend on the weather. So a
    detector that groups the page is believed by default, and ``--text`` is
    there for when you want less than it found.
    """
    if choice == AUTO_TEXT:
        return KEEP["all"] if grouped else KEEP["page"]
    return KEEP[choice]


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
    # How sure the detector was that this is a block of text. Always 1.0 from
    # the heuristic path, which has no opinion; a real number from a model, and
    # the thing to sort by when checking a page over.
    confidence: float = 1.0
    language: str = "unknown"
    # Why this group was left out of the results, if it was. Empty for a group
    # that was kept. Nothing is dropped without saying so: a missing line is far
    # harder to notice than a wrong one.
    drop_reason: str = ""
    # Set once the page has been masked: the area the white actually covers, and
    # so the room the English lettered in its place has to play with.
    mask_bbox: Box | None = None

    @property
    def glyph_size(self) -> float:
        """Character size of the original lettering.

        Fragments are single columns or lines, so their short side is one
        character; the median across a group ignores a stray furigana run.
        """
        return statistics.median([f.glyph_size for f in self.boxes] or [self.bbox.w])

    def to_dict(self) -> dict:
        out = {
            "bbox": self.bbox.as_list(),
            "text": self.text,
            "translation": self.translation,
            "kind": self.kind,
            "confidence": round(self.confidence, 3),
            "language": self.language,
            "plainness": round(self.plainness, 3),
            "region": self.region.to_dict() if self.region else None,
            "mask_bbox": self.mask_bbox.as_list() if self.mask_bbox else None,
            "fragments": len(self.boxes),
            "fragment_boxes": [b.as_list() for b in self.boxes],
        }
        if self.drop_reason:
            out["drop_reason"] = self.drop_reason
        return out


@dataclass
class Page:
    path: Path
    width: int
    height: int
    fragments_detected: int
    groups: list[TextGroup]
    regions: list[Region] = field(default_factory=list)
    dropped: list[TextGroup] = field(default_factory=list)
    # Per-pixel lettering mask, when the detector produced one. The eraser
    # prefers it to working the strokes out from the page's own tones.
    text_mask: object | None = None

    def to_dict(self) -> dict:
        return {
            "image": str(self.path),
            "width": self.width,
            "height": self.height,
            "fragments_detected": self.fragments_detected,
            "groups": [g.to_dict() for g in self.groups],
            # Everything the page found and then decided against, each with its
            # reason. Text that falls out of the pipeline used to leave no trace
            # but a log line, which made a dropped bubble look exactly like a
            # bubble that was never there.
            "dropped": [g.to_dict() for g in self.dropped],
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


def region_for(regions: list[Region], box: Box, contains: float) -> Region | None:
    """The smallest region ``box`` sits on, or ``None``.

    Smallest wins so that a bubble always beats the pale panel it is drawn on.
    """
    best: Region | None = None
    for region in regions:
        if region.area >= (best.area if best else float("inf")):
            continue
        if region.coverage(box) >= contains:
            best = region
    return best


def build_groups_from_blocks(masks, result, regions: list[Region], args) -> list[TextGroup]:
    """Turn a detector's own text blocks into groups.

    There is no clustering to do here - the model already said which lettering
    belongs to which utterance, which is the whole reason for preferring it. The
    page geometry is still consulted, but only for what it is actually good at:
    the *shape* a bubble has, which is what the eraser repaints and what the
    letterer fits English into. Whether something is text is the model's call;
    what shape it is sitting on is the page's.

    Crucially, failing to find a bubble under a block no longer costs the block.
    It loses the bubble-shaped typesetting box and nothing else.
    """
    groups: list[TextGroup] = []
    for block in result.blocks:
        members = block.fragments or [block.box]
        region = region_for(regions, block.box, args.contains)
        if region is not None and region.area > args.max_bubble_ratio * block.box.area:
            # A region far bigger than the text on it is a blank patch of panel,
            # not a bubble. Handing it to the eraser would licence repainting a
            # chunk of artwork, so drop the association - but keep the text.
            region = None

        if region is not None:
            kind, plainness, polarity = BUBBLE, 1.0, region.polarity
        else:
            glyph = statistics.median([f.glyph_size for f in members])
            polarity, plainness = backing_of(masks, block.box, int(glyph))
            kind = PLATE if plainness >= args.plain_threshold else ART

        groups.append(
            TextGroup(
                bbox=block.box,
                boxes=members,
                kind=kind,
                plainness=plainness,
                polarity=polarity,
                region=region,
                confidence=block.confidence,
                language=block.language,
            )
        )
    return groups


def process_image(path: Path, detector, mocr, args, log=lambda _msg: None) -> Page:
    import numpy as np  # noqa: PLC0415
    from PIL import Image, ImageOps  # noqa: PLC0415

    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    width, height = image.size

    grey = np.array(image.convert("L"))
    masks = page_masks(grey)
    result = detector(np.array(image), log=log)
    masks.text = result.text_mask

    regions = find_regions(
        masks,
        seal=args.seal,
        min_solidity=args.min_solidity,
        max_midtone=args.max_midtone,
        inverted=not args.no_inverted,
    )
    if result.grouped:
        groups = build_groups_from_blocks(masks, result, regions, args)
    else:
        groups = build_groups(masks, result.fragments, regions, args)

    kept, dropped = [], []
    allowed = keep_kinds(args.text, result.grouped)
    for group in groups:
        if group.kind not in allowed:
            group.drop_reason = f"kind {group.kind!r} not in --text {args.text}"
        elif len(group.boxes) < args.min_fragments:
            group.drop_reason = (
                f"{len(group.boxes)} fragment(s), below --min-fragments "
                f"{args.min_fragments}"
            )
        elif group.bbox.w < args.min_group_px and group.bbox.h < args.min_group_px:
            group.drop_reason = (
                f"{group.bbox.w}x{group.bbox.h}px, below --min-group-px "
                f"{args.min_group_px}"
            )
        elif group.confidence < args.min_confidence:
            group.drop_reason = (
                f"confidence {group.confidence:.2f}, below --min-confidence "
                f"{args.min_confidence}"
            )
        if group.drop_reason:
            dropped.append(group)
        else:
            kept.append(group)

    for group in dropped:
        log(f"  skipping text at {group.bbox.as_list()}: {group.drop_reason}")

    kept = sort_reading_order(kept, args.order)
    dropped = sort_reading_order(dropped, args.order)

    if mocr is not None:
        for group in kept:
            pad = max(2, round(args.pad * group.glyph_size))
            crop = group.bbox.padded(pad, width, height)
            group.text = mocr(
                image.crop((crop.x0, crop.y0, crop.x1, crop.y1))
            ).strip()
        if args.drop_empty:
            for group in kept:
                if not group.text:
                    group.drop_reason = "recognised as empty"
            dropped += [g for g in kept if g.drop_reason]
            kept = [g for g in kept if not g.drop_reason]

    return Page(
        path=path,
        width=width,
        height=height,
        fragments_detected=len(result.fragments),
        groups=kept,
        regions=regions,
        dropped=dropped,
        text_mask=result.text_mask,
    )
