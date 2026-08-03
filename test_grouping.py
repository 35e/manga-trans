"""Tests for the geometry, segmentation and layout logic.

Needs numpy, OpenCV and Pillow (all pulled in by easyocr), but no models and no
network.

Run with:  python -m pytest test_grouping.py    or    python test_grouping.py
"""

import cv2
import numpy as np

from mangatrans.detect import (
    CANVAS_MAX,
    CANVAS_MIN,
    CANVAS_STEP,
    DETECT_BASE_BYTES,
    auto_canvas_size,
    available_memory_bytes,
    is_memory_error,
)
from mangatrans.cli import canvas_size_arg, natural_key
from mangatrans.erase import TIGHT, WHOLE, _stroke_mask, erase_text, text_mask
from mangatrans.geometry import (
    Box,
    box_gap,
    group_boxes,
    joins,
    overlaps_any,
    sort_reading_order,
    union_box,
)
from mangatrans.letter import fit_text, inscribed_rect, load_font, wrap_text
from mangatrans.pipeline import ART, BUBBLE, PLATE, TextGroup, build_groups
from mangatrans.regions import (
    assign_regions,
    backing_of,
    find_regions,
    page_masks,
    thresholds,
)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def _groups_as_sets(boxes, **kwargs):
    return {frozenset(indices) for indices in group_boxes(boxes, **kwargs)}


def test_box_gap():
    a = Box(0, 0, 10, 10)
    assert box_gap(a, Box(0, 0, 10, 10)) == 0  # identical
    assert box_gap(a, Box(5, 5, 15, 15)) == 0  # overlapping
    assert box_gap(a, Box(20, 0, 30, 10)) == 10  # horizontal gap
    assert box_gap(a, Box(0, 20, 10, 30)) == 10  # vertical gap
    assert box_gap(a, Box(13, 14, 20, 20)) == 5  # diagonal gap (3-4-5)


def test_two_bubbles_are_separate_groups():
    # Two columns of "characters" (20px glyphs) far apart on the page.
    bubble_a = [Box(100, y, 120, y + 20) for y in range(100, 200, 22)]
    bubble_b = [Box(600, y, 620, y + 20) for y in range(100, 200, 22)]
    groups = _groups_as_sets(bubble_a + bubble_b)

    assert len(groups) == 2


def test_adjacent_vertical_lines_join_one_group():
    # Two vertical text lines of one bubble, 8px apart: same text box.
    line1 = [Box(100, y, 120, y + 20) for y in range(100, 180, 22)]
    line2 = [Box(128, y, 148, y + 20) for y in range(100, 180, 22)]
    assert len(_groups_as_sets(line1 + line2)) == 1


def test_a_column_continues_across_a_wide_break():
    # "haru" and a trailing "...?" set well below it in the same column: one
    # utterance, even though the gap is more than twice the glyph size.
    top = Box(832, 484, 864, 514)
    trailing = Box(842, 542, 854, 564)
    assert box_gap(top, trailing) > 1.2 * min(top.glyph_size, trailing.glyph_size)
    assert len(_groups_as_sets([top, trailing])) == 1
    # ... but not when the column may not continue at all.
    assert len(_groups_as_sets([top, trailing], stack_factor=0.5)) == 2


def test_neighbouring_columns_keep_a_tight_limit():
    # Same 2.3x glyph gap as above, but side by side rather than stacked: this
    # is the spacing between two bubbles drawn overlapping, not one bubble.
    left = Box(100, 100, 130, 300)
    right = Box(200, 100, 230, 300)
    assert len(_groups_as_sets([left, right])) == 2
    assert len(_groups_as_sets([left, right], gap_factor=4.0)) == 1


def test_blocks_offset_diagonally_stay_apart():
    # Two utterances in one blob of paper: neighbouring columns, but neither
    # continues the other, so they line up on neither axis.
    a = [Box(946, 1102, 972, 1266), Box(984, 1100, 1024, 1244)]
    b = [Box(896, 1228, 924, 1368), Box(856, 1228, 894, 1302)]
    groups = _groups_as_sets(a + b)
    assert len(groups) == 2


def test_touching_fragments_always_join():
    # Diagonal, so they line up on neither axis - but a 1px gap settles it.
    assert joins(Box(0, 0, 40, 40), Box(41, 30, 81, 90), gap=1.2, stack=3.0)
    # The clamps still have the last word.
    assert not joins(Box(0, 0, 40, 40), Box(41, 30, 81, 90), gap=0.0, stack=0.0)


def test_threshold_is_relative_to_glyph_size():
    # Small text: 10px glyphs, 15px apart -> separate columns.
    small = [Box(0, 0, 10, 10), Box(25, 0, 35, 10)]
    # Large text: 60px glyphs, 15px apart -> one block at the same factor.
    large = [Box(0, 0, 60, 60), Box(75, 0, 135, 60)]
    assert len(_groups_as_sets(small, gap_factor=1.0)) == 2
    assert len(_groups_as_sets(large, gap_factor=1.0)) == 1


def test_max_gap_px_clamps_large_glyphs():
    large = [Box(0, 0, 60, 60), Box(75, 0, 135, 60)]  # 15px apart
    assert len(_groups_as_sets(large, max_gap_px=10)) == 2


def test_min_gap_px_joins_tiny_glyphs():
    tiny = [Box(0, 0, 4, 4), Box(12, 0, 16, 4)]  # 8px apart, 4px glyphs
    assert len(_groups_as_sets(tiny)) == 2
    assert len(_groups_as_sets(tiny, min_gap_px=10)) == 1


def test_single_linkage_chains_through_middle_box():
    boxes = [Box(0, 0, 20, 20), Box(30, 0, 50, 20), Box(60, 0, 80, 20)]
    assert len(_groups_as_sets(boxes)) == 1


def test_union_box():
    assert union_box([Box(10, 20, 30, 40), Box(5, 25, 35, 38)]).as_list() == [5, 20, 35, 40]


def test_empty_input():
    assert group_boxes([]) == []


def _group(x0, y0, x1, y1):
    return TextGroup(bbox=Box(x0, y0, x1, y1))


def test_reading_order_right_to_left_then_down():
    top_left = _group(50, 50, 150, 150)
    top_right = _group(400, 50, 500, 150)
    bottom = _group(200, 400, 300, 500)

    ordered = sort_reading_order([top_left, bottom, top_right], order="rtl")
    assert [g.bbox.x0 for g in ordered] == [400, 50, 200]

    ordered_ltr = sort_reading_order([top_right, bottom, top_left], order="ltr")
    assert [g.bbox.x0 for g in ordered_ltr] == [50, 400, 200]

    unordered = sort_reading_order([top_right, bottom, top_left], order="none")
    assert [g.bbox.x0 for g in unordered] == [400, 200, 50]


def test_box_clipped_to_image():
    # The detector scales boxes back up from the canvas and can overshoot.
    assert Box(-5, -2, 50, 60).clipped(100, 100).as_list() == [0, 0, 50, 60]
    assert Box(80, 80, 130, 140).clipped(100, 100).as_list() == [80, 80, 100, 100]
    assert Box(10, 10, 20, 20).clipped(100, 100).as_list() == [10, 10, 20, 20]


def test_overlaps_any():
    box = Box(100, 100, 200, 200)
    assert not overlaps_any(box, [])
    assert not overlaps_any(box, [Box(300, 100, 400, 200)])  # clear to the right
    assert not overlaps_any(box, [Box(200, 100, 300, 200)])  # edges touching only
    assert overlaps_any(box, [Box(150, 150, 250, 250)])  # corner overlap
    assert overlaps_any(box, [Box(0, 0, 500, 500)])  # fully enclosing


# ---------------------------------------------------------------------------
# a synthetic page
# ---------------------------------------------------------------------------

BUBBLE_BOX = Box(320, 300, 480, 420)
BUBBLE_TEXT = Box(360, 330, 440, 390)
SFX_TEXT = Box(120, 560, 200, 620)
NOTE_TEXT = Box(300, 730, 420, 762)  # on the blank margin, outside the panel


def _synthetic_page():
    """A page with one bubble, one sound effect on artwork, one plain note.

    * a white ellipse with black bars in it, drawn over artwork: a bubble;
    * black bars straight on the hatched artwork: a sound effect;
    * black bars on the blank paper margin: free-standing text.
    """
    page = np.full((800, 800), 255, np.uint8)
    page[40:700, 40:760] = 160  # a panel of shaded artwork
    for y in range(40, 700, 6):  # hatched, so it is neither paper nor flat ink
        page[y : y + 2, 40:760] = 0

    cv2.ellipse(page, (400, 360), (80, 60), 0, 0, 360, 255, -1)
    cv2.ellipse(page, (400, 360), (80, 60), 0, 0, 360, 0, 3)
    for box in (BUBBLE_TEXT, SFX_TEXT, NOTE_TEXT):
        for x in range(box.x0, box.x1, 18):
            page[box.y0 : box.y1, x : x + 10] = 0
    return page


def _text_boxes():
    return [
        Box(BUBBLE_TEXT.x0, BUBBLE_TEXT.y0, BUBBLE_TEXT.x1, BUBBLE_TEXT.y1),
        SFX_TEXT,
        NOTE_TEXT,
    ]


class _Args:
    """The handful of settings the grouping stage reads."""

    gap = 1.2
    stack_gap = 3.0
    min_gap_px = 0.0
    max_gap_px = None
    contains = 0.6
    max_bubble_ratio = 10.0
    plain_threshold = 0.85


def test_thresholds_follow_the_pages_white_point():
    bright = np.full((10, 10), 250, np.uint8)
    dim = np.full((10, 10), 180, np.uint8)
    assert thresholds(bright)[0] > thresholds(dim)[0]
    assert thresholds(bright)[1] > thresholds(dim)[1]


def test_find_regions_finds_the_bubble_and_not_the_page():
    masks = page_masks(_synthetic_page())
    regions = find_regions(masks)
    boxes = [r.box for r in regions]
    # The bubble is there ...
    assert any(
        abs(b.cx - 400) < 8 and abs(b.cy - 360) < 8 and b.w < 200 for b in boxes
    ), boxes
    # ... and the paper margin, which touches the page edge, is not.
    assert not any(b.w > 700 and b.h > 700 for b in boxes)


def test_hatched_artwork_holds_no_text():
    masks = page_masks(_synthetic_page())
    regions = find_regions(masks)
    held, free = assign_regions(regions, _text_boxes())
    assert 1 in free  # the sound effect belongs to no shape
    assert all(not r.box.overlaps(SFX_TEXT) for r, _ in held)


def test_assign_regions_prefers_the_smallest_region():
    masks = page_masks(_synthetic_page())
    regions = find_regions(masks)
    held, free = assign_regions(regions, _text_boxes())
    assert len(held) == 1
    region, indices = held[0]
    assert indices == [0]  # only the bubble's text sits on a region
    assert region.box.w < 200
    assert set(free) == {1, 2}


def test_build_groups_sorts_text_into_bubble_plate_and_art():
    masks = page_masks(_synthetic_page())
    regions = find_regions(masks)
    groups = build_groups(masks, _text_boxes(), regions, _Args())
    kinds = {g.bbox.x0: g.kind for g in groups}
    assert kinds[BUBBLE_TEXT.x0] == BUBBLE
    assert kinds[SFX_TEXT.x0] == ART  # hatching behind it
    assert kinds[NOTE_TEXT.x0] == PLATE  # blank paper behind it


def test_a_bubble_far_bigger_than_its_text_is_not_a_bubble():
    masks = page_masks(_synthetic_page())
    regions = find_regions(masks)

    class Strict(_Args):
        max_bubble_ratio = 0.5

    groups = build_groups(masks, _text_boxes(), regions, Strict())
    assert all(g.kind != BUBBLE for g in groups)


def test_backing_of_tells_paper_from_artwork():
    masks = page_masks(_synthetic_page())
    _, plain_note = backing_of(masks, NOTE_TEXT, 20)
    _, plain_sfx = backing_of(masks, SFX_TEXT, 20)
    assert plain_note > 0.9
    assert plain_sfx < plain_note


# ---------------------------------------------------------------------------
# erasing
# ---------------------------------------------------------------------------


def test_stroke_mask_covers_the_glyphs_not_the_box():
    masks = page_masks(_synthetic_page())
    group = TextGroup(bbox=BUBBLE_TEXT, boxes=[BUBBLE_TEXT], kind=BUBBLE)
    mask = _stroke_mask(masks, group, bleed=2, shape=masks.shape)
    covered = np.count_nonzero(mask[BUBBLE_TEXT.y0 : BUBBLE_TEXT.y1, BUBBLE_TEXT.x0 : BUBBLE_TEXT.x1])
    assert 0.2 < covered / BUBBLE_TEXT.area < 0.95  # strokes, not the whole box


def test_erasing_a_bubble_leaves_it_blank():
    from PIL import Image

    page = _synthetic_page()
    masks = page_masks(page)
    regions = find_regions(masks)
    held, _ = assign_regions(regions, _text_boxes())
    region = held[0][0]
    group = TextGroup(
        bbox=BUBBLE_TEXT, boxes=[BUBBLE_TEXT], kind=BUBBLE, region=region
    )

    image = Image.fromarray(cv2.cvtColor(page, cv2.COLOR_GRAY2RGB))
    cleaned = np.asarray(erase_text(image, [group], masks, 0.12))[:, :, 0]

    inside = cleaned[BUBBLE_TEXT.y0 : BUBBLE_TEXT.y1, BUBBLE_TEXT.x0 : BUBBLE_TEXT.x1]
    assert inside.min() > 200, "the lettering should be gone"
    # The bubble's own outline survives.
    ring = cleaned[BUBBLE_BOX.y0 - 2 : BUBBLE_BOX.y0 + 6, 395:405]
    assert ring.min() < 100
    # And nothing outside the bubble was touched.
    untouched = cleaned[SFX_TEXT.y0 : SFX_TEXT.y1, SFX_TEXT.x0 : SFX_TEXT.x1]
    assert untouched.min() < 50


# ---------------------------------------------------------------------------
# masking: a flat patch to letter English onto
# ---------------------------------------------------------------------------


def _bubble_group():
    masks = page_masks(_synthetic_page())
    regions = find_regions(masks)
    held, _ = assign_regions(regions, _text_boxes())
    group = TextGroup(
        bbox=BUBBLE_TEXT, boxes=[BUBBLE_TEXT], kind=BUBBLE, region=held[0][0]
    )
    return masks, group


def test_tight_mask_covers_every_stroke_of_the_text():
    """The point of the tight mode: no Japanese survives inside its own box."""
    masks, group = _bubble_group()
    mask = text_mask(masks, [group], masks.shape, mode=TIGHT)

    window = (
        slice(BUBBLE_TEXT.y0, BUBBLE_TEXT.y1),
        slice(BUBBLE_TEXT.x0, BUBBLE_TEXT.x1),
    )
    ink = ~masks.pale[window]
    assert ink.any(), "the fixture should have lettering to cover"
    assert not (ink & ~mask[window].astype(bool)).any(), "ink left showing"


def test_tight_mask_hugs_the_text_instead_of_filling_the_bubble():
    masks, group = _bubble_group()
    tight = text_mask(masks, [group], masks.shape, mode=TIGHT)
    whole = text_mask(masks, [group], masks.shape, mode=WHOLE)

    assert np.count_nonzero(tight) < 0.75 * np.count_nonzero(whole)
    # Both stay inside the bubble: neither may cross the drawn outline.
    outside = np.ones(masks.shape, bool)
    outside[BUBBLE_BOX.y0 : BUBBLE_BOX.y1, BUBBLE_BOX.x0 : BUBBLE_BOX.x1] = False
    assert not tight[outside].any()
    assert not whole[outside].any()


def test_knitting_closes_the_gaps_between_glyphs_without_leaving_the_text():
    """Closing is what makes one patch of a column of type rather than stripes."""
    masks, group = _bubble_group()
    # With the bleed turned down the gaps between the fixture's bars survive,
    # which is the thing knitting exists to close.
    def mask_for(knit):
        return text_mask(
            masks, [group], masks.shape, bleed_glyphs=0.0, knit_glyphs=knit, mode=TIGHT
        )

    window = (
        slice(BUBBLE_TEXT.y0, BUBBLE_TEXT.y1),
        slice(BUBBLE_TEXT.x0, BUBBLE_TEXT.x1),
    )
    traced = np.count_nonzero(mask_for(0.0)[window]) / BUBBLE_TEXT.area
    knitted = np.count_nonzero(mask_for(0.25)[window]) / BUBBLE_TEXT.area

    assert traced < 0.9, "tracing each bar should leave the gaps between them"
    assert knitted > 0.99, "knitting should close them into one patch"


def test_masking_with_a_colour_lays_that_colour_down_flat():
    from PIL import Image

    page = _synthetic_page()
    masks, group = _bubble_group()
    image = Image.fromarray(cv2.cvtColor(page, cv2.COLOR_GRAY2RGB))
    painted = np.asarray(
        erase_text(image, [group], masks, 0.12, mode=TIGHT, fill=(255, 0, 0))
    )

    window = (
        slice(BUBBLE_TEXT.y0, BUBBLE_TEXT.y1),
        slice(BUBBLE_TEXT.x0, BUBBLE_TEXT.x1),
    )
    ink = ~masks.pale[window]
    assert (painted[window][ink] == (255, 0, 0)).all(), "the text should be covered"
    # The artwork outside the bubble is not this operation's business.
    sfx = painted[SFX_TEXT.y0 : SFX_TEXT.y1, SFX_TEXT.x0 : SFX_TEXT.x1]
    assert not (sfx == (255, 0, 0)).all(-1).any()


def test_masking_records_where_there_is_now_room():
    masks, group = _bubble_group()
    text_mask(masks, [group], masks.shape, mode=TIGHT)

    assert group.mask_bbox is not None
    # It covers the text it was built from, and stays inside the bubble.
    assert group.mask_bbox.x0 <= BUBBLE_TEXT.x0 and group.mask_bbox.x1 >= BUBBLE_TEXT.x1
    assert group.mask_bbox.y0 <= BUBBLE_TEXT.y0 and group.mask_bbox.y1 >= BUBBLE_TEXT.y1
    assert group.mask_bbox.x0 >= BUBBLE_BOX.x0 and group.mask_bbox.x1 <= BUBBLE_BOX.x1


def test_masking_free_standing_text_never_leaves_its_own_fragments():
    """No bubble means no backing to appeal to, so the paint stays in the boxes."""
    masks = page_masks(_synthetic_page())
    group = TextGroup(bbox=SFX_TEXT, boxes=[SFX_TEXT], kind=ART, plainness=0.2)
    mask = text_mask(masks, [group], masks.shape, mode=TIGHT, knit_glyphs=0.5)

    allowed = np.zeros(masks.shape, bool)
    bleed = max(2, round(0.12 * group.glyph_size))
    grown = SFX_TEXT.padded(bleed, masks.shape[1], masks.shape[0])
    allowed[grown.y0 : grown.y1, grown.x0 : grown.x1] = True
    assert not mask[~allowed].any()


# ---------------------------------------------------------------------------
# lettering
# ---------------------------------------------------------------------------


def test_wrap_text_fits_the_width():
    font = load_font(None, 20)
    text = "Failed my college entrance exams and been a NEET for three years"
    wide = wrap_text(text, font, 10_000)
    narrow = wrap_text(text, font, 150)

    assert wide == [text]  # all on one line when there is room
    assert len(narrow) > 1
    assert " ".join(narrow) == text  # wrapping never loses or reorders words
    for line in narrow:
        assert font.getlength(line) <= 150 or " " not in line


def test_wrap_text_does_not_hyphenate_unless_asked():
    font = load_font(None, 20)
    plain = wrap_text("extraordinarily", font, 40)
    broken = wrap_text("extraordinarily", font, 40, hyphenate=True)
    assert plain == ["extraordinarily"]  # left to overflow, so the size can drop
    assert len(broken) > 1
    assert "".join(line.rstrip("-") for line in broken) == "extraordinarily"


def test_wrap_text_empty():
    assert wrap_text("   ", load_font(None, 20), 100) == []


def test_inscribed_rect_stays_inside_the_shape():
    mask = np.zeros((200, 300), np.uint8)
    cv2.ellipse(mask, (150, 100), (120, 80), 0, 0, 360, 1, -1)

    tall = inscribed_rect(mask, (150, 100), aspect=0.5)
    wide = inscribed_rect(mask, (150, 100), aspect=3.0)
    for rect in (tall, wide):
        assert rect is not None
        assert mask[rect.y0 : rect.y1, rect.x0 : rect.x1].all()
    assert wide.w > tall.w and wide.h < tall.h
    # One pixel bigger in each direction would stick out of the ellipse.
    grown = mask[wide.y0 - 3 : wide.y1 + 3, wide.x0 - 3 : wide.x1 + 3]
    assert not grown.all()


def test_inscribed_rect_outside_the_shape_is_none():
    mask = np.zeros((50, 50), np.uint8)
    mask[10:20, 10:20] = 1
    assert inscribed_rect(mask, (40, 40), aspect=1.0) is None


def test_fit_text_returns_none_when_there_is_no_room():
    from PIL import Image, ImageDraw

    class Args:
        font = None
        line_spacing = 0.16

    draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    assert fit_text(draw, "a very long sentence indeed", 6, 6, Args(), 40) is None
    assert fit_text(draw, "hi", 400, 400, Args(), 40) is not None


# ---------------------------------------------------------------------------
# detection budget and cli helpers
# ---------------------------------------------------------------------------


def test_auto_canvas_size_shrinks_with_the_budget():
    gb = 1024**3
    roomy = auto_canvas_size(2894, 4093, 16 * gb)
    tight = auto_canvas_size(2894, 4093, 4 * gb)
    assert roomy == CANVAS_MAX  # plenty of memory: no reason to downscale
    assert CANVAS_MIN <= tight < roomy
    # 4 GB is the machine the OOM kills were seen on; 2048 was fatal there.
    assert tight < 2048


def test_auto_canvas_size_never_upscales_a_small_page():
    assert auto_canvas_size(800, 1200, 64 * 1024**3) == 1200


def test_auto_canvas_size_makes_room_for_a_magnification():
    """--mag-ratio is clamped by the canvas, so the canvas has to allow for it.

    Without this the flag silently does nothing on any page smaller than the
    canvas, which is every page the auto budget has already sized to itself.
    """
    plain = auto_canvas_size(800, 1200, 64 * 1024**3, mag_ratio=1.0)
    magnified = auto_canvas_size(800, 1200, 64 * 1024**3, mag_ratio=2.0)
    assert plain == 1200
    assert magnified == 2400
    # ... but never past the ceiling, and never past what the memory allows.
    assert auto_canvas_size(2894, 4093, 64 * 1024**3, mag_ratio=4.0) == CANVAS_MAX
    assert auto_canvas_size(800, 1200, DETECT_BASE_BYTES, mag_ratio=4.0) == CANVAS_MIN


def test_auto_canvas_size_stays_within_bounds():
    assert auto_canvas_size(2894, 4093, DETECT_BASE_BYTES) == CANVAS_MIN  # no room
    assert auto_canvas_size(2894, 4093, None) == CANVAS_MAX  # unknown budget
    assert auto_canvas_size(2894, 4093, 4 * 1024**3) % CANVAS_STEP == 0


def test_available_memory_is_plausible_or_unknown():
    budget = available_memory_bytes()
    assert budget is None or budget > 64 * 1024**2


def test_is_memory_error():
    assert is_memory_error(MemoryError())
    # What torch raises when a CPU allocation fails.
    assert is_memory_error(RuntimeError("DefaultCPUAllocator: can't allocate memory"))
    assert is_memory_error(RuntimeError("std::bad_alloc"))
    assert not is_memory_error(RuntimeError("shape mismatch"))


def test_canvas_size_arg():
    assert canvas_size_arg("auto") is None
    assert canvas_size_arg("AUTO") is None
    assert canvas_size_arg("1280") == 1280
    for bad in ("banana", str(CANVAS_MIN - 1)):
        try:
            canvas_size_arg(bad)
        except Exception:
            pass
        else:
            raise AssertionError(f"{bad!r} should have been rejected")


def test_natural_key_orders_page2_before_page10():
    from pathlib import Path

    paths = [Path("p10.webp"), Path("p2.webp"), Path("p1.webp")]
    assert [p.name for p in sorted(paths, key=natural_key)] == [
        "p1.webp",
        "p2.webp",
        "p10.webp",
    ]


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    print("all passed" if not failures else f"{failures} failure(s)")
    raise SystemExit(1 if failures else 0)
