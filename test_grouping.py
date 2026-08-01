"""Tests for the geometry/grouping logic (no models or heavy deps needed).

Run with:  python -m pytest test_grouping.py    or    python test_grouping.py
"""

from manga_ocr_groups import (
    CANVAS_MAX,
    CANVAS_MIN,
    CANVAS_STEP,
    DETECT_BASE_BYTES,
    Box,
    TextGroup,
    auto_canvas_size,
    available_memory_bytes,
    box_gap,
    canvas_size_arg,
    group_boxes,
    bubble_score,
    is_memory_error,
    load_font,
    overlaps_any,
    sort_reading_order,
    union_box,
    wrap_text,
)


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
    groups = _groups_as_sets(bubble_a + bubble_b, gap_factor=1.0)

    assert len(groups) == 2
    assert {frozenset(range(len(bubble_a))), frozenset(range(len(bubble_a), len(bubble_a) + len(bubble_b)))} == groups


def test_adjacent_vertical_lines_join_one_group():
    # Two vertical text lines of one bubble, 8px apart: same text box.
    line1 = [Box(100, y, 120, y + 20) for y in range(100, 180, 22)]
    line2 = [Box(128, y, 148, y + 20) for y in range(100, 180, 22)]
    groups = _groups_as_sets(line1 + line2, gap_factor=1.0)

    assert len(groups) == 1


def test_gap_factor_controls_merging():
    boxes = [Box(0, 0, 20, 20), Box(50, 0, 70, 20)]  # 30px apart, 20px glyphs
    assert len(_groups_as_sets(boxes, gap_factor=1.0)) == 2  # 20px threshold
    assert len(_groups_as_sets(boxes, gap_factor=2.0)) == 1  # 40px threshold


def test_threshold_is_relative_to_glyph_size():
    # Small text: 10px glyphs, 15px apart -> separate at gap_factor 1.0.
    small = [Box(0, 0, 10, 10), Box(25, 0, 35, 10)]
    # Large text: 60px glyphs, 15px apart -> together at the same factor.
    large = [Box(0, 0, 60, 60), Box(75, 0, 135, 60)]
    assert len(_groups_as_sets(small, gap_factor=1.0)) == 2
    assert len(_groups_as_sets(large, gap_factor=1.0)) == 1


def test_max_gap_px_clamps_large_glyphs():
    large = [Box(0, 0, 60, 60), Box(75, 0, 135, 60)]  # 15px apart
    assert len(_groups_as_sets(large, gap_factor=1.0, max_gap_px=10)) == 2


def test_min_gap_px_joins_tiny_glyphs():
    tiny = [Box(0, 0, 4, 4), Box(10, 0, 14, 4)]  # 6px apart, 4px glyphs
    assert len(_groups_as_sets(tiny, gap_factor=1.0)) == 2
    assert len(_groups_as_sets(tiny, gap_factor=1.0, min_gap_px=8)) == 1


def test_single_linkage_chains_through_middle_box():
    boxes = [Box(0, 0, 20, 20), Box(30, 0, 50, 20), Box(60, 0, 80, 20)]
    assert len(_groups_as_sets(boxes, gap_factor=1.0)) == 1


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


def _page_with_a_bubble_and_a_panel():
    """Dark artwork with a small white bubble and a large pale panel."""
    import numpy as np

    from manga_ocr_groups import white_components

    page = np.zeros((600, 600), dtype=np.uint8)
    page[50:170, 50:170] = 255  # bubble blob, 14400 px
    page[300:560, 300:560] = 255  # panel background, 67600 px - under the
    return white_components(page), 600 * 600  # page-fraction backstop of 20%


def test_bubble_score_keeps_text_in_a_bubble():
    components, page_area = _page_with_a_bubble_and_a_panel()
    # 60x60 of text in a 120x120 bubble: 4x bigger, well inside the ratio.
    assert bubble_score(components, Box(80, 80, 140, 140), page_area) > 0.9


def test_bubble_score_rejects_sfx_on_a_pale_panel():
    components, page_area = _page_with_a_bubble_and_a_panel()
    # Same whiteness as a bubble, but the blob dwarfs the text (75x), so it is
    # panel background rather than a bubble drawn around the lettering.
    assert bubble_score(components, Box(400, 400, 430, 430), page_area) == 0.0


def test_bubble_score_rejects_sfx_on_dark_artwork():
    components, page_area = _page_with_a_bubble_and_a_panel()
    assert bubble_score(components, Box(200, 200, 260, 260), page_area) == 0.0


def test_bubble_score_without_opencv_keeps_everything():
    assert bubble_score(None, Box(0, 0, 10, 10), 1000) == 1.0


def test_overlaps_any():
    box = Box(100, 100, 200, 200)
    assert not overlaps_any(box, [])
    assert not overlaps_any(box, [Box(300, 100, 400, 200)])  # clear to the right
    assert not overlaps_any(box, [Box(200, 100, 300, 200)])  # edges touching only
    assert overlaps_any(box, [Box(150, 150, 250, 250)])  # corner overlap
    assert overlaps_any(box, [Box(0, 0, 500, 500)])  # fully enclosing


def test_wrap_text_fits_the_width():
    font = load_font(None, 20)
    text = "Failed my college entrance exams and been a NEET for three years"
    wide = wrap_text(text, font, 10_000)
    narrow = wrap_text(text, font, 150)

    assert wide == [text]  # all on one line when there is room
    assert len(narrow) > 1
    assert " ".join(narrow) == text  # wrapping never loses or reorders words
    # Every line fits, except one that is a single unbreakable word.
    for line in narrow:
        assert font.getlength(line) <= 150 or " " not in line


def test_wrap_text_empty():
    assert wrap_text("   ", load_font(None, 20), 100) == []


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
