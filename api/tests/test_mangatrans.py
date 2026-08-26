"""Unit tests. The detector is stubbed, so they need no model and no network."""

from __future__ import annotations

import io
import json
import os
import sys
import threading
import types
import unittest
from unittest import mock

import cv2
import numpy as np
from PIL import Image, ImageDraw

from mangatrans import (
    bubble,
    detect,
    inpaint,
    languages,
    ollama,
    read,
    render,
    server,
)
from mangatrans.detect import FREE, SPEECH, Block
from mangatrans.geometry import Box

DARK = (20, 20, 20)
TONE = (200, 200, 200)
INK = Box(80, 50, 120, 90)
NEAR = 5


def page(width: int = 200, height: int = 140, colour=DARK) -> Image.Image:
    return Image.new("RGB", (width, height), colour)


def toned(ink: Box | None = INK, rim: int = 0) -> Image.Image:
    """Flat tone with ink on it, and a rim of half-ink around it if asked."""
    image = page(colour=TONE)
    draw = ImageDraw.Draw(image)
    if ink is not None:
        if rim:
            draw.rectangle(
                (ink.x0 - rim, ink.y0 - rim, ink.x1 - 1 + rim, ink.y1 - 1 + rim),
                outline=(100, 100, 100),
                width=rim,
            )
        draw.rectangle((ink.x0, ink.y0, ink.x1 - 1, ink.y1 - 1), fill=DARK)
    return image


def payload(image: Image.Image, mask: Image.Image | None = None, **fields) -> dict:
    """A multipart body: the image, a mask if there is one, and JSON beside them."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    body = {"image": (buffer, "page.png")}
    if mask is not None:
        stencil = io.BytesIO()
        mask.save(stencil, format="PNG")
        stencil.seek(0)
        body["mask"] = (stencil, "mask.png")
    body.update(
        {
            name: value if isinstance(value, str) else json.dumps(value)
            for name, value in fields.items()
        }
    )
    return body


def stencil(box: Box | None = None, fill: int = 255, size=(200, 140)) -> Image.Image:
    """A greyscale mask: black everywhere, ``fill`` inside ``box``."""
    mask = Image.new("L", size, 0)
    if box is not None:
        ImageDraw.Draw(mask).rectangle(
            (box.x0, box.y0, box.x1 - 1, box.y1 - 1), fill=fill
        )
    return mask


def opened(response) -> Image.Image:
    return Image.open(io.BytesIO(response.data)).convert("RGB")


def patch(pixels, box: Box) -> np.ndarray:
    return np.array(pixels)[box.y0 : box.y1, box.x0 : box.x1]


class StubRegions:
    """One block, in one balloon, whatever the page."""

    found = Block(Box(10, 10, 60, 40), 0.912, SPEECH)
    balloon = Box(2, 2, 97, 67)

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, image, rtl: bool = True):
        assert image.ndim == 3 and image.shape[2] == 3, "expects an RGB array"
        return [self.found], [self.balloon]


class StubLetters:
    """Some ink inside that block, whatever the page."""

    ink = Box(20, 15, 50, 35)

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, image, grow=2):
        assert image.ndim == 3 and image.shape[2] == 3, "expects an RGB array"
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[self.ink.y0 : self.ink.y1, self.ink.x0 : self.ink.x1] = 255
        return mask


class StubLama:
    """A painter that does what LaMa does, crudely: fill from what is around."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, page, hole):
        assert page.shape[:2] == hole.shape[:2], "the hole is not the page's size"
        around = page[hole == 0]
        out = page.copy()
        out[hole > 0] = (
            np.median(around, axis=0) if len(around) else np.uint8(255)
        )
        return out


_stub_lama = mock.patch.object(inpaint, "Lama", StubLama)

Lama = inpaint.Lama


def setUpModule():
    _stub_lama.start()


def tearDownModule():
    _stub_lama.stop()


class StubReader:
    """The size of whatever it was handed, so the crop can be checked."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, image, boxes, language=None):
        code = (language or languages.DEFAULT).code
        return [f"{code} {box.w}×{box.h}" for box in boxes]


def client():
    return server.create_app().test_client()


class TestBox(unittest.TestCase):
    def test_corners_are_put_in_order(self):
        self.assertEqual(Box.from_list([60, 40, 10, 5]), Box(10, 5, 60, 40))

    def test_floats_and_strings_are_rounded(self):
        self.assertEqual(Box.from_list(["1.4", 2.6, 30, 40.0]), Box(1, 3, 30, 40))

    def test_clipping_keeps_the_box_on_the_page(self):
        clipped = Box(-20, -5, 400, 300).clipped(200, 140)
        self.assertEqual(clipped, Box(0, 0, 200, 140))

    def test_a_box_wholly_off_the_page_comes_back_empty(self):
        clipped = Box(500, 500, 600, 600).clipped(200, 140)
        self.assertEqual((clipped.w, clipped.h), (0, 0))


class TestMarked(unittest.TestCase):
    """Boxes and a mask gathered into one greyscale page of what to hide."""

    box = Box(10, 10, 60, 40)
    elsewhere = Box(100, 100, 140, 130)

    def marks(self, boxes=None, mask=None) -> np.ndarray:
        return np.array(render.marked((200, 140), boxes or [self.box], mask))

    def test_the_box_is_marked_and_the_rest_is_not(self):
        marks = self.marks()
        self.assertTrue((patch(marks, self.box) == 255).all())
        self.assertEqual(marks[0, 0], 0)
        self.assertEqual(marks[45, 65], 0)

    def test_the_far_edge_is_exclusive(self):
        marks = self.marks()
        self.assertEqual(marks[39, 59], 255)
        self.assertEqual(marks[40, 60], 0)

    def test_an_empty_box_marks_nothing(self):
        self.assertFalse(self.marks([Box(10, 10, 10, 40)]).any())

    def test_a_mask_is_taken_in_alongside_the_boxes(self):
        marks = self.marks(mask=stencil(self.elsewhere))
        self.assertTrue((patch(marks, self.box) == 255).all())
        self.assertTrue((patch(marks, self.elsewhere) == 255).all())

    def test_a_light_hand_over_a_box_does_not_thin_it(self):
        marks = self.marks(mask=stencil(self.box, fill=40))
        self.assertTrue((patch(marks, self.box) == 255).all())


class TestHidden(unittest.TestCase):
    """Which of the two ways of hiding is taken, and which is taken unasked."""

    def marks(self) -> Image.Image:
        return render.marked((200, 140), [INK])

    def test_white_paints_the_mark_flat(self):
        out = render.hidden(toned(), self.marks(), render.WHITE_OUT)
        self.assertTrue((patch(out, INK) == 255).all())

    def test_the_art_around_it_is_what_it_fills_with_unasked(self):
        filled = patch(render.hidden(toned(), self.marks()), INK).astype(int)
        self.assertFalse((filled == 255).any(), "the mark was painted white")
        self.assertTrue((abs(filled - TONE[0]) <= NEAR).all())

    def test_the_original_is_left_alone(self):
        original = toned()
        render.hidden(original, self.marks())
        self.assertEqual(tuple(np.array(original)[0, 0]), TONE)


class TestFill(unittest.TestCase):
    """Making the fill out of the page around the mark."""

    def test_a_mark_comes_back_as_what_surrounds_it(self):
        filled = patch(inpaint.fill(toned(), stencil(INK)), INK).astype(int)
        self.assertTrue((abs(filled - TONE[0]) <= NEAR).all(), "the ink is still there")

    def test_nothing_marked_leaves_the_page_alone(self):
        out = inpaint.fill(toned(), stencil())
        self.assertTrue((np.array(out) == np.array(toned())).all())

    def test_grey_lays_on_only_some_of_the_fill(self):
        filled = patch(inpaint.fill(toned(), stencil(INK, fill=128)), INK).astype(int)
        self.assertTrue((filled > DARK[0] + NEAR).all(), "no fill was laid on")
        self.assertTrue((filled < TONE[0] - NEAR).all(), "all of it was laid on")

    def test_the_rim_around_a_mark_is_not_what_it_is_made_of(self):
        filled = patch(inpaint.fill(toned(rim=2), stencil(INK)), INK).astype(int)
        self.assertTrue((abs(filled - TONE[0]) <= NEAR).all(), "the rim was read")

    def test_the_rim_is_still_left_where_it_was(self):
        out = np.array(inpaint.fill(toned(rim=2), stencil(INK)))
        self.assertEqual(tuple(out[INK.y0 - 1, INK.x0]), (100, 100, 100))

    def test_a_page_marked_all_over_has_nothing_left_to_look_at(self):
        out = inpaint.fill(toned(), stencil(Box(0, 0, 200, 140)))
        self.assertTrue((np.array(out) == 255).all())

    def test_the_original_is_left_alone(self):
        original = toned()
        inpaint.fill(original, stencil(Box(0, 0, 200, 140)))
        self.assertEqual(tuple(np.array(original)[0, 0]), TONE)


class TestPainter(unittest.TestCase):
    """The fill a model makes, and the seam it is laid back through."""

    def marker(self):
        """A painter that signs its work, so its pixels can be told from Telea's."""
        seen = []

        def paint(page, hole):
            seen.append((page.shape, int(np.count_nonzero(hole))))
            out = page.copy()
            out[hole > 0] = (7, 200, 13)
            return out

        return paint, seen

    def test_the_painter_is_what_makes_the_fill(self):
        paint, seen = self.marker()
        out = np.array(inpaint.fill(toned(), stencil(INK), paint))
        self.assertEqual(tuple(out[int(INK.cy), int(INK.cx)]), (7, 200, 13))
        self.assertEqual(len(seen), 1, "the painter was not asked")

    def test_nothing_outside_the_mark_is_painted_over(self):
        paint, _ = self.marker()
        out = np.array(inpaint.fill(toned(), stencil(INK), paint))
        self.assertEqual(tuple(out[INK.y0 - 1, INK.x0]), TONE)
        self.assertEqual(tuple(out[INK.y1 + 1, INK.x0]), TONE)

    def test_a_page_marked_all_over_never_reaches_the_painter(self):
        paint, seen = self.marker()
        out = inpaint.fill(toned(), stencil(Box(0, 0, 200, 140)), paint)
        self.assertTrue((np.array(out) == 255).all())
        self.assertEqual(seen, [], "the painter was asked to make a page up")

    def test_a_crop_is_padded_out_to_whole_blocks(self):
        made = Lama.__new__(Lama)
        made._lock = threading.Lock()

        class Session:
            def run(self, wanted, feed):
                image = feed["image"]
                assert image.shape[2] % inpaint.BLOCK == 0, image.shape
                assert image.shape[3] % inpaint.BLOCK == 0, image.shape
                return [np.zeros_like(image)]

        made.session = Session()
        crop = np.full((50, 30, 3), 200, np.uint8)
        hole = np.zeros((50, 30), np.uint8)
        hole[10:20, 10:20] = 255
        self.assertEqual(made.patch(crop, hole).shape, crop.shape)

    def test_only_the_marked_pieces_of_a_page_go_through(self):
        hole = np.zeros((400, 400), np.uint8)
        hole[20:40, 20:40] = 255
        hole[300:320, 300:320] = 255
        found = inpaint.patches(hole, 400, 400)
        self.assertEqual(len(found), 2)
        for x0, y0, x1, y1 in found:
            self.assertLess((x1 - x0) * (y1 - y0), 400 * 400)

    def test_a_crop_within_the_cap_goes_through_as_it_is(self):
        self.assertIsNone(Lama.working((800, 600)))

    def test_a_crop_over_the_cap_is_brought_down_to_it(self):
        wide, tall = Lama.working((2055, 2406))
        self.assertLessEqual(wide * tall, inpaint.LARGEST)
        self.assertAlmostEqual(wide / tall, 2406 / 2055, places=1, msg="shape changed")

    def test_a_scaled_crop_still_comes_back_the_size_it_went_in(self):
        made = Lama.__new__(Lama)
        made._lock = threading.Lock()

        class Session:
            def run(self, wanted, feed):
                image = feed["image"]
                assert image[0, 0].size <= inpaint.LARGEST * 1.1, "no smaller than it was"
                return [np.zeros_like(image)]

        made.session = Session()
        crop = np.full((2406, 2055, 3), 200, np.uint8)
        hole = np.zeros((2406, 2055), np.uint8)
        hole[1000:1100, 1000:1100] = 255
        self.assertEqual(made.patch(crop, hole).shape, crop.shape)

    def test_a_thin_stroke_survives_being_scaled_down(self):
        hole = np.zeros((2400, 2000), np.uint8)
        hole[1200:1202, 500:1500] = 255
        small = Lama.working(hole.shape)
        shrunk = cv2.resize(hole, small, interpolation=cv2.INTER_AREA)
        self.assertTrue((shrunk > 0).any(), "the mark was rounded away")

    def test_marks_close_together_go_through_as_one(self):
        hole = np.zeros((400, 400), np.uint8)
        hole[100:120, 100:120] = 255
        hole[100:120, 125:145] = 255
        self.assertEqual(len(inpaint.patches(hole, 400, 400)), 1)


class TestGrowIsScanIndependent(unittest.TestCase):
    """`grow` means the same thing however large the page was scanned."""

    def grown(self, width: int, height: int, grow: int = detect.GROW) -> int:
        """How many page pixels a lit spot spreads to, on a page this size."""
        seg = np.zeros((detect.INPUT_SIZE, detect.INPUT_SIZE), np.float32)
        seg[500:524, 500:524] = 1.0
        _, pad_w, pad_h = detect.letterbox(np.zeros((height, width, 3), np.uint8))
        mask = detect.page_mask(seg, width, height, pad_w, pad_h, grow)
        lit = np.flatnonzero(mask.any(axis=0))
        return int(lit[-1] - lit[0]) if len(lit) else 0

    def test_a_bigger_scan_of_one_page_is_grown_proportionally(self):
        small = self.grown(1000, 1400)
        large = self.grown(2480, 3508)
        self.assertAlmostEqual(large / small, 3508 / 1400, delta=0.15)

    def test_growing_by_nothing_still_grows_by_nothing(self):
        plain = self.grown(2480, 3508, 0)
        self.assertLess(plain, self.grown(2480, 3508, detect.GROW))

    def test_a_small_page_is_not_grown_away(self):
        self.assertGreater(self.grown(200, 140, detect.GROW), 0)


class TestPageMask(unittest.TestCase):
    """The padded square the model answers with, put back onto the page."""

    size = (200, 140)

    def pads(self):
        _, pad_w, pad_h = detect.letterbox(np.zeros((140, 200, 3), np.uint8))
        return pad_w, pad_h

    def seg(self, box: Box | None = None, value: float = 1.0):
        """A per-pixel map with ``box`` — in page pixels — lit."""
        seg = np.zeros((detect.INPUT_SIZE, detect.INPUT_SIZE), np.float32)
        if box is not None:
            scale = detect.INPUT_SIZE / 200
            seg[
                round(box.y0 * scale) : round(box.y1 * scale),
                round(box.x0 * scale) : round(box.x1 * scale),
            ] = value
        return seg

    def test_the_mask_comes_back_the_size_of_the_page(self):
        pad_w, pad_h = self.pads()
        mask = detect.page_mask(self.seg(), 200, 140, pad_w, pad_h)
        self.assertEqual(mask.shape, (140, 200))

    def test_the_lit_part_lands_where_it_was(self):
        pad_w, pad_h = self.pads()
        mask = detect.page_mask(self.seg(Box(10, 10, 60, 40)), 200, 140, pad_w, pad_h, 0)
        self.assertEqual(mask[25, 35], 255, "the middle of the ink is not marked")
        self.assertEqual(mask[120, 180], 0, "the far corner of the page is marked")

    def test_the_padding_is_taken_back_off(self):
        pad_w, pad_h = self.pads()
        seg = np.zeros((detect.INPUT_SIZE, detect.INPUT_SIZE), np.float32)
        seg[: detect.INPUT_SIZE - pad_h, : detect.INPUT_SIZE - pad_w] = 1.0
        mask = detect.page_mask(seg, 200, 140, pad_w, pad_h, 0)
        self.assertTrue((mask == 255).all())

    def test_the_model_has_to_be_sure_enough(self):
        pad_w, pad_h = self.pads()
        unsure = detect.page_mask(
            self.seg(Box(10, 10, 60, 40), value=0.4), 200, 140, pad_w, pad_h, 0
        )
        sure = detect.page_mask(
            self.seg(Box(10, 10, 60, 40), value=0.6), 200, 140, pad_w, pad_h, 0
        )
        self.assertEqual(unsure.max(), 0)
        self.assertEqual(sure.max(), 255)

    def test_growing_covers_more_than_not_growing(self):
        pad_w, pad_h = self.pads()
        seg = self.seg(Box(10, 10, 60, 40))
        tight = detect.page_mask(seg, 200, 140, pad_w, pad_h, 0)
        grown = detect.page_mask(seg, 200, 140, pad_w, pad_h, 3)
        self.assertGreater((grown > 0).sum(), (tight > 0).sum())


class TestCoverMask(unittest.TestCase):
    box = Box(10, 10, 60, 40)

    def test_white_in_the_mask_is_painted_out(self):
        out = render.cover_mask(page(), stencil(self.box))
        self.assertTrue((patch(out, self.box) == 255).all())
        self.assertEqual(tuple(np.array(out)[0, 0]), DARK)

    def test_black_in_the_mask_leaves_the_page_alone(self):
        out = render.cover_mask(page(), stencil())
        self.assertTrue((np.array(out) == np.array(page())).all())

    def test_grey_lays_on_only_some_of_the_white(self):
        out = render.cover_mask(page(), stencil(self.box, fill=128))
        inside = patch(out, Box(20, 20, 50, 30))
        self.assertTrue((inside > DARK[0]).all(), "no white was laid on")
        self.assertTrue((inside < 255).all(), "all of it was laid on")

    def test_the_original_is_left_alone(self):
        original = page()
        render.cover_mask(original, stencil(Box(0, 0, 200, 140)))
        self.assertEqual(tuple(np.array(original)[0, 0]), DARK)


def ballooned(
    balloon: Box = Box(120, 60, 480, 300),
    column: Box = Box(285, 100, 315, 260),
    size=(600, 800),
    ground=(255, 255, 255),
    ink=(0, 0, 0),
) -> Image.Image:
    """A page of artwork with one balloon on it, and a column of writing inside."""
    image = Image.new("RGB", size, (120, 120, 120))
    draw = ImageDraw.Draw(image)
    draw.ellipse(
        (balloon.x0, balloon.y0, balloon.x1 - 1, balloon.y1 - 1),
        fill=ground,
        outline=ink,
        width=3,
    )
    for y in range(column.y0, column.y1, 30):
        draw.rectangle((column.x0, y, column.x1 - 1, y + 20), fill=ink)
    return image


def run_together(
    small: Box = Box(80, 80, 300, 240),
    large: Box = Box(240, 180, 520, 520),
    join: Box = Box(255, 195, 285, 290),
    column: Box = Box(150, 110, 180, 210),
    size=(600, 800),
) -> Image.Image:
    """Two balloons whose outline is broken where they meet, writing in the small."""
    image = Image.new("RGB", size, (120, 120, 120))
    draw = ImageDraw.Draw(image)
    for balloon in (small, large):
        draw.ellipse(
            (balloon.x0, balloon.y0, balloon.x1 - 1, balloon.y1 - 1),
            fill=(255, 255, 255),
            outline=(0, 0, 0),
            width=3,
        )
    draw.rectangle((join.x0, join.y0, join.x1 - 1, join.y1 - 1), fill=(255, 255, 255))
    for y in range(column.y0, column.y1, 30):
        draw.rectangle((column.x0, y, column.x1 - 1, y + 20), fill=(0, 0, 0))
    return image


def stub_balloon(size=(300, 200)) -> Image.Image:
    """A page with a balloon drawn around the block :class:`StubRegions` finds."""
    image = Image.new("RGB", size, (120, 120, 120))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (2, 2, 96, 66), radius=12, fill=(255, 255, 255), outline=(0, 0, 0), width=3
    )
    ink = StubLetters.ink
    draw.rectangle((ink.x0, ink.y0, ink.x1 - 1, ink.y1 - 1), fill=(0, 0, 0))
    return image


def grey(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("L"))


class TestHolding(unittest.TestCase):
    """The largest rectangle around a block, which is not the largest anywhere."""

    def test_the_rectangle_is_opened_out_around_the_block(self):
        mask = np.zeros((20, 30), bool)
        mask[4:12, 5:25] = True
        self.assertEqual(bubble.holding(mask, Box(10, 6, 14, 9)), Box(5, 4, 25, 12))

    def test_a_larger_rectangle_the_block_is_not_in_does_not_win(self):
        mask = np.zeros((40, 40), bool)
        mask[0:6, 0:8] = True
        mask[10:30, 10:35] = True
        self.assertEqual(bubble.holding(mask, Box(2, 2, 5, 4)), Box(0, 0, 8, 6))

    def test_a_shape_that_does_not_hold_the_whole_block_has_no_rectangle(self):
        mask = np.zeros((20, 20), bool)
        mask[5:15, 5:15] = True
        self.assertIsNone(bubble.holding(mask, Box(10, 10, 18, 18)))

    def test_nothing_set_is_no_rectangle(self):
        self.assertIsNone(bubble.holding(np.zeros((10, 10), bool), Box(2, 2, 5, 5)))

    def test_the_block_is_still_held_when_the_search_is_scaled_down(self):
        mask = np.zeros((600, 600), np.uint8)
        mask[100:500, 100:500] = 255
        block = Box(203, 307, 251, 353)
        found = bubble.largest(mask, block)
        assert found is not None
        self.assertEqual(bubble.within(found, block), 1.0)


class TestSolid(unittest.TestCase):
    def test_a_hole_inside_the_shape_is_filled_in(self):
        region = np.zeros((40, 40), np.uint8)
        region[5:35, 5:35] = 255
        region[15:25, 15:25] = 0
        self.assertTrue((bubble.solid(region)[15:25, 15:25] == 255).all())

    def test_the_outside_is_left_outside(self):
        region = np.zeros((40, 40), np.uint8)
        region[5:35, 5:35] = 255
        self.assertTrue((bubble.solid(region)[0:4, 0:4] == 0).all())

    def test_a_shape_running_to_the_edge_does_not_swallow_the_window(self):
        region = np.zeros((40, 40), np.uint8)
        region[:, 0:10] = 255
        self.assertTrue((bubble.solid(region)[:, 20:] == 0).all())


class TestInside(unittest.TestCase):
    """The room inside a balloon the detector has already found."""

    balloon = Box(120, 60, 480, 300)
    column = Box(285, 100, 315, 260)

    def found(
        self, image: Image.Image, box: Box | None = None, room: Box | None = None
    ) -> Box | None:
        return bubble.inside(grey(image), room or self.balloon, box or self.column)

    def test_a_column_of_writing_answers_with_the_balloon_around_it(self):
        found = self.found(ballooned())
        assert found is not None
        self.assertGreater(found.w, self.column.w * 3)
        self.assertGreater(found.w, found.h, "English wants the wider way round")

    def test_the_answer_stays_inside_the_balloon(self):
        found = self.found(ballooned())
        assert found is not None
        self.assertGreater(found.x0, self.balloon.x0)
        self.assertGreater(found.y0, self.balloon.y0)
        self.assertLess(found.x1, self.balloon.x1)
        self.assertLess(found.y1, self.balloon.y1)

    def test_the_writing_is_still_covered_by_where_it_will_be_set(self):
        found = self.found(ballooned())
        assert found is not None
        self.assertEqual(bubble.within(found, self.column), 1.0)

    def test_the_next_balloon_along_is_no_longer_reachable(self):
        column = Box(150, 110, 180, 210)
        found = self.found(run_together(), column, Box(120, 60, 300, 260))
        assert found is not None
        self.assertEqual(bubble.within(found, column), 1.0)
        self.assertLess(found.y1, 261, "the room ran on into the other balloon")

    def test_the_column_is_not_measured_as_the_gap_beside_it(self):
        found = self.found(ballooned())
        assert found is not None
        self.assertGreater(found.x1 - found.x0, self.balloon.w / 2)

    def test_white_words_on_a_dark_balloon_are_found_the_same_way(self):
        image = ballooned(ground=(15, 15, 15), ink=(255, 255, 255))
        found = self.found(image)
        assert found is not None
        self.assertGreater(found.w, self.column.w * 3)

    def test_a_balloon_no_wider_than_the_words_is_left_alone(self):
        wide = Box(40, 170, 560, 230)
        words = Box(60, 180, 540, 220)
        image = ballooned(balloon=wide, column=words)
        self.assertIsNone(self.found(image, words, wide))

    def test_a_box_too_small_to_hold_lettering_has_no_answer(self):
        self.assertIsNone(self.found(ballooned(), Box(300, 180, 302, 182)))

    def test_a_balloon_that_does_not_hold_its_block_has_no_answer(self):
        self.assertIsNone(self.found(ballooned(), Box(500, 400, 560, 460)))

    def test_a_box_off_the_page_has_no_answer(self):
        self.assertIsNone(self.found(ballooned(), Box(900, 900, 1000, 1000)))


EM = 20


def lettering(x: int, y: int, columns: int, rows: int, em: int = EM) -> list[Box]:
    """Characters set solid on a square em, ``columns`` across and ``rows`` down."""
    ink = round(em * 0.85)
    return [
        Box(x + c * em, y + r * em, x + c * em + ink, y + r * em + ink)
        for c in range(columns)
        for r in range(rows)
    ]


class TestRegionBlocks(unittest.TestCase):
    """The wiring in Regions.__call__: decode, pad, tell the classes apart, sort."""

    size = (200, 140)

    def regions(self, found: list[tuple[int, Box, float]]) -> detect.Regions:
        made = detect.Regions.__new__(detect.Regions)
        labels = np.array([kind for kind, _, _ in found], np.int64)
        boxes = np.array([box.as_list() for _, box, _ in found], np.float32)
        scores = np.array([score for _, _, score in found], np.float32)
        made.run = lambda image: (labels, boxes, scores)
        return made

    def page(self) -> np.ndarray:
        return np.zeros((self.size[1], self.size[0], 3), np.uint8)

    def test_a_balloon_and_the_text_in_it_are_told_apart(self):
        blocks, balloons = self.regions(
            [
                (detect.BUBBLE, Box(10, 10, 120, 100), 0.98),
                (detect.TEXT_BUBBLE, Box(30, 30, 90, 80), 0.95),
            ]
        )(self.page())
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(balloons), 1)
        self.assertEqual(balloons[0], Box(10, 10, 120, 100))

    def test_text_in_no_balloon_is_still_a_block(self):
        blocks, balloons = self.regions(
            [(detect.TEXT_FREE, Box(30, 30, 90, 80), 0.9)]
        )(self.page())
        self.assertEqual(len(blocks), 1)
        self.assertEqual(balloons, [])

    def test_a_row_the_model_is_unsure_of_is_dropped(self):
        blocks, _ = self.regions(
            [
                (detect.TEXT_BUBBLE, Box(30, 30, 90, 80), 0.95),
                (detect.TEXT_BUBBLE, Box(10, 100, 40, 130), detect.REGIONS_CONF / 2),
            ]
        )(self.page())
        self.assertEqual(len(blocks), 1)

    def test_a_block_comes_back_with_a_margin_around_its_lettering(self):
        tight = Box(40, 40, 90, 100)
        [found] = self.regions([(detect.TEXT_BUBBLE, tight, 0.9)])(self.page())[0]
        self.assertLess(found.box.x0, tight.x0, "no margin on the left")
        self.assertLess(found.box.y0, tight.y0, "no margin on the top")
        self.assertGreater(found.box.x1, tight.x1, "no margin on the right")
        self.assertGreater(found.box.y1, tight.y1, "no margin on the bottom")

    def test_the_margin_never_runs_off_the_page(self):
        [found] = self.regions([(detect.TEXT_BUBBLE, Box(0, 0, 40, 40), 0.9)])(
            self.page()
        )[0]
        self.assertEqual(found.box, found.box.clipped(*self.size))

    def test_a_box_running_past_the_page_is_brought_back_onto_it(self):
        [found] = self.regions(
            [(detect.TEXT_BUBBLE, Box(-20, -10, 260, 190), 0.9)]
        )(self.page())[0]
        self.assertEqual(found.box, Box(0, 0, *self.size))

    def test_the_same_lettering_found_twice_comes_back_once(self):
        blocks, _ = self.regions(
            [
                (detect.TEXT_BUBBLE, Box(40, 40, 90, 100), 0.91),
                (detect.TEXT_BUBBLE, Box(42, 43, 89, 99), 0.62),
            ]
        )(self.page())
        self.assertEqual(len(blocks), 1)
        self.assertEqual(round(blocks[0].confidence, 2), 0.91, "the surer one went")

    def test_the_blocks_come_back_in_reading_order(self):
        found = self.regions(
            [
                (detect.TEXT_BUBBLE, Box(20, 60, 60, 100), 0.9),
                (detect.TEXT_BUBBLE, Box(130, 20, 170, 60), 0.9),
            ]
        )(self.page())[0]
        self.assertEqual(len(found), 2)
        self.assertLess(found[0].box.y0, found[1].box.y0)

    def test_blocks_level_with_each_other_are_read_right_to_left(self):
        found = self.regions(
            [
                (detect.TEXT_BUBBLE, Box(20, 20, 60, 60), 0.9),
                (detect.TEXT_BUBBLE, Box(130, 20, 170, 60), 0.9),
            ]
        )(self.page())[0]
        self.assertEqual(len(found), 2)
        self.assertGreater(found[0].box.x0, found[1].box.x0)

    def test_a_language_read_the_other_way_puts_them_the_other_way_round(self):
        made = self.regions(
            [
                (detect.TEXT_BUBBLE, Box(20, 20, 60, 60), 0.9),
                (detect.TEXT_BUBBLE, Box(130, 20, 170, 60), 0.9),
            ]
        )
        found = made(self.page(), rtl=False)[0]
        self.assertEqual(len(found), 2)
        self.assertLess(found[0].box.x0, found[1].box.x0)


class TestKeptPass(unittest.TestCase):
    """The last page's forward pass is kept, because it is asked for twice."""

    def detector(self):
        """:class:`Letters`, counting how often its net is actually run."""
        made = detect.Letters.__new__(detect.Letters)
        made._lock = threading.Lock()
        made._last = None
        made.passes = 0

        seg = np.zeros((1, 1, detect.INPUT_SIZE, detect.INPUT_SIZE), np.float32)

        class Net:
            def setInput(self, blob):
                pass

            def forward(self, names):
                made.passes += 1
                return [seg]

        made.net = Net()
        return made

    def finder(self):
        """:class:`Regions`, counting the same thing."""
        made = detect.Regions.__new__(detect.Regions)
        made._lock = threading.Lock()
        made._last = None
        made.passes = 0
        made.answers = ["labels", "boxes", "scores"]

        class Session:
            def run(self, wanted, feed):
                made.passes += 1
                return [
                    np.zeros((1, 1), np.int64),
                    np.zeros((1, 1, 4), np.float32),
                    np.zeros((1, 1), np.float32),
                ]

        made.session = Session()
        return made

    def page(self, fill: int = 0) -> np.ndarray:
        return np.full((140, 200, 3), fill, np.uint8)

    def test_the_same_page_twice_is_one_pass(self):
        made = self.detector()
        made.run(self.page())
        made.run(self.page())
        self.assertEqual(made.passes, 1)

    def test_a_different_page_is_a_new_pass(self):
        made = self.detector()
        made.run(self.page(0))
        made.run(self.page(7))
        self.assertEqual(made.passes, 2)

    def test_only_the_last_page_is_kept(self):
        made = self.detector()
        made.run(self.page(0))
        made.run(self.page(7))
        made.run(self.page(0))
        self.assertEqual(made.passes, 3, "more than one page was kept")

    def test_the_kept_answer_is_the_one_it_worked_out(self):
        made = self.detector()
        first = made.run(self.page())
        again = made.run(self.page())
        self.assertIs(first[0], again[0])
        self.assertEqual(first[1:], again[1:])

    def test_two_masks_off_one_page_share_the_pass(self):
        made = self.detector()
        made(self.page(), grow=2)
        made(self.page(), grow=8)
        self.assertEqual(made.passes, 1)

    def test_the_region_pass_is_kept_the_same_way(self):
        made = self.finder()
        made.run(self.page())
        made.run(self.page())
        self.assertEqual(made.passes, 1)

    def test_a_different_page_is_a_new_region_pass(self):
        made = self.finder()
        made.run(self.page(0))
        made.run(self.page(7))
        self.assertEqual(made.passes, 2)


class TestSuppressed(unittest.TestCase):
    """The same lettering found twice, thinned to the surest of them."""

    def test_a_piece_that_repeats_a_whole_block_is_dropped(self):
        blocks = [
            Block(Box(540, 300, 620, 500), 0.91),
            Block(Box(543, 305, 619, 501), 0.62),
            Block(Box(540, 640, 620, 840), 0.91),
        ]
        kept = detect.suppressed(blocks)
        self.assertEqual(len(kept), 2)
        self.assertNotIn(0.62, [b.confidence for b in kept], "the surer one went")

    def test_a_small_block_wholly_inside_a_large_one_is_dropped(self):
        blocks = [Block(Box(0, 0, 200, 200), 0.9), Block(Box(50, 50, 90, 90), 0.8)]
        self.assertEqual(len(detect.suppressed(blocks)), 1)

    def test_blocks_merely_next_to_each_other_are_both_kept(self):
        blocks = [Block(Box(0, 0, 100, 100), 0.9), Block(Box(110, 0, 210, 100), 0.9)]
        self.assertEqual(len(detect.suppressed(blocks)), 2)

    def test_nothing_is_dropped_where_nothing_repeats(self):
        blocks = [Block(Box(0, 0, 50, 50), 0.9), Block(Box(0, 60, 50, 110), 0.8)]
        self.assertEqual(len(detect.suppressed(blocks)), 2)


class TestDivided(unittest.TestCase):
    """One balloon shared out between the blocks written in it."""

    room = Box(100, 100, 500, 400)

    def test_two_blocks_side_by_side_get_a_side_each(self):
        blocks = [Box(150, 150, 200, 350), Box(400, 150, 450, 350)]
        left, right = bubble.divided(self.room, blocks)
        self.assertEqual(left.covers(right), 0.0, "the two shares overlap")
        self.assertLess(left.x1, right.x0 + 1)
        self.assertEqual((left.y0, left.y1), (self.room.y0, self.room.y1))

    def test_two_blocks_one_above_the_other_get_a_half_each(self):
        blocks = [Box(150, 120, 450, 180), Box(150, 320, 450, 380)]
        top, bottom = bubble.divided(self.room, blocks)
        self.assertEqual(top.covers(bottom), 0.0)
        self.assertLess(top.y1, bottom.y0 + 1)
        self.assertEqual((top.x0, top.x1), (self.room.x0, self.room.x1))

    def test_the_shares_come_back_against_the_blocks_they_belong_to(self):
        blocks = [Box(400, 150, 450, 350), Box(150, 150, 200, 350)]
        first, second = bubble.divided(self.room, blocks)
        self.assertGreater(first.cx, second.cx, "the shares were handed back swapped")

    def test_every_share_holds_the_block_it_was_cut_for(self):
        blocks = [Box(150, 150, 200, 350), Box(280, 150, 330, 350), Box(400, 150, 450, 350)]
        for share, block in zip(bubble.divided(self.room, blocks), blocks):
            self.assertLessEqual(share.x0, block.x0)
            self.assertGreaterEqual(share.x1, block.x1)

    def test_blocks_two_across_and_two_down_get_a_quarter_each(self):
        """The arrangement one line of cuts cannot describe."""
        blocks = [
            Box(140, 130, 220, 220),
            Box(380, 130, 460, 220),
            Box(140, 280, 220, 370),
            Box(380, 280, 460, 370),
        ]
        shares = bubble.divided(self.room, blocks)
        for one in range(len(shares)):
            for other in range(one + 1, len(shares)):
                self.assertEqual(
                    shares[one].covers(shares[other]),
                    0.0,
                    f"shares {one} and {other} overlap",
                )
        for share, block in zip(shares, blocks):
            self.assertGreaterEqual(share.covers(block), 0.99, "a block lost its share")

    def test_blocks_in_an_ell_are_each_given_their_own_room(self):
        blocks = [
            Box(380, 130, 460, 220),
            Box(140, 200, 220, 300),
            Box(380, 280, 460, 370),
        ]
        shares = bubble.divided(self.room, blocks)
        for one in range(len(shares)):
            for other in range(one + 1, len(shares)):
                self.assertEqual(shares[one].covers(shares[other]), 0.0)
        for share, block in zip(shares, blocks):
            self.assertGreaterEqual(share.covers(block), 0.99)

    def test_blocks_that_overlap_every_way_still_get_a_piece_each(self):
        blocks = [Box(150, 150, 400, 350), Box(160, 160, 410, 360)]
        first, second = bubble.divided(self.room, blocks)
        self.assertEqual(first.covers(second), 0.0)
        self.assertNotEqual(first, second)

    def test_the_shares_use_up_the_whole_balloon(self):
        blocks = [Box(150, 150, 200, 350), Box(400, 150, 450, 350)]
        shares = bubble.divided(self.room, blocks)
        self.assertEqual(min(s.x0 for s in shares), self.room.x0)
        self.assertEqual(max(s.x1 for s in shares), self.room.x1)


class TestAssigned(unittest.TestCase):
    """Which balloon each block was written in."""

    def test_a_block_inside_a_balloon_is_given_it(self):
        blocks = [Box(30, 30, 70, 70)]
        self.assertEqual(bubble.assigned(blocks, [Box(0, 0, 100, 100)]), [0])

    def test_lettering_in_no_balloon_is_given_none(self):
        blocks = [Box(300, 300, 340, 340)]
        self.assertEqual(bubble.assigned(blocks, [Box(0, 0, 100, 100)]), [None])

    def test_two_blocks_in_one_balloon_are_both_given_it(self):
        blocks = [Box(10, 10, 40, 40), Box(50, 50, 90, 90)]
        self.assertEqual(bubble.assigned(blocks, [Box(0, 0, 100, 100)]), [0, 0])

    def test_each_block_is_given_the_balloon_it_is_in(self):
        blocks = [Box(210, 10, 240, 40), Box(10, 10, 40, 40)]
        balloons = [Box(0, 0, 100, 100), Box(200, 0, 300, 100)]
        self.assertEqual(bubble.assigned(blocks, balloons), [1, 0])

    def test_the_smaller_of_two_balloons_around_a_block_wins(self):
        blocks = [Box(40, 40, 60, 60)]
        balloons = [Box(0, 0, 200, 200), Box(30, 30, 70, 70)]
        self.assertEqual(bubble.assigned(blocks, balloons), [1])

    def test_a_balloon_merely_reaching_over_a_block_does_not_hold_it(self):
        blocks = [Box(80, 80, 160, 160)]
        self.assertEqual(bubble.assigned(blocks, [Box(0, 0, 100, 100)]), [None])


class TestCropped(unittest.TestCase):
    def test_what_comes_back_is_the_part_they_have_in_common(self):
        room, cell = Box(0, 0, 100, 100), Box(50, 20, 200, 80)
        self.assertEqual(bubble.cropped(room, cell), Box(50, 20, 100, 80))

    def test_a_box_already_inside_is_left_as_it_is(self):
        room = Box(10, 10, 20, 20)
        self.assertEqual(bubble.cropped(room, Box(0, 0, 100, 100)), room)

    def test_two_that_do_not_meet_leave_nothing(self):
        empty = bubble.cropped(Box(0, 0, 10, 10), Box(50, 50, 60, 60))
        self.assertEqual((empty.w, empty.h), (0, 0))


class TestWithin(unittest.TestCase):
    def test_a_block_wholly_inside_a_room_is_all_of_it(self):
        self.assertEqual(bubble.within(Box(0, 0, 100, 100), Box(10, 10, 30, 30)), 1.0)

    def test_half_a_block_hanging_out_is_half_of_it(self):
        self.assertEqual(bubble.within(Box(0, 0, 100, 100), Box(50, 0, 150, 100)), 0.5)

    def test_a_block_larger_than_the_room_is_measured_against_itself(self):
        self.assertEqual(bubble.within(Box(0, 0, 50, 100), Box(0, 0, 100, 100)), 0.5)

    def test_a_block_nowhere_near_is_none_of_it(self):
        self.assertEqual(bubble.within(Box(0, 0, 10, 10), Box(50, 50, 60, 60)), 0.0)


class TestRooms(unittest.TestCase):
    """Sharing one balloon out, which is what keeps two translations apart."""

    page = np.full((600, 600), 255, np.uint8)

    def rooms(
        self, boxes: list[Box], balloons: list[Box], answers: list[Box | None]
    ) -> list[Box | None]:
        def inside(_grey, _balloon, block):
            return answers[boxes.index(block)]

        with mock.patch.object(bubble, "inside", side_effect=inside):
            return bubble.rooms(self.page, boxes, balloons)

    def lettered(self, boxes: list[Box], found: list[Box | None]) -> list[Box]:
        """Where each block is really lettered: its balloon, or its own box."""
        return [box if room is None else room for box, room in zip(boxes, found)]

    def test_a_block_in_no_balloon_keeps_its_own_box(self):
        boxes = [Box(200, 200, 240, 300)]
        self.assertEqual(self.rooms(boxes, [], [None]), [None])

    def test_two_balloons_that_do_not_touch_are_left_alone(self):
        boxes = [Box(60, 200, 100, 300), Box(400, 200, 440, 300)]
        balloons = [Box(20, 150, 220, 350), Box(360, 150, 560, 350)]
        answers = [Box(20, 150, 220, 350), Box(360, 150, 560, 350)]
        self.assertEqual(self.rooms(boxes, balloons, answers), answers)

    def test_two_blocks_in_one_balloon_are_cut_a_side_each(self):
        boxes = [Box(150, 200, 190, 400), Box(300, 200, 340, 400)]
        balloons = [Box(100, 180, 400, 420)]
        answers = [Box(100, 180, 400, 420)] * 2

        found = self.rooms(boxes, balloons, answers)
        first, second = self.lettered(boxes, found)
        self.assertEqual(first.covers(second), 0.0, "the two translations overlap")

    def test_every_block_keeps_its_own_words_inside_its_share(self):
        boxes = [Box(120, 200, 160, 400), Box(260, 200, 300, 400), Box(400, 200, 440, 400)]
        balloons = [Box(100, 180, 460, 420)]
        answers = [Box(100, 180, 460, 420)] * 3

        found = self.rooms(boxes, balloons, answers)
        rooms = self.lettered(boxes, found)
        for at, room in enumerate(rooms):
            self.assertGreaterEqual(
                bubble.within(room, boxes[at]), 0.99, f"block {at} is not in its room"
            )
        for one in range(len(rooms)):
            for other in range(one + 1, len(rooms)):
                self.assertEqual(rooms[one].covers(rooms[other]), 0.0)

    def test_a_block_in_another_balloon_is_not_cut_against_this_one(self):
        boxes = [Box(120, 300, 160, 400), Box(200, 300, 240, 400), Box(430, 150, 470, 250)]
        balloons = [Box(100, 280, 420, 420), Box(400, 120, 560, 300)]
        answers = [Box(100, 280, 420, 420), Box(100, 280, 420, 420), Box(400, 120, 560, 300)]

        found = self.rooms(boxes, balloons, answers)
        self.assertEqual(found[2], answers[2], "a balloon of its own was cut up")
        first, second = self.lettered(boxes, found)[:2]
        self.assertEqual(first.covers(second), 0.0)
        for at, room in enumerate(self.lettered(boxes, found)):
            self.assertGreaterEqual(bubble.within(room, boxes[at]), 0.99)

    def test_a_share_that_no_longer_holds_its_block_is_refused(self):
        boxes = [Box(150, 200, 400, 400), Box(160, 210, 410, 410)]
        balloons = [Box(100, 180, 460, 440)]
        answers = [Box(100, 180, 460, 440)] * 2

        found = self.rooms(boxes, balloons, answers)
        for at, room in enumerate(found):
            if room is not None:
                self.assertGreaterEqual(bubble.within(room, boxes[at]), 0.85)


def reply(content: str = "", thinking: str = "") -> dict:
    """One answer from Ollama, shaped the way it shapes them."""
    return {"message": {"role": "assistant", "content": content, "thinking": thinking}}


def translations(*texts) -> str:
    return json.dumps({"translations": list(texts)})


class TestOllama(unittest.TestCase):
    """The talking to Ollama. Nothing here goes near the network."""

    def test_models_come_back_named_and_sorted(self):
        listing = {"models": [{"name": "qwen3:8b"}, {"name": "gemma4:12b"}]}
        with mock.patch.object(ollama, "ask", return_value=listing):
            self.assertEqual(ollama.models(), ["gemma4:12b", "qwen3:8b"])

    def test_a_page_goes_over_in_one_request(self):
        asked = []

        def ask(path, body=None, **kwargs):
            asked.append((path, body))
            return reply(translations("Good morning", "What is this?"))

        with mock.patch.object(ollama, "ask", ask):
            got = ollama.translate(["おはよう", "なにこれ"], "gemma4:12b")
        self.assertEqual(got, ["Good morning", "What is this?"])
        self.assertEqual(len(asked), 1, "the lines were not sent together")
        self.assertEqual(asked[0][0], "/api/chat")
        self.assertIn("1. おはよう", asked[0][1]["messages"][1]["content"])

    def test_the_answer_is_taken_from_thinking_when_content_is_empty(self):
        with mock.patch.object(
            ollama, "ask", return_value=reply("", translations("Good morning"))
        ):
            done = ollama.translate(["おはよう"], "m")
        self.assertEqual(done, ["Good morning"])

    def test_a_fenced_answer_is_still_read(self):
        fenced = f"```json\n{translations('Good morning')}\n```"
        with mock.patch.object(ollama, "ask", return_value=reply(fenced)):
            done = ollama.translate(["おはよう"], "m")
        self.assertEqual(done, ["Good morning"])

    def test_a_miscounted_page_is_asked_again_whole(self):
        answers = [
            reply(translations("only one")),
            reply(translations("first", "second")),
        ]
        with mock.patch.object(ollama, "ask", side_effect=answers) as ask:
            got = ollama.translate(["いち", "に"], "m")
        self.assertEqual(got, ["first", "second"])
        self.assertEqual(ask.call_count, 2, "it did not simply ask again")

    def test_asking_again_shows_the_model_what_it_did(self):
        asked = []

        def ask(path, body=None, **kwargs):
            asked.append(body["messages"])
            return reply(translations("only one"))

        with mock.patch.object(ollama, "ask", ask):
            ollama.translate(["いち", "に"], "m")
        again = asked[1]
        self.assertEqual(again[2]["role"], "assistant")
        self.assertIn("only one", again[2]["content"])
        self.assertIn("1 translations for 2 lines", again[3]["content"])

    def test_losing_count_twice_falls_back_to_one_line_at_a_time(self):
        answers = [
            reply(translations("only one")),
            reply(translations("still only one")),
            reply(translations("first")),
            reply(translations("second")),
        ]
        with mock.patch.object(ollama, "ask", side_effect=answers) as ask:
            got = ollama.translate(["いち", "に"], "m")
        self.assertEqual(got, ["first", "second"])
        self.assertEqual(ask.call_count, 4, "it did not ask again line by line")

    def test_the_context_is_asked_for(self):
        asked = []

        def ask(path, body=None, **kwargs):
            asked.append(body)
            return reply(translations("Good morning"))

        with mock.patch.object(ollama, "ask", ask):
            ollama.translate(["おはよう"], "m")
        self.assertEqual(asked[0]["options"]["num_ctx"], ollama.CONTEXT)
        self.assertGreater(ollama.CONTEXT, 4096)

    def test_an_empty_line_stays_empty_and_is_never_sent(self):
        def ask(path, body=None, **kwargs):
            sent_lines = body["messages"][1]["content"].splitlines()
            self.assertEqual(
                sent_lines,
                ["1. おはよう", "2. 行こう"],
                "the blank was sent over, or the rest were not renumbered",
            )
            return reply(translations("Good morning", "Let's go"))

        with mock.patch.object(ollama, "ask", ask):
            got = ollama.translate(["おはよう", "   ", "行こう"], "m")
        self.assertEqual(got, ["Good morning", "", "Let's go"])

    def test_nothing_to_translate_asks_nothing(self):
        with mock.patch.object(ollama, "ask", side_effect=AssertionError("asked")):
            self.assertEqual(ollama.translate(["", "  "], "m"), ["", ""])

    def test_the_briefing_says_what_to_translate_into(self):
        self.assertIn("Dutch", ollama.briefing("Dutch"))

    def test_the_briefing_says_what_the_page_was_written_in(self):
        self.assertIn("Korean", ollama.briefing("Dutch", source="Korean"))
        self.assertIn("Japanese", ollama.briefing("Dutch"))

    def test_the_source_reaches_the_request(self):
        asked = []

        def ask(path, body=None, **kwargs):
            asked.append(body["messages"][0]["content"])
            return reply(translations("Goedemorgen"))

        with mock.patch.object(ollama, "ask", ask):
            ollama.translate(["안녕"], "m", "Dutch", source="Korean")
        self.assertIn("Korean", asked[0])

    def test_a_briefing_of_your_own_is_used_instead(self):
        said = ollama.briefing("Dutch", "Turn this into {target}, in pirate.")
        self.assertEqual(said, "Turn this into Dutch, in pirate.")

    def test_a_briefing_may_have_braces_of_its_own(self):
        said = ollama.briefing("Dutch", 'Answer like {"a": 1} but in {target}.')
        self.assertEqual(said, 'Answer like {"a": 1} but in Dutch.')

    def test_the_briefing_reaches_the_request(self):
        asked = []

        def ask(path, body=None, **kwargs):
            asked.append(body["messages"][0]["content"])
            return reply(translations("Goedemorgen"))

        with mock.patch.object(ollama, "ask", ask):
            ollama.translate(["おはよう"], "m", "Dutch", system="Be brief in {target}.")
        self.assertTrue(asked[0].startswith("Be brief in Dutch."), asked[0])

    def test_the_briefing_holds_when_it_falls_back_to_one_at_a_time(self):
        asked = []

        def ask(path, body=None, **kwargs):
            asked.append(body["messages"][0]["content"])
            return reply(translations("only one")) if len(asked) <= 2 else reply(
                translations("a line")
            )

        with mock.patch.object(ollama, "ask", ask):
            ollama.translate(["いち", "に"], "m", "Dutch", system="Mine.")
        self.assertEqual(len(asked), 4, "it did not ask again line by line")
        for said in asked:
            self.assertTrue(said.startswith("Mine."), said)

    def test_where_ollama_is_can_be_said(self):
        self.assertEqual(ollama.base("http://elsewhere:11434/"), "http://elsewhere:11434")


class TestSaidAboutEachLine(unittest.TestCase):
    """What the caller knows about a block and the model cannot see."""

    def asking(self, answer):
        """Ollama patched to answer that, collecting the whole request."""
        asked = []

        def ask(path, body=None, **kwargs):
            asked.append(body["messages"])
            return answer

        return asked, mock.patch.object(ollama, "ask", ask)

    def test_a_line_says_whether_it_is_spoken(self):
        asked, patched = self.asking(reply(translations("Morning", "THUD")))
        with patched:
            ollama.translate(["おはよう", "ドン"], "m", kinds=[SPEECH, FREE])
        system, page = asked[0][0]["content"], asked[0][1]["content"]
        self.assertIn("1. [speech] おはよう", page)
        self.assertIn("2. [free] ドン", page)
        self.assertIn(ollama.KINDS_NOTE, system)

    def test_a_line_says_how_much_room_it_has(self):
        asked, patched = self.asking(reply(translations("Morning")))
        with patched:
            ollama.translate(["おはよう"], "m", budgets=[28])
        system, page = asked[0][0]["content"], asked[0][1]["content"]
        self.assertIn("1. <=28 おはよう", page)
        self.assertIn(ollama.BUDGET_NOTE, system)

    def test_a_line_with_nothing_said_about_it_is_sent_as_it_was(self):
        asked, patched = self.asking(reply(translations("Morning")))
        with patched:
            ollama.translate(["おはよう"], "m")
        system, page = asked[0][0]["content"], asked[0][1]["content"]
        self.assertEqual(page, "1. おはよう")
        self.assertNotIn(ollama.KINDS_NOTE, system)
        self.assertNotIn(ollama.BUDGET_NOTE, system)

    def test_what_is_said_about_a_line_follows_a_blank_being_dropped(self):
        asked, patched = self.asking(reply(translations("Morning", "THUD")))
        with patched:
            ollama.translate(
                ["おはよう", "   ", "ドン"],
                "m",
                kinds=[SPEECH, SPEECH, FREE],
                budgets=[28, 99, 12],
            )
        page = asked[0][1]["content"]
        self.assertEqual(
            page.splitlines(),
            ["1. [speech] <=28 おはよう", "2. [free] <=12 ドン"],
        )

    def test_what_is_said_about_a_line_holds_when_it_is_asked_about_alone(self):
        asked, patched = self.asking(reply(translations("only one")))
        with patched:
            ollama.translate(["おはよう", "ドン"], "m", kinds=[SPEECH, FREE])
        self.assertEqual(len(asked), 4)
        self.assertIn("1. [speech] おはよう", asked[2][1]["content"])
        self.assertIn("1. [free] ドン", asked[3][1]["content"])


class TestSaidTwice(unittest.TestCase):
    """The same words twice on one page. Asked about once, lettered twice."""

    def sending(self, answer):
        asked = []

        def ask(path, body=None, **kwargs):
            asked.append(body["messages"][1]["content"])
            return answer

        return asked, mock.patch.object(ollama, "ask", ask)

    def test_the_same_line_twice_is_one_question(self):
        asked, patched = self.sending(reply(translations("...", "Let's go")))
        with patched:
            done = ollama.translate(["……", "行こう", "……"], "m")
        self.assertEqual(asked[0].splitlines(), ["1. ……", "2. 行こう"])
        self.assertEqual(done, ["...", "Let's go", "..."])

    def test_the_same_words_in_different_lettering_are_two_questions(self):
        asked, patched = self.sending(reply(translations("Wham!", "wham")))
        with patched:
            done = ollama.translate(
                ["ドン", "ドン"], "m", kinds=[FREE, SPEECH]
            )
        self.assertEqual(len(asked[0].splitlines()), 2)
        self.assertEqual(done, ["Wham!", "wham"])

    def test_the_tightest_room_of_the_two_is_the_one_asked_for(self):
        asked, patched = self.sending(reply(translations("Morning")))
        with patched:
            ollama.translate(["おはよう", "おはよう"], "m", budgets=[40, 18])
        self.assertEqual(asked[0], "1. <=18 おはよう")

    def test_a_page_is_not_pushed_to_vary_what_it_repeats(self):
        asked = []

        def ask(path, body=None, **kwargs):
            asked.append(body["options"])
            return reply(translations("Morning"))

        with mock.patch.object(ollama, "ask", ask):
            ollama.translate(["おはよう"], "m")
        self.assertEqual(asked[0]["repeat_penalty"], 1.0)
        self.assertEqual(asked[0]["num_predict"], ollama.PREDICT)


class TestOllamaHost(unittest.TestCase):
    """Finding Ollama when nothing said where it is."""

    def setUp(self):
        for patch in (
            mock.patch.dict(os.environ, {ollama.OLLAMA_ENV: ""}),
            mock.patch.object(ollama, "_answering", None),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def answering(self, *hosts: str):
        """An Ollama that answers at those hosts and nowhere else."""
        self.asked: list[str] = []

        def ask(path, body=None, timeout=None, host=None):
            self.asked.append(host)
            if host not in hosts:
                raise ollama.Unreachable(f"no ollama answering at {host}")
            return {"models": []}

        return mock.patch.object(ollama, "ask", ask)

    def test_the_host_that_answers_is_the_one_used(self):
        with self.answering("http://host.docker.internal:11434"):
            self.assertEqual(ollama.base(), "http://host.docker.internal:11434")

    def test_every_usual_place_is_tried(self):
        with self.answering("http://host.containers.internal:11434"):
            self.assertEqual(ollama.base(), "http://host.containers.internal:11434")
        self.assertEqual(self.asked, list(ollama.OLLAMA_HOSTS))

    def test_the_one_that_answered_is_not_looked_for_again(self):
        with self.answering(*ollama.OLLAMA_HOSTS):
            ollama.base()
            ollama.base()
        self.assertEqual(self.asked, [ollama.OLLAMA_HOSTS[0]], "it looked twice")

    def test_a_miss_is_not_remembered(self):
        with self.answering():
            with self.assertRaises(ollama.Unreachable):
                ollama.base()
        with self.answering(ollama.OLLAMA_HOSTS[-1]):
            self.assertEqual(ollama.base(), ollama.OLLAMA_HOSTS[-1])

    def test_nothing_answering_says_everywhere_it_looked(self):
        with self.answering():
            with self.assertRaises(ollama.Unreachable) as caught:
                ollama.base()
        for host in ollama.OLLAMA_HOSTS:
            self.assertIn(host, str(caught.exception))

    def test_a_host_that_was_set_is_never_looked_for(self):
        with mock.patch.dict(os.environ, {ollama.OLLAMA_ENV: "http://said:11434/"}):
            with self.answering(*ollama.OLLAMA_HOSTS):
                self.assertEqual(ollama.base(), "http://said:11434")
        self.assertEqual(self.asked, [], "it went looking anyway")


class TestLanguages(unittest.TestCase):
    """The table both ends look a language up in."""

    def test_a_language_is_found_by_its_code(self):
        self.assertEqual(languages.of("ko").name, "Korean")

    def test_the_case_of_a_code_does_not_matter(self):
        self.assertEqual(languages.of("ZH-HANT"), languages.of("zh-Hant"))

    def test_nothing_asked_for_means_the_one_this_was_written_for(self):
        self.assertEqual(languages.of(None), languages.DEFAULT)
        self.assertEqual(languages.of("  ").code, "ja")

    def test_a_language_nothing_here_reads_is_not_answered_with_one(self):
        with self.assertRaises(KeyError):
            languages.of("klingon")

    def test_japanese_is_the_only_one_manga_ocr_is_asked_about(self):
        for language in languages.LANGUAGES:
            if language.reader == languages.PPOCR:
                self.assertTrue(language.recogniser, language.code)
            else:
                self.assertEqual(language.code, "ja")


def crop_of(*groups: list[Box], size, ground=(255, 255, 255), ink=DARK) -> Image.Image:
    """A crop with those glyphs drawn in it, the way a reader is handed one."""
    image = Image.new("RGB", size, ground)
    draw = ImageDraw.Draw(image)
    for group in groups:
        for glyph in group:
            draw.rectangle((glyph.x0, glyph.y0, glyph.x1 - 1, glyph.y1 - 1), fill=ink)
    return image


def column(rows: int, at: int = 0) -> list[Box]:
    return lettering(2 + at * EM, 2, 1, rows, EM)


class TestLines(unittest.TestCase):
    """Taking a balloon apart for a reader that only knows lines across a page."""

    def test_the_ink_is_the_dark_of_a_light_balloon(self):
        found = read.inked(crop_of(column(4), size=(24, 84)))
        self.assertTrue(found[10, 10], "the glyph was not read as ink")
        self.assertFalse(found[0, 0], "the ground was read as ink")

    def test_white_lettering_on_a_dark_balloon_is_the_ink_the_other_way_round(self):
        light = crop_of(column(4), size=(24, 84), ground=DARK, ink=(255, 255, 255))
        found = read.inked(light)
        self.assertTrue(found[10, 10], "the glyph was not read as ink")
        self.assertFalse(found[0, 0], "the ground was read as ink")

    def test_a_run_is_where_the_ink_starts_and_stops(self):
        self.assertEqual(
            read.runs(np.array([False] * 4 + [True] * 5 + [False] * 3)), [(4, 9)]
        )

    def test_a_speck_is_not_a_run(self):
        self.assertEqual(read.runs(np.array([True, False, False, False])), [])

    def test_a_column_of_characters_is_read_as_running_down_the_page(self):
        self.assertTrue(read.upright(read.inked(crop_of(column(5), size=(24, 104)))))

    def test_a_line_of_characters_is_not(self):
        across = lettering(2, 2, 5, 1, EM)
        self.assertFalse(read.upright(read.inked(crop_of(across, size=(104, 24)))))

    def test_a_column_is_set_out_as_a_line_of_the_same_characters(self):
        crop = crop_of(column(5), size=(24, 104))
        line = read.unstacked(crop, read.inked(crop))
        self.assertGreater(line.width, line.height, "it is still a column")
        self.assertGreater(line.width, 4 * crop.width, "the characters ran together")

    def test_a_column_of_one_character_is_left_as_it_is(self):
        crop = crop_of(column(1), size=(24, 24))
        self.assertIs(read.unstacked(crop, read.inked(crop)), crop)

    def test_a_column_set_too_solid_to_cut_is_cut_on_its_own_width(self):
        solid = Image.new("RGB", (20, 100), (255, 255, 255))
        ImageDraw.Draw(solid).rectangle((2, 2, 17, 97), fill=DARK)
        self.assertEqual(len(read.cells(read.inked(solid))), 5)

    def test_columns_are_handed_over_right_to_left(self):
        crop = crop_of(column(5), column(2, at=2), size=(64, 104))
        cut = read.pieces(crop, languages.of("zh"))
        self.assertEqual(len(cut), 2)
        self.assertLess(cut[0].width, cut[1].width, "the left column was read first")

    def test_lines_are_handed_over_top_to_bottom(self):
        short = lettering(2, 2, 2, 1, EM)
        long = lettering(2, 2 + 2 * EM, 5, 1, EM)
        crop = crop_of(short, long, size=(104, 64))
        cut = read.pieces(crop, languages.of("ko"))
        self.assertEqual(len(cut), 2)
        self.assertLess(cut[0].height, crop.height, "the block was not cut at all")

    def test_a_script_that_does_not_stack_keeps_a_tall_block_whole(self):
        crop = crop_of(column(5), size=(24, 104))
        self.assertEqual(len(read.pieces(crop, languages.of("en"))), 1)

    def test_a_crop_with_nothing_in_it_is_still_handed_over(self):
        blank = Image.new("RGB", (40, 40), (255, 255, 255))
        self.assertEqual(read.pieces(blank, languages.of("zh")), [blank])


class TestPpocr(unittest.TestCase):
    """Reading a block a line at a time, and joining what comes back."""

    class Answer:
        def __init__(self, said: str) -> None:
            self.txts = (said,)

    def reader(self, code: str, answers: list[str]) -> read.Ppocr:
        made = read.Ppocr.__new__(read.Ppocr)
        made.language = languages.of(code)
        self.shown: list[np.ndarray] = []
        said = iter(answers)

        def engine(pixels, **kwargs):
            self.shown.append(pixels)
            return self.Answer(next(said))

        made.engine = engine
        return made

    def test_the_columns_of_one_block_are_read_one_at_a_time(self):
        crop = crop_of(column(5), column(4, at=2), size=(64, 104))
        self.assertEqual(self.reader("zh", ["你好", "再见"])(crop), "你好再见")
        self.assertEqual(len(self.shown), 2, "the block went over in one piece")

    def test_a_spaced_language_has_its_lines_joined_with_a_space(self):
        short = lettering(2, 2, 2, 1, EM)
        long = lettering(2, 2 + 2 * EM, 5, 1, EM)
        crop = crop_of(short, long, size=(104, 64))
        self.assertEqual(self.reader("ko", ["안녕", "하세요"])(crop), "안녕 하세요")

    def test_a_line_nothing_was_made_of_is_left_out_rather_than_joined_in(self):
        short = lettering(2, 2, 2, 1, EM)
        long = lettering(2, 2 + 2 * EM, 5, 1, EM)
        crop = crop_of(short, long, size=(104, 64))
        self.assertEqual(self.reader("ko", ["", "hello"])(crop), "hello")

    def test_the_engine_is_handed_the_pixels_the_way_round_it_wants_them(self):
        made = self.reader("zh", ["你"])
        made.line(Image.new("RGB", (40, 20), (200, 210, 220)))
        self.assertEqual(tuple(self.shown[0][0, 0]), (220, 210, 200))


class TestQuieted(unittest.TestCase):
    """Standing a reader up without onnxruntime's hunt for a GPU in the log."""

    HUNT = (
        b'2026-01-01 00:00:00 [W:onnxruntime:Default, device_discovery.cc:285 '
        b'GetGpuDevices] Failed to detect devices under "/sys/class/drm/card0"\n'
    )

    def said(self, write) -> str:
        """Whatever is left on stderr after `write` has run inside quieted()."""
        kept = io.StringIO()
        with mock.patch.object(sys, "stderr", kept):
            with read.quieted():
                write()
        return kept.getvalue()

    def test_the_gpu_hunt_is_dropped(self):
        self.assertEqual(self.said(lambda: os.write(2, self.HUNT)), "")

    def test_everything_else_is_let_through(self):
        trouble = b"[ERROR] Download failed: https://example.invalid/rec.onnx\n"
        said = self.said(lambda: os.write(2, self.HUNT + trouble))
        self.assertNotIn("GetGpuDevices", said)
        self.assertIn("Download failed", said)

    def test_what_was_caught_is_let_out_even_when_the_load_fails(self):
        def write():
            os.write(2, b"halfway through\n")
            raise RuntimeError("no weights")

        kept = io.StringIO()
        with mock.patch.object(sys, "stderr", kept):
            with self.assertRaises(RuntimeError):
                with read.quieted():
                    write()
        self.assertIn("halfway through", kept.getvalue())

    def test_stderr_is_put_back_afterwards(self):
        before = os.fstat(2)
        with read.quieted():
            caught = os.fstat(2)
        after = os.fstat(2)
        self.assertNotEqual(caught.st_ino, before.st_ino, "stderr was not caught")
        self.assertEqual((after.st_dev, after.st_ino), (before.st_dev, before.st_ino))


class TestUnfetched(unittest.TestCase):
    """A reader whose weights are not there and cannot be had."""

    def rapidocr(self, trouble: Exception):
        """A stand-in for the package, whose engine will not build."""
        module = types.ModuleType("rapidocr")
        module.EngineType = types.SimpleNamespace(ONNXRUNTIME="onnxruntime")
        module.ModelType = types.SimpleNamespace(MOBILE="mobile")
        module.LangRec = lambda value: value
        module.OCRVersion = lambda value: value

        def engine(**kwargs):
            raise trouble

        module.RapidOCR = engine
        return mock.patch.dict(sys.modules, {"rapidocr": module})

    def test_it_says_which_language_and_where_the_weights_go(self):
        with self.rapidocr(RuntimeError("Failed to download https://example.invalid")):
            with self.assertRaises(read.Unfetched) as caught:
                read.Ppocr.load(languages.of("zh"))
        said = str(caught.exception)
        self.assertIn("Chinese (simplified)", said)
        self.assertIn(str(read.ppocr_models()), said)
        self.assertIn("Failed to download", said, "what went wrong was lost")


class TestRead(unittest.TestCase):
    """The cropping and the loop. The model itself is never stood up here."""

    def test_a_box_is_given_air_around_it(self):
        padded = read.padded(Box(50, 50, 100, 100), 200, 140)
        self.assertLess(padded.x0, 50)
        self.assertGreater(padded.x1, 100)

    def test_padding_never_runs_off_the_page(self):
        padded = read.padded(Box(150, 100, 200, 140), 200, 140)
        self.assertEqual((padded.x1, padded.y1), (200, 140))
        self.assertEqual(padded, padded.clipped(200, 140))

    def test_each_box_is_read_and_its_answer_stripped(self):
        reader = read.Reader(ocr=lambda image: "  こんにちは  ")
        self.assertEqual(reader(page(), [Box(10, 10, 60, 40)]), ["こんにちは"])

    def test_the_crop_handed_over_is_the_padded_box(self):
        sizes = []
        reader = read.Reader(ocr=lambda image: sizes.append(image.size) or "")
        reader(page(), [Box(50, 50, 100, 100)])
        wanted = read.padded(Box(50, 50, 100, 100), 200, 140)
        self.assertEqual(sizes, [(wanted.w, wanted.h)])

    def test_a_box_too_small_to_hold_lettering_is_not_read(self):
        asked = []
        reader = read.Reader(ocr=lambda image: asked.append(image.size) or "text")
        self.assertEqual(reader(page(), [Box(10, 10, 12, 12)]), [""])
        self.assertEqual(asked, [], "the model was asked about three pixels")

    def test_the_answers_come_back_in_the_order_the_boxes_were_given(self):
        reader = read.Reader(ocr=lambda image: str(image.size[0]))
        texts = reader(page(), [Box(0, 0, 60, 40), Box(0, 0, 20, 20), Box(0, 0, 40, 40)])
        self.assertEqual(len(texts), 3)
        self.assertGreater(int(texts[0]), int(texts[1]))

    def counting(self, stood: list) -> read.Reader:
        """A reader that notes which model it stands up rather than loading one."""

        class Counting(read.Reader):
            def load(self, model=None):
                stood.append("manga-ocr")
                return lambda image: ""

        return Counting()

    def reading(self, stood: list):
        def ppocr(language):
            stood.append(language.code)
            return lambda image: ""

        return mock.patch.object(read, "Ppocr", ppocr)

    def test_a_reader_is_stood_up_once_per_language_and_kept(self):
        stood: list[str] = []
        reader = self.counting(stood)
        with self.reading(stood):
            for code in ("ja", "ja", "ko", "ko", "zh"):
                reader(page(), [Box(10, 10, 60, 40)], languages.of(code))
        self.assertEqual(stood, ["manga-ocr", "ko", "zh"])

    def test_a_request_that_names_no_language_reads_it_as_japanese(self):
        stood: list[str] = []
        reader = self.counting(stood)
        with self.reading(stood):
            reader(page(), [Box(10, 10, 60, 40)])
        self.assertEqual(stood, ["manga-ocr"])

    def test_reading_korean_never_stands_manga_ocr_up(self):
        stood: list[str] = []
        reader = self.counting(stood)
        with self.reading(stood):
            reader(page(), [Box(10, 10, 60, 40)], languages.of("ko"))
        self.assertEqual(stood, ["ko"], "torch was loaded to read a Korean page")


def finding(blocks: list[Block], balloons: list[Box]):
    """A region detector that always answers with these."""

    class Stub:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __call__(self, image, rtl: bool = True):
            return list(blocks), list(balloons)

    return Stub


class TestApi(unittest.TestCase):
    def test_detect_answers_with_the_boxes(self):
        with mock.patch.object(server, "Regions", StubRegions):
            response = client().post("/api/detect", data=payload(stub_balloon()))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["width"], 300)
        [region] = response.json["regions"]
        self.assertEqual(region["box"], [10, 10, 60, 40])
        self.assertEqual(region["confidence"], 0.912)

    def test_detect_answers_with_the_balloon_a_block_is_written_in(self):
        with mock.patch.object(server, "Regions", StubRegions):
            response = client().post("/api/detect", data=payload(stub_balloon()))
        [region] = response.json["regions"]
        self.assertIsNotNone(region["bubble"], "the balloon was not found")
        self.assertGreater(
            region["bubble"][2] - region["bubble"][0],
            60 - 10,
            "the room is no wider than the words",
        )

    def test_detect_says_whether_a_block_is_spoken(self):
        with mock.patch.object(server, "Regions", StubRegions):
            response = client().post("/api/detect", data=payload(stub_balloon()))
        [region] = response.json["regions"]
        self.assertEqual(region["kind"], "speech")

        shout = finding([Block(Box(10, 10, 60, 40), 0.9, FREE)], [])
        with mock.patch.object(server, "Regions", shout):
            response = client().post("/api/detect", data=payload(stub_balloon()))
        [region] = response.json["regions"]
        self.assertEqual(region["kind"], "free")

    def test_detect_leaves_lettering_in_no_balloon_in_its_own_box(self):
        alone = finding([StubRegions.found], [])
        with mock.patch.object(server, "Regions", alone):
            response = client().post("/api/detect", data=payload(stub_balloon()))
        [region] = response.json["regions"]
        self.assertIsNone(region["bubble"])

    def test_bubbles_answers_with_one_balloon_per_box_in_order(self):
        boxes = [[285, 100, 315, 260], [0, 0, 30, 30]]
        drawn = finding([], [Box(120, 60, 480, 300)])
        with mock.patch.object(server, "Regions", drawn):
            response = client().post(
                "/api/bubbles", data=payload(ballooned(), boxes=boxes)
            )
        self.assertEqual(response.status_code, 200)
        first, second = response.json["regions"]
        self.assertEqual(first["box"], boxes[0])
        self.assertGreater(first["bubble"][2] - first["bubble"][0], 100)
        self.assertIsNone(second["bubble"], "the corner of the page is no balloon")

    def test_bubbles_finds_the_balloons_itself(self):
        with mock.patch.object(server, "Regions", side_effect=AssertionError):
            response = client().post(
                "/api/bubbles", data=payload(ballooned(), boxes=[[285, 100, 315, 260]])
            )
        self.assertEqual(response.status_code, 500)

    def test_bubbles_clips_a_box_that_runs_off_the_page(self):
        drawn = finding([], [Box(120, 60, 480, 300)])
        with mock.patch.object(server, "Regions", drawn):
            response = client().post(
                "/api/bubbles", data=payload(ballooned(), boxes=[[285, 100, 900, 900]])
            )
        self.assertEqual(response.json["regions"][0]["box"], [285, 100, 600, 800])

    def test_bubbles_needs_boxes(self):
        response = client().post("/api/bubbles", data=payload(ballooned()))
        self.assertEqual(response.status_code, 400)
        self.assertIn("boxes", response.json["error"])

    def test_detect_needs_an_image(self):
        with mock.patch.object(server, "Regions", StubRegions):
            response = client().post("/api/detect", data={})
        self.assertEqual(response.status_code, 400)
        self.assertIn("image", response.json["error"])

    def test_detect_rejects_something_that_is_not_an_image(self):
        body = {"image": (io.BytesIO(b"not a picture"), "page.png")}
        with mock.patch.object(server, "Regions", StubRegions):
            response = client().post("/api/detect", data=body)
        self.assertEqual(response.status_code, 400)

    def test_letters_answers_with_a_mask_of_the_ink(self):
        with mock.patch.object(server, "Letters", StubLetters):
            response = client().post("/api/letters", data=payload(page()))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "image/png")

        mask = Image.open(io.BytesIO(response.data))
        self.assertEqual(mask.size, (200, 140))
        self.assertEqual(mask.mode, "RGBA", "a canvas wants white on clear")
        alpha = np.array(mask.getchannel("A"))
        self.assertEqual(alpha[25, 35], 255, "the ink is not opaque")
        self.assertEqual(alpha[120, 180], 0, "the page around it is not clear")

    def test_clean_reads_an_opaque_mask_by_its_brightness(self):
        canvas = Image.new("RGBA", (200, 140), (0, 0, 0, 255))
        ImageDraw.Draw(canvas).rectangle((10, 10, 59, 39), fill=(255, 255, 255, 255))

        response = client().post(
            "/api/clean", data=payload(page(), mask=canvas, fill="white")
        )
        self.assertEqual(response.status_code, 200)
        out = opened(response)
        self.assertTrue((patch(out, Box(10, 10, 60, 40)) == 255).all())
        self.assertEqual(
            tuple(np.array(out)[0, 0]), DARK, "the whole page was painted out"
        )

    def test_clean_hides_nothing_for_a_mask_that_is_all_clear(self):
        nothing = Image.new("RGBA", (200, 140), (255, 255, 255, 0))
        response = client().post("/api/clean", data=payload(page(), mask=nothing))
        self.assertEqual(response.status_code, 200)
        self.assertTrue((np.array(opened(response)) == np.array(page())).all())

    def test_clean_reads_a_mask_by_its_transparency(self):
        letters = Image.new("RGBA", (200, 140), (255, 255, 255, 0))
        letters.putalpha(stencil(Box(10, 10, 60, 40)))

        response = client().post(
            "/api/clean", data=payload(page(), mask=letters, fill="white")
        )
        self.assertEqual(response.status_code, 200)
        out = opened(response)
        self.assertTrue((patch(out, Box(10, 10, 60, 40)) == 255).all())
        self.assertEqual(
            tuple(np.array(out)[0, 0]), DARK, "the whole page was painted out"
        )

    def test_read_answers_with_one_text_per_box_in_order(self):
        with mock.patch.object(server, "Reader", StubReader):
            response = client().post(
                "/api/read",
                data=payload(page(), boxes=[[10, 10, 60, 40], [70, 10, 90, 30]]),
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"texts": ["ja 50×30", "ja 20×20"]})

    def test_read_is_read_in_the_language_it_was_sent(self):
        with mock.patch.object(server, "Reader", StubReader):
            response = client().post(
                "/api/read",
                data=payload(page(), boxes=[[10, 10, 60, 40]], language="ko"),
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"texts": ["ko 50×30"]})

    def test_a_language_nothing_here_reads_is_refused(self):
        with mock.patch.object(server, "Reader", StubReader):
            response = client().post(
                "/api/read", data=payload(page(), boxes=[], language="klingon")
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn("language", response.json["error"])

    def test_the_languages_that_can_be_read_are_handed_out(self):
        response = client().get("/api/languages")
        self.assertEqual(response.status_code, 200)
        offered = response.json["languages"]
        self.assertIn({"code": "ja", "name": "Japanese", "rtl": True}, offered)
        self.assertEqual(
            [language["code"] for language in offered], list(languages.CODES)
        )

    def test_read_needs_boxes(self):
        with mock.patch.object(server, "Reader", StubReader):
            response = client().post("/api/read", data=payload(page()))
        self.assertEqual(response.status_code, 400)
        self.assertIn("boxes", response.json["error"])

    def test_read_needs_an_image(self):
        with mock.patch.object(server, "Reader", StubReader):
            response = client().post("/api/read", data={"boxes": "[]"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("image", response.json["error"])

    def test_clean_hides_the_boxes(self):
        response = client().post(
            "/api/clean", data=payload(toned(), boxes=[INK.as_list()])
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "image/png")
        out = opened(response)
        self.assertFalse((patch(out, INK) < 128).any(), "the ink is still there")
        self.assertEqual(tuple(np.array(out)[0, 0]), TONE)

    def test_clean_fills_from_the_art_around_the_mark_unasked(self):
        response = client().post("/api/clean", data=payload(toned(), mask=stencil(INK)))
        self.assertEqual(response.status_code, 200)
        filled = patch(opened(response), INK).astype(int)
        self.assertFalse((filled == 255).any(), "the mark was painted white")
        self.assertTrue((abs(filled - TONE[0]) <= NEAR).all())

    def test_clean_paints_flat_white_when_it_is_asked_to(self):
        response = client().post(
            "/api/clean", data=payload(toned(), mask=stencil(INK), fill="white")
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue((patch(opened(response), INK) == 255).all())

    def test_clean_rejects_a_fill_it_has_never_heard_of(self):
        response = client().post(
            "/api/clean", data=payload(toned(), mask=stencil(INK), fill="beige")
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("fill", response.json["error"])

    def test_clean_clips_a_box_that_runs_off_the_page(self):
        response = client().post(
            "/api/clean", data=payload(page(), boxes=[[-50, -50, 500, 500]])
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue((np.array(opened(response)) == 255).all())

    def test_clean_takes_a_mask_and_boxes_together(self):
        response = client().post(
            "/api/clean",
            data=payload(
                page(),
                mask=stencil(Box(10, 10, 60, 40)),
                boxes=[[100, 100, 140, 130]],
                fill="white",
            ),
        )
        self.assertEqual(response.status_code, 200)
        out = opened(response)
        self.assertTrue((patch(out, Box(10, 10, 60, 40)) == 255).all())
        self.assertTrue((patch(out, Box(100, 100, 140, 130)) == 255).all())

    def test_clean_rejects_a_mask_that_is_not_the_page_size(self):
        response = client().post(
            "/api/clean", data=payload(page(), mask=stencil(size=(50, 50)))
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("mask", response.json["error"])

    def test_clean_needs_boxes_or_a_mask(self):
        response = client().post("/api/clean", data=payload(page()))
        self.assertEqual(response.status_code, 400)
        self.assertIn("boxes", response.json["error"])
        self.assertIn("mask", response.json["error"])

    def test_clean_rejects_boxes_that_are_not_json(self):
        buffer = io.BytesIO()
        page().save(buffer, format="PNG")
        buffer.seek(0)
        response = client().post(
            "/api/clean", data={"image": (buffer, "page.png"), "boxes": "10,20"}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("JSON", response.json["error"])

    def test_models_are_listed(self):
        with mock.patch.object(
            server.ollama, "models", return_value=["gemma4:12b"]
        ):
            response = client().get("/api/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"models": ["gemma4:12b"]})

    def test_models_says_so_when_ollama_is_not_there(self):
        with mock.patch.object(
            server.ollama, "models", side_effect=server.ollama.Unreachable("no ollama")
        ):
            response = client().get("/api/models")
        self.assertEqual(response.status_code, 503)
        self.assertIn("ollama", response.json["error"])

    def test_translate_answers_with_one_text_per_text(self):
        with mock.patch.object(
            server.ollama, "translate", return_value=["Good morning"]
        ) as translating:
            response = client().post(
                "/api/translate",
                data={"texts": json.dumps(["おはよう"]), "model": "gemma4:12b"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"texts": ["Good morning"]})
        self.assertEqual(translating.call_args.args[1], "gemma4:12b")

    def test_translate_takes_the_language_to_translate_into(self):
        with mock.patch.object(
            server.ollama, "translate", return_value=[""]
        ) as into:
            client().post(
                "/api/translate",
                data={"texts": "[]", "model": "m", "target": "Dutch"},
            )
        self.assertEqual(into.call_args.args[2], "Dutch")

    def test_translate_passes_what_each_line_is_on(self):
        with mock.patch.object(
            server.ollama, "translate", return_value=[""]
        ) as told:
            client().post(
                "/api/translate",
                data={
                    "texts": json.dumps(["おはよう", "ドン"]),
                    "model": "m",
                    "kinds": json.dumps(["speech", "free"]),
                    "budgets": json.dumps([28, 12]),
                },
            )
        self.assertEqual(told.call_args.kwargs["kinds"], ["speech", "free"])
        self.assertEqual(told.call_args.kwargs["budgets"], [28, 12])

    def test_translate_without_them_sends_none(self):
        with mock.patch.object(
            server.ollama, "translate", return_value=[""]
        ) as told:
            client().post("/api/translate", data={"texts": "[]", "model": "m"})
        self.assertIsNone(told.call_args.kwargs["kinds"])
        self.assertIsNone(told.call_args.kwargs["budgets"])

    def test_a_block_classified_by_nothing_is_a_real_answer(self):
        with mock.patch.object(
            server.ollama, "translate", return_value=[""]
        ) as told:
            response = client().post(
                "/api/translate",
                data={
                    "texts": json.dumps(["おはよう"]),
                    "model": "m",
                    "kinds": json.dumps([""]),
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(told.call_args.kwargs["kinds"], [""])

    def test_what_is_said_about_the_lines_has_to_line_up_with_them(self):
        response = client().post(
            "/api/translate",
            data={
                "texts": json.dumps(["おはよう", "ドン"]),
                "model": "m",
                "kinds": json.dumps(["speech"]),
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("kinds", response.json["error"])

    def test_a_kind_this_does_not_know_is_refused(self):
        response = client().post(
            "/api/translate",
            data={
                "texts": json.dumps(["おはよう"]),
                "model": "m",
                "kinds": json.dumps(["shouting"]),
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("kinds", response.json["error"])

    def test_the_default_prompt_is_handed_out(self):
        response = client().get("/api/prompt")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"prompt": server.ollama.SYSTEM_DEFAULT})
        self.assertIn("{target}", response.json["prompt"])

    def test_translate_passes_a_prompt_of_your_own_on(self):
        with mock.patch.object(
            server.ollama, "translate", return_value=[""]
        ) as told:
            client().post(
                "/api/translate",
                data={"texts": "[]", "model": "m", "system": "Be brief."},
            )
        self.assertEqual(told.call_args.kwargs["system"], "Be brief.")

    def test_translate_without_a_prompt_leaves_the_default_alone(self):
        with mock.patch.object(
            server.ollama, "translate", return_value=[""]
        ) as told:
            client().post("/api/translate", data={"texts": "[]", "model": "m"})
        self.assertIsNone(told.call_args.kwargs["system"])

    def test_translate_needs_a_model(self):
        response = client().post("/api/translate", data={"texts": "[]"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("model", response.json["error"])

    def test_translate_needs_texts(self):
        response = client().post("/api/translate", data={"model": "m"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("texts", response.json["error"])

    def test_every_answer_may_be_read_from_a_browser(self):
        response = client().post(
            "/api/clean", data=payload(page(), boxes=[[10, 10, 60, 40]])
        )
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], "*")


if __name__ == "__main__":
    unittest.main()
