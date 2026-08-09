"""Unit tests. The detector is stubbed, so they need no model and no network."""

from __future__ import annotations

import io
import json
import unittest
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw

from mangatrans import bubble, detect, inpaint, ollama, read, render, server
from mangatrans.detect import Block
from mangatrans.geometry import Box

DARK = (20, 20, 20)
# A page of flat tone, and something dark drawn on it to be taken back off: what
# a fill made out of the surroundings should come back as, and what it should not.
TONE = (200, 200, 200)
INK = Box(80, 50, 120, 90)
# Filling from a flat surround lands on that flat value, give or take rounding.
NEAR = 5


def page(width: int = 200, height: int = 140, colour=DARK) -> Image.Image:
    return Image.new("RGB", (width, height), colour)


def toned(ink: Box | None = INK, rim: int = 0) -> Image.Image:
    """Flat tone with ink on it, and a rim of half-ink around it if asked.

    The rim is what a soft-edged letter leaves just outside a mask that stops at
    the ink — the thing a fill must not be made out of.
    """
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
    """A multipart body: the image, a mask if there is one, and JSON beside them.

    Strings go up as they are — a word the API reads back with request.form.get
    is not JSON, and would arrive still wearing its quotes.
    """
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


class StubDetector:
    """One block, and some ink inside it, whatever the page."""

    found = Block(Box(10, 10, 60, 40), 0.912)
    ink = Box(20, 15, 50, 35)

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, image):
        assert image.ndim == 3 and image.shape[2] == 3, "expects an RGB array"
        return [self.found]

    def letters(self, image, grow=2):
        assert image.ndim == 3 and image.shape[2] == 3, "expects an RGB array"
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[self.ink.y0 : self.ink.y1, self.ink.x0 : self.ink.x1] = 255
        return mask


class StubReader:
    """The size of whatever it was handed, so the crop can be checked."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, image, boxes):
        return [f"{box.w}×{box.h}" for box in boxes]


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
        # Half-ink just outside the mask is the letter, not the art: read as art
        # it would be carried inwards and the letter put back as a smudge.
        filled = patch(inpaint.fill(toned(rim=2), stencil(INK)), INK).astype(int)
        self.assertTrue((abs(filled - TONE[0]) <= NEAR).all(), "the rim was read")

    def test_the_rim_is_still_left_where_it_was(self):
        # Kept out of what the fill is made of is not the same as painted over:
        # nothing outside the mark is touched.
        out = np.array(inpaint.fill(toned(rim=2), stencil(INK)))
        self.assertEqual(tuple(out[INK.y0 - 1, INK.x0]), (100, 100, 100))

    def test_a_page_marked_all_over_has_nothing_left_to_look_at(self):
        out = inpaint.fill(toned(), stencil(Box(0, 0, 200, 140)))
        self.assertTrue((np.array(out) == 255).all())

    def test_the_original_is_left_alone(self):
        original = toned()
        inpaint.fill(original, stencil(Box(0, 0, 200, 140)))
        self.assertEqual(tuple(np.array(original)[0, 0]), TONE)


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
            scale = detect.INPUT_SIZE / 200  # the page's long side fills the square
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
        # Everything the model saw is lit, so every page pixel should be, and
        # nothing should be lost to the black bars letterbox() added.
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


class TestOverlay(unittest.TestCase):
    box = Box(20, 20, 180, 120)

    def rendered(self, text: str) -> np.ndarray:
        out = render.overlay(page(), [render.Region(self.box, text)])
        return patch(out, self.box)

    def test_the_text_is_drawn_dark_on_the_white(self):
        inside = self.rendered("HELLO THERE")
        self.assertTrue((inside < 128).any(), "no lettering was drawn")
        self.assertTrue((inside == 255).any(), "the box was not whited out")

    def test_a_region_with_no_text_is_only_hidden(self):
        self.assertTrue((self.rendered("   ") == 255).all())

    def test_long_text_is_wrapped_onto_several_lines(self):
        font = render.load_font(None, 20)
        lines = render.wrap("one two three four five six seven", font, 60)
        self.assertGreater(len(lines), 1)
        self.assertTrue(all(line for line in lines))

    def test_more_words_are_set_smaller(self):
        draw = ImageDraw.Draw(page())
        short = render.fit(draw, "HI", self.box, None)
        long = render.fit(draw, "HI " * 40, self.box, None)
        self.assertLess(long.font.size, short.font.size)

    def test_text_that_cannot_fit_is_still_drawn(self):
        draw = ImageDraw.Draw(page())
        layout = render.fit(draw, "WAY TOO MANY WORDS " * 30, Box(0, 0, 24, 12), None)
        self.assertFalse(layout.fits)
        self.assertEqual(layout.font.size, render.FONT_MIN)
        self.assertTrue((self.rendered("WAY TOO MANY WORDS " * 30) < 128).any())


def ballooned(
    balloon: Box = Box(120, 60, 480, 300),
    column: Box = Box(285, 100, 315, 260),
    size=(600, 800),
    ground=(255, 255, 255),
    ink=(0, 0, 0),
) -> Image.Image:
    """A page of artwork with one balloon on it, and a column of writing inside.

    The column is what a detector hands back for Japanese: tall, narrow, and
    down the middle, which cuts the balloon's ground in two. The artwork around
    it is mid tone, so nothing outside the balloon is light enough to be mistaken
    for part of it.
    """
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


def stub_balloon(size=(300, 200)) -> Image.Image:
    """A page with a balloon drawn around the block :class:`StubDetector` finds.

    Rounded rather than oval, because that block is nearly square and nearly
    fills it: the largest rectangle inside an oval drawn round a square is
    smaller than the square, which is an answer /api/bubbles is right to
    withhold and not what this is here to show.
    """
    image = Image.new("RGB", size, (120, 120, 120))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (2, 2, 96, 66), radius=12, fill=(255, 255, 255), outline=(0, 0, 0), width=3
    )
    ink = StubDetector.ink
    draw.rectangle((ink.x0, ink.y0, ink.x1 - 1, ink.y1 - 1), fill=(0, 0, 0))
    return image


def grey(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("L"))


class TestUnder(unittest.TestCase):
    def test_one_flat_row_is_as_wide_as_it_is(self):
        self.assertEqual(bubble.under([2, 2, 2]), (6, 0, 3, 2))

    def test_a_dip_is_worth_more_wide_than_tall(self):
        area, x0, x1, height = bubble.under([2, 1, 2])
        self.assertEqual((area, x0, x1, height), (3, 0, 3, 1))

    def test_a_gap_ends_the_rectangle(self):
        self.assertEqual(bubble.under([3, 0, 1]), (3, 0, 1, 3))

    def test_nothing_standing_is_no_rectangle(self):
        self.assertEqual(bubble.under([0, 0]), (0, 0, 0, 0))


class TestStanding(unittest.TestCase):
    def test_the_rectangle_is_where_the_pixels_are(self):
        mask = np.zeros((20, 30), np.uint8)
        mask[4:12, 5:25] = 255
        self.assertEqual(bubble.standing(mask > 0), Box(5, 4, 25, 12))

    def test_an_empty_mask_has_no_rectangle(self):
        self.assertIsNone(bubble.standing(np.zeros((10, 10), np.uint8) > 0))

    def test_the_larger_of_two_shapes_wins(self):
        mask = np.zeros((40, 40), np.uint8)
        mask[0:4, 0:4] = 255
        mask[10:30, 10:35] = 255
        self.assertEqual(bubble.standing(mask > 0), Box(10, 10, 35, 30))


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


class TestAround(unittest.TestCase):
    balloon = Box(120, 60, 480, 300)
    column = Box(285, 100, 315, 260)

    def found(self, image: Image.Image, box: Box | None = None) -> Box | None:
        return bubble.around(grey(image), box or self.column)

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
        self.assertLess(found.x0, self.column.cx)
        self.assertGreater(found.x1, self.column.cx)
        self.assertLess(found.y0, self.column.cy)
        self.assertGreater(found.y1, self.column.cy)

    def test_the_column_is_not_measured_as_the_gap_beside_it(self):
        # Without the block painted in, the flood starts in one half of a
        # balloon a line of Japanese has cut in two, and answers with that half.
        found = self.found(ballooned())
        assert found is not None
        self.assertGreater(found.x1 - found.x0, self.balloon.w / 2)

    def test_white_words_on_a_dark_balloon_are_found_the_same_way(self):
        image = ballooned(ground=(15, 15, 15), ink=(255, 255, 255))
        found = self.found(image)
        assert found is not None
        self.assertGreater(found.w, self.column.w * 3)

    def test_lettering_with_no_balloon_around_it_has_no_answer(self):
        # A page of flat white with writing on it: the flood has the whole page
        # to spread over, which is not a balloon however it is measured.
        image = Image.new("RGB", (600, 800), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        for y in range(self.column.y0, self.column.y1, 30):
            draw.rectangle(
                (self.column.x0, y, self.column.x1 - 1, y + 20), fill=(0, 0, 0)
            )
        self.assertIsNone(self.found(image))

    def test_a_balloon_no_wider_than_the_words_is_left_alone(self):
        # There is nothing to be won by moving a line into a balloon that is
        # already drawn tight around it.
        wide = Box(40, 170, 560, 230)
        image = ballooned(balloon=wide, column=Box(60, 180, 540, 220))
        self.assertIsNone(self.found(image, Box(60, 180, 540, 220)))

    def test_a_box_too_small_to_hold_lettering_has_no_answer(self):
        self.assertIsNone(self.found(ballooned(), Box(300, 180, 302, 182)))

    def test_a_box_off_the_page_has_no_answer(self):
        self.assertIsNone(self.found(ballooned(), Box(900, 900, 1000, 1000)))


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
        # Some builds file a schema-held answer under 'thinking' instead.
        with mock.patch.object(
            ollama, "ask", return_value=reply("", translations("Good morning"))
        ):
            self.assertEqual(ollama.translate(["おはよう"], "m"), ["Good morning"])

    def test_a_fenced_answer_is_still_read(self):
        fenced = f"```json\n{translations('Good morning')}\n```"
        with mock.patch.object(ollama, "ask", return_value=reply(fenced)):
            self.assertEqual(ollama.translate(["おはよう"], "m"), ["Good morning"])

    def test_losing_count_falls_back_to_one_line_at_a_time(self):
        answers = [
            reply(translations("only one")),  # two were asked about
            reply(translations("first")),
            reply(translations("second")),
        ]
        with mock.patch.object(ollama, "ask", side_effect=answers) as ask:
            got = ollama.translate(["いち", "に"], "m")
        self.assertEqual(got, ["first", "second"])
        self.assertEqual(ask.call_count, 3, "it did not ask again line by line")

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

    def test_a_briefing_of_your_own_is_used_instead(self):
        said = ollama.briefing("Dutch", "Turn this into {target}, in pirate.")
        self.assertEqual(said, "Turn this into Dutch, in pirate.")

    def test_a_briefing_may_have_braces_of_its_own(self):
        # Substituted rather than formatted: str.format would choke on these.
        said = ollama.briefing("Dutch", 'Answer like {"a": 1} but in {target}.')
        self.assertEqual(said, 'Answer like {"a": 1} but in Dutch.')

    def test_the_briefing_reaches_the_request(self):
        asked = []

        def ask(path, body=None, **kwargs):
            asked.append(body["messages"][0]["content"])
            return reply(translations("Goedemorgen"))

        with mock.patch.object(ollama, "ask", ask):
            ollama.translate(["おはよう"], "m", "Dutch", system="Be brief in {target}.")
        self.assertEqual(asked, ["Be brief in Dutch."])

    def test_the_briefing_holds_when_it_falls_back_to_one_at_a_time(self):
        asked = []

        def ask(path, body=None, **kwargs):
            asked.append(body["messages"][0]["content"])
            # Miscounted the first time, so each line is asked about alone.
            return reply(translations("only one")) if len(asked) == 1 else reply(
                translations("a line")
            )

        with mock.patch.object(ollama, "ask", ask):
            ollama.translate(["いち", "に"], "m", "Dutch", system="Mine.")
        self.assertEqual(asked, ["Mine.", "Mine.", "Mine."])

    def test_where_ollama_is_can_be_said(self):
        self.assertEqual(ollama.base("http://elsewhere:11434/"), "http://elsewhere:11434")


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


class TestApi(unittest.TestCase):
    def test_detect_answers_with_the_boxes(self):
        with mock.patch.object(server, "Detector", StubDetector):
            response = client().post("/api/detect", data=payload(page()))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json,
            {
                "width": 200,
                "height": 140,
                "regions": [
                    # Flat tone from edge to edge: there is no balloon to find,
                    # and saying so is the answer that leaves the box alone.
                    {"box": [10, 10, 60, 40], "confidence": 0.912, "bubble": None}
                ],
            },
        )

    def test_detect_answers_with_the_balloon_a_block_is_written_in(self):
        with mock.patch.object(server, "Detector", StubDetector):
            response = client().post("/api/detect", data=payload(stub_balloon()))
        [region] = response.json["regions"]
        self.assertIsNotNone(region["bubble"], "the balloon was not found")

    def test_bubbles_answers_with_one_balloon_per_box_in_order(self):
        boxes = [[285, 100, 315, 260], [0, 0, 30, 30]]
        response = client().post(
            "/api/bubbles", data=payload(ballooned(), boxes=boxes)
        )
        self.assertEqual(response.status_code, 200)
        first, second = response.json["regions"]
        self.assertEqual(first["box"], boxes[0])
        self.assertGreater(first["bubble"][2] - first["bubble"][0], 100)
        self.assertIsNone(second["bubble"], "the corner of the page is no balloon")

    def test_bubbles_stands_no_model_up(self):
        # Nothing here needs the detector, and a caller that only wants balloons
        # should not pay ~95 MB to find that out.
        with mock.patch.object(server, "Detector", side_effect=AssertionError):
            response = client().post(
                "/api/bubbles", data=payload(ballooned(), boxes=[[285, 100, 315, 260]])
            )
        self.assertEqual(response.status_code, 200)

    def test_bubbles_clips_a_box_that_runs_off_the_page(self):
        response = client().post(
            "/api/bubbles", data=payload(ballooned(), boxes=[[285, 100, 900, 900]])
        )
        self.assertEqual(response.json["regions"][0]["box"], [285, 100, 600, 800])

    def test_bubbles_needs_boxes(self):
        response = client().post("/api/bubbles", data=payload(ballooned()))
        self.assertEqual(response.status_code, 400)
        self.assertIn("boxes", response.json["error"])

    def test_detect_needs_an_image(self):
        with mock.patch.object(server, "Detector", StubDetector):
            response = client().post("/api/detect", data={})
        self.assertEqual(response.status_code, 400)
        self.assertIn("image", response.json["error"])

    def test_detect_rejects_something_that_is_not_an_image(self):
        body = {"image": (io.BytesIO(b"not a picture"), "page.png")}
        with mock.patch.object(server, "Detector", StubDetector):
            response = client().post("/api/detect", data=body)
        self.assertEqual(response.status_code, 400)

    def test_letters_answers_with_a_mask_of_the_ink(self):
        with mock.patch.object(server, "Detector", StubDetector):
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
        # What a browser canvas exports: black, white where it was marked, and
        # opaque from edge to edge, because a canvas always carries an alpha
        # channel whether or not anything was made see-through. Going by that
        # alpha would hide the whole page.
        canvas = Image.new("RGBA", (200, 140), (0, 0, 0, 255))
        ImageDraw.Draw(canvas).rectangle((10, 10, 59, 39), fill=(255, 255, 255, 255))

        # Flat white, so what came back says plainly what was read as marked.
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
        # What /api/letters hands back is white everywhere and only transparent
        # where the page should be left alone. Read by brightness it would say
        # "hide all of it", so /api/clean has to go by the alpha channel.
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
        self.assertEqual(response.json, {"texts": ["50×30", "20×20"]})

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
        # And a page marked from edge to edge has nothing left to make a fill
        # out of, so white is all that can be said.
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

    def test_render_sets_the_text_in_the_box(self):
        response = client().post(
            "/api/render",
            data=payload(
                page(), regions=[{"box": [20, 20, 180, 120], "text": "HELLO THERE"}]
            ),
        )
        self.assertEqual(response.status_code, 200)
        inside = patch(opened(response), Box(20, 20, 180, 120))
        self.assertTrue((inside < 128).any())
        self.assertTrue((inside == 255).any())

    def test_render_gives_the_new_text_a_clear_ground_unasked(self):
        # The other way round from /api/clean: black lettering is set into the
        # box here, and it wants white under it rather than whatever was there.
        response = client().post(
            "/api/render",
            data=payload(toned(), regions=[{"box": INK.as_list(), "text": "HI"}]),
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue((patch(opened(response), INK) == 255).any())

    def test_render_can_fill_from_the_art_when_asked(self):
        response = client().post(
            "/api/render",
            data=payload(
                toned(), regions=[{"box": INK.as_list(), "text": "HI"}], fill="art"
            ),
        )
        self.assertEqual(response.status_code, 200)
        inside = patch(opened(response), INK)
        self.assertFalse((inside == 255).any(), "the box was whited out")
        self.assertTrue((inside < 128).any(), "no lettering was drawn")

    def test_render_rejects_a_region_without_a_box(self):
        response = client().post(
            "/api/render", data=payload(page(), regions=[{"text": "HELLO"}])
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("box", response.json["error"])

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
        with mock.patch.object(server.ollama, "translate", return_value=[""]) as into:
            client().post(
                "/api/translate",
                data={"texts": "[]", "model": "m", "target": "Dutch"},
            )
        self.assertEqual(into.call_args.args[2], "Dutch")

    def test_the_default_prompt_is_handed_out(self):
        response = client().get("/api/prompt")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"prompt": server.ollama.SYSTEM_DEFAULT})
        self.assertIn("{target}", response.json["prompt"])

    def test_translate_passes_a_prompt_of_your_own_on(self):
        with mock.patch.object(server.ollama, "translate", return_value=[""]) as told:
            client().post(
                "/api/translate",
                data={"texts": "[]", "model": "m", "system": "Be brief."},
            )
        self.assertEqual(told.call_args.kwargs["system"], "Be brief.")

    def test_translate_without_a_prompt_leaves_the_default_alone(self):
        with mock.patch.object(server.ollama, "translate", return_value=[""]) as told:
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
