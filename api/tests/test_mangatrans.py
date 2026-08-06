"""Unit tests. The detector is stubbed, so they need no model and no network."""

from __future__ import annotations

import io
import json
import unittest
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw

from mangatrans import detect, read, render, server
from mangatrans.detect import Block
from mangatrans.geometry import Box

DARK = (20, 20, 20)


def page(width: int = 200, height: int = 140, colour=DARK) -> Image.Image:
    return Image.new("RGB", (width, height), colour)


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
    body.update({name: json.dumps(value) for name, value in fields.items()})
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


class TestCover(unittest.TestCase):
    def test_the_box_is_white_and_the_rest_is_not(self):
        out = render.cover(page(), [Box(10, 10, 60, 40)])
        self.assertTrue((patch(out, Box(10, 10, 60, 40)) == 255).all())
        self.assertEqual(tuple(np.array(out)[0, 0]), DARK)
        self.assertEqual(tuple(np.array(out)[45, 65]), DARK)

    def test_the_far_edge_is_exclusive(self):
        out = np.array(render.cover(page(), [Box(10, 10, 60, 40)]))
        self.assertEqual(tuple(out[39, 59]), (255, 255, 255))
        self.assertEqual(tuple(out[40, 60]), DARK)

    def test_the_original_is_left_alone(self):
        original = page()
        render.cover(original, [Box(0, 0, 200, 140)])
        self.assertEqual(tuple(np.array(original)[0, 0]), DARK)

    def test_an_empty_box_paints_nothing(self):
        out = render.cover(page(), [Box(10, 10, 10, 40)])
        self.assertFalse((np.array(out) == 255).any())


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
                "regions": [{"box": [10, 10, 60, 40], "confidence": 0.912}],
            },
        )

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

        response = client().post("/api/clean", data=payload(page(), mask=canvas))
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

        response = client().post("/api/clean", data=payload(page(), mask=letters))
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
            "/api/clean", data=payload(page(), boxes=[[10, 10, 60, 40]])
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "image/png")
        out = opened(response)
        self.assertTrue((patch(out, Box(10, 10, 60, 40)) == 255).all())
        self.assertEqual(tuple(np.array(out)[0, 0]), DARK)

    def test_clean_clips_a_box_that_runs_off_the_page(self):
        response = client().post(
            "/api/clean", data=payload(page(), boxes=[[-50, -50, 500, 500]])
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue((np.array(opened(response)) == 255).all())

    def test_clean_hides_what_the_mask_marks(self):
        response = client().post(
            "/api/clean", data=payload(page(), mask=stencil(Box(10, 10, 60, 40)))
        )
        self.assertEqual(response.status_code, 200)
        out = opened(response)
        self.assertTrue((patch(out, Box(10, 10, 60, 40)) == 255).all())
        self.assertEqual(tuple(np.array(out)[0, 0]), DARK)

    def test_clean_takes_a_mask_and_boxes_together(self):
        response = client().post(
            "/api/clean",
            data=payload(
                page(), mask=stencil(Box(10, 10, 60, 40)), boxes=[[100, 100, 140, 130]]
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

    def test_render_rejects_a_region_without_a_box(self):
        response = client().post(
            "/api/render", data=payload(page(), regions=[{"text": "HELLO"}])
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("box", response.json["error"])

    def test_every_answer_may_be_read_from_a_browser(self):
        response = client().post(
            "/api/clean", data=payload(page(), boxes=[[10, 10, 60, 40]])
        )
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], "*")


if __name__ == "__main__":
    unittest.main()
