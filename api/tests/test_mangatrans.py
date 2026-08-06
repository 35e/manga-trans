"""Unit tests. The detector is stubbed, so they need no model and no network."""

from __future__ import annotations

import io
import json
import unittest
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw

from mangatrans import render, server
from mangatrans.detect import Block
from mangatrans.geometry import Box

DARK = (20, 20, 20)


def page(width: int = 200, height: int = 140, colour=DARK) -> Image.Image:
    return Image.new("RGB", (width, height), colour)


def payload(image: Image.Image, **fields) -> dict:
    """A multipart body: the image, and any JSON fields sent beside it."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    body = {"image": (buffer, "page.png")}
    body.update({name: json.dumps(value) for name, value in fields.items()})
    return body


def opened(response) -> Image.Image:
    return Image.open(io.BytesIO(response.data)).convert("RGB")


def patch(pixels, box: Box) -> np.ndarray:
    return np.array(pixels)[box.y0 : box.y1, box.x0 : box.x1]


class StubDetector:
    """One block, whatever the page."""

    found = Block(Box(10, 10, 60, 40), 0.912)

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, image):
        assert image.ndim == 3 and image.shape[2] == 3, "expects an RGB array"
        return [self.found]


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

    def test_clean_needs_boxes(self):
        response = client().post("/api/clean", data=payload(page()))
        self.assertEqual(response.status_code, 400)
        self.assertIn("boxes", response.json["error"])

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
