"""Unit tests. No models needed: the detector is stubbed where it is involved."""

import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw

from mangatrans import ocr, render, translate
from mangatrans.detect import Block, Detection
from mangatrans.geometry import Box
from mangatrans.server import Backend, Pages, create_app


def page_with_text(size=(200, 120), box=(40, 30, 160, 90)):
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline="black", width=2)
    for x in range(box[0] + 10, box[2] - 10, 14):
        draw.rectangle((x, box[1] + 12, x + 6, box[3] - 12), fill="black")
    return image


class TestBox(unittest.TestCase):
    def test_from_list_normalises(self):
        self.assertEqual(Box.from_list([30, 40, 10, 20]), Box(10, 20, 30, 40))

    def test_from_list_rounds_floats(self):
        self.assertEqual(Box.from_list([1.4, 2.6, 10.0, 20.0]), Box(1, 3, 10, 20))

    def test_clipped_and_padded_stay_inside(self):
        self.assertEqual(Box(-5, -5, 500, 500).clipped(100, 80), Box(0, 0, 100, 80))
        self.assertEqual(Box(10, 10, 20, 20).padded(30, 100, 80), Box(0, 0, 50, 50))

    def test_size(self):
        box = Box(10, 20, 40, 60)
        self.assertEqual((box.w, box.h, box.area), (30, 40, 1200))


class TestFitting(unittest.TestCase):
    def setUp(self):
        self.draw = ImageDraw.Draw(Image.new("RGB", (400, 400)))

    def test_wrap_breaks_lines_to_width(self):
        font = render.load_font(None, 20)
        lines, whole = render.wrap("one two three four five six", font, 120)
        self.assertEqual(" ".join(lines), "one two three four five six")
        self.assertGreater(len(lines), 1)
        self.assertTrue(all(font.getlength(line) <= 120 for line in lines))
        self.assertTrue(whole)

    def test_a_word_too_long_for_the_width_is_broken(self):
        font = render.load_font(None, 20)
        lines, whole = render.wrap("supercalifragilistic", font, 60)
        self.assertGreater(len(lines), 1)
        self.assertTrue(all(font.getlength(line) <= 60 for line in lines))
        self.assertEqual("".join(lines).replace("-", ""), "supercalifragilistic")
        self.assertFalse(whole)

    def test_fit_stays_within_the_box(self):
        box = Box(0, 0, 160, 90)
        layout = render.fit(self.draw, "hello there friend", box, None)
        left, top, right, bottom = render.measure(self.draw, layout)
        self.assertTrue(layout.fits)
        self.assertLessEqual(right - left, box.w)
        self.assertLessEqual(bottom - top, box.h)

    def test_a_bigger_box_gets_bigger_type(self):
        small = render.fit(self.draw, "hello there", Box(0, 0, 60, 40), None)
        large = render.fit(self.draw, "hello there", Box(0, 0, 240, 160), None)
        self.assertGreater(large.font.size, small.font.size)

    def test_a_word_is_only_broken_when_no_size_keeps_it_whole(self):
        roomy = render.fit(self.draw, "you understand", Box(0, 0, 100, 400), None)
        self.assertNotIn("-", roomy.block)
        self.assertTrue(roomy.whole)
        narrow = render.fit(self.draw, "you understand", Box(0, 0, 40, 90), None)
        self.assertIn("-", narrow.block)
        self.assertTrue(narrow.fits)
        self.assertFalse(narrow.whole)

    def test_punctuation_does_not_get_a_line_of_its_own(self):
        font = render.load_font(None, 20)
        lines, _ = render.wrap("unbelievable, really", font, 60)
        self.assertFalse(any(line.lstrip().startswith(",") for line in lines))

    def test_a_box_too_tight_for_the_words_still_gets_them(self):
        """The bug this fixes: nothing was drawn at all, so the bubble was wiped."""
        layout = render.fit(self.draw, "I can't believe you", Box(0, 0, 24, 40), None)
        self.assertFalse(layout.fits)
        self.assertEqual(layout.font.size, render.FONT_MIN)
        self.assertTrue(layout.block.strip())


class TestCover(unittest.TestCase):
    def test_ink_is_the_minority_tone_without_a_mask(self):
        grey = np.full((40, 40), 255, np.uint8)
        grey[10:20, 10:20] = 0
        ink = render.ink_of(grey, None, Box(0, 0, 40, 40))
        self.assertTrue(ink[15, 15])
        self.assertFalse(ink[35, 35])

    def test_the_detector_mask_wins_when_it_has_one(self):
        grey = np.full((40, 40), 255, np.uint8)
        mask = np.zeros((40, 40), np.uint8)
        mask[5:9, 5:9] = 255
        ink = render.ink_of(grey, mask, Box(0, 0, 40, 40))
        self.assertTrue(ink[6, 6])
        self.assertEqual(int(ink.sum()), 16)

    def test_dark_text_is_covered_with_the_paper_under_it(self):
        image = page_with_text()
        pixels = np.array(image)
        grey = np.array(image.convert("L"))
        box = Box(45, 35, 155, 85)
        render.cover(pixels, grey, None, box)
        inside = pixels[box.y0 : box.y1, box.x0 : box.x1]
        self.assertGreater(inside.min(), 200)

    def test_light_text_on_a_dark_plate_is_covered_too(self):
        image = Image.new("RGB", (200, 120), "black")
        draw = ImageDraw.Draw(image)
        for x in range(50, 150, 14):
            draw.rectangle((x, 42, x + 6, 78), fill="white")
        pixels = np.array(image)
        box = Box(40, 35, 160, 85)
        render.cover(pixels, np.array(image.convert("L")), None, box)
        # Only the dark-plate branch fills the box back in with black.
        self.assertLess(pixels[box.y0 : box.y1, box.x0 : box.x1].max(), 60)

    def test_nothing_outside_the_box_is_touched(self):
        image = page_with_text()
        before = np.array(image)
        pixels = before.copy()
        box = Box(45, 35, 155, 85)
        render.cover(pixels, np.array(image.convert("L")), None, box)
        outside = np.ones(pixels.shape[:2], bool)
        outside[box.y0 : box.y1, box.x0 : box.x1] = False
        self.assertTrue((pixels[outside] == before[outside]).all())


class TestOverlay(unittest.TestCase):
    @staticmethod
    def ink(image, box):
        return (np.array(image)[box.y0 : box.y1, box.x0 : box.x1] < 100).sum()

    def test_the_original_text_goes_and_the_translation_arrives(self):
        image = page_with_text()
        box = Box(45, 35, 155, 85)
        out = render.overlay(image, None, [render.Region(box, "hello there")])
        self.assertGreater(self.ink(out.image, box), 0)  # lettered
        self.assertLess(self.ink(out.image, box), self.ink(image, box))
        self.assertEqual((out.blank, out.overflow), ([], []))

    def test_a_region_with_no_text_is_only_cleaned(self):
        image = page_with_text()
        box = Box(45, 35, 155, 85)
        out = render.overlay(image, None, [render.Region(box, "  ")])
        self.assertGreater(
            np.array(out.image)[box.y0 : box.y1, box.x0 : box.x1].min(), 200
        )
        self.assertEqual(out.blank, [0])

    def test_a_box_too_tight_is_lettered_anyway_and_reported(self):
        image = page_with_text()
        box = Box(60, 40, 90, 80)
        out = render.overlay(image, None, [render.Region(box, "I can't believe you")])
        self.assertGreater(self.ink(out.image, box), 0)
        self.assertEqual(out.overflow, [0])

    def test_a_box_only_fitted_by_hyphenating_says_so(self):
        image = page_with_text()
        box = Box(45, 35, 85, 85)
        out = render.overlay(image, None, [render.Region(box, "you understand")])
        self.assertEqual((out.overflow, out.tight), ([], [0]))

    def test_the_text_can_be_set_somewhere_else_than_it_was_cleaned(self):
        image = page_with_text()
        box = Box(45, 35, 155, 85)
        elsewhere = Box(10, 92, 190, 118)
        out = render.overlay(image, None, [render.Region(box, "hi", elsewhere)])
        self.assertGreater(self.ink(out.image, elsewhere), 0)
        self.assertEqual(self.ink(out.image, box), 0)  # cleaned, and left clean

    def test_light_type_is_used_on_a_plate_left_dark(self):
        image = Image.new("RGB", (200, 120), "black")
        draw = ImageDraw.Draw(image)
        for x in range(50, 150, 14):
            draw.rectangle((x, 42, x + 6, 78), fill="white")
        box = Box(40, 35, 160, 85)
        out = render.overlay(image, None, [render.Region(box, "hello")])
        patch = np.array(out.image)[box.y0 : box.y1, box.x0 : box.x1]
        self.assertGreater((patch > 200).sum(), 0)

    def test_regions_outside_the_page_are_ignored(self):
        image = page_with_text()
        out = render.overlay(image, None, [render.Region(Box(500, 500, 600, 600), "no")])
        self.assertEqual(out.image.size, image.size)
        self.assertEqual(out.blank, [0])


class TestTranslate(unittest.TestCase):
    def test_extract_json_digs_the_object_out_of_prose(self):
        found = translate.extract_json('sure!\n```json\n{"translations": []}\n```')
        self.assertEqual(found, {"translations": []})

    def test_bad_json_is_reported(self):
        with self.assertRaises(translate.OllamaError):
            translate.extract_json("no json here")

    def test_translations_line_up_with_their_input(self):
        reply = '{"translations": [{"id": 1, "text": "1. Hi"}, {"id": 3, "text": "Bye"}]}'
        with mock.patch.object(translate, "chat", return_value=reply):
            out = translate.translate(["こんにちは", "", "さようなら"])
        self.assertEqual(out, ["Hi", "", "Bye"])

    def test_a_skipped_line_is_asked_for_again(self):
        replies = [
            '{"translations": [{"id": 1, "text": "Hi"}]}',
            '{"translations": [{"id": 1, "text": "Bye"}]}',
        ]
        with mock.patch.object(translate, "chat", side_effect=replies) as chat:
            out = translate.translate(["こんにちは", "さようなら"])
        self.assertEqual(out, ["Hi", "Bye"])
        self.assertEqual(chat.call_count, 2)

    def test_nothing_to_translate_needs_no_model(self):
        with mock.patch.object(translate, "chat", side_effect=AssertionError):
            self.assertEqual(translate.translate(["", "  "]), ["", ""])


class TestServer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        root = tmp / "pages"
        root.mkdir()
        page_with_text().save(root / "001.png")
        self.backend = Backend(Pages(root, tmp / "out"))
        self.backend._detector = lambda image: Detection(
            blocks=[Block(box=Box(45, 35, 155, 85), confidence=0.9)],
            mask=None,
        )
        self.client = create_app(self.backend).test_client()

    def tearDown(self):
        self.tmp.cleanup()

    def test_pages_are_listed(self):
        self.assertEqual(self.client.get("/api/pages").json["pages"], ["001.png"])

    def test_detect_reads_every_block(self):
        with mock.patch.object(ocr, "read", return_value="こんにちは"):
            body = self.client.post("/api/detect", json={"page": "001.png"}).json
        self.assertEqual(body["width"], 200)
        self.assertEqual(len(body["regions"]), 1)
        self.assertEqual(body["regions"][0]["text"], "こんにちは")
        self.assertIsNone(body["warning"])

    def test_detect_survives_manga_ocr_missing(self):
        with mock.patch.object(ocr, "read", side_effect=ocr.OcrUnavailable("nope")):
            body = self.client.post("/api/detect", json={"page": "001.png"}).json
        self.assertEqual(body["warning"], "nope")
        self.assertEqual(body["regions"][0]["text"], "")

    def test_a_resized_region_is_read_again(self):
        with mock.patch.object(ocr, "read", return_value="ありがとう") as read:
            body = self.client.post(
                "/api/read", json={"page": "001.png", "box": [10, 10, 190, 110]}
            ).json
        self.assertEqual(body["text"], "ありがとう")
        self.assertEqual(read.call_args.args[1], Box(10, 10, 190, 110))

    def test_a_region_is_clipped_to_the_page(self):
        with mock.patch.object(ocr, "read", return_value="") as read:
            self.client.post(
                "/api/read", json={"page": "001.png", "box": [-50, -50, 9999, 9999]}
            )
        self.assertEqual(read.call_args.args[1], Box(0, 0, 200, 120))

    def test_render_writes_the_overlaid_page(self):
        body = self.client.post(
            "/api/render",
            json={"page": "001.png", "regions": [{"box": [45, 35, 155, 85], "text": "hi"}]},
        ).json
        self.assertEqual(body["output"], "001.png")
        self.assertEqual((body["blank"], body["overflow"]), ([], []))
        self.assertTrue((self.backend.pages.out / "001.png").is_file())
        response = self.client.get("/api/output/001.png")
        self.assertEqual(response.status_code, 200)
        response.close()

    def test_render_sets_the_text_where_the_text_box_says(self):
        with mock.patch.object(render, "overlay", wraps=render.overlay) as overlay:
            self.client.post(
                "/api/render",
                json={
                    "page": "001.png",
                    "regions": [
                        {
                            "box": [45, 35, 155, 85],
                            "text_box": [10, 90, 190, 118],
                            "text": "hi",
                        }
                    ],
                },
            )
        region = overlay.call_args.args[2][0]
        self.assertEqual(region.box, Box(45, 35, 155, 85))
        self.assertEqual(region.text_box, Box(10, 90, 190, 118))
        self.assertEqual(region.where, Box(10, 90, 190, 118))

    def test_a_region_covered_with_nothing_in_it_is_reported(self):
        body = self.client.post(
            "/api/render",
            json={"page": "001.png", "regions": [{"box": [45, 35, 155, 85], "text": ""}]},
        ).json
        self.assertEqual(body["blank"], [0])

    def test_the_browser_can_fetch_the_font_the_page_is_set_in(self):
        response = self.client.get("/api/font")
        self.assertEqual(response.status_code, 200 if render.font_file() else 404)
        response.close()

    def test_pages_outside_the_folder_are_refused(self):
        for name in ("../../etc/passwd", "..%2f..%2fetc%2fpasswd"):
            self.assertEqual(
                self.client.post("/api/detect", json={"page": name}).status_code, 404
            )

    def test_a_missing_page_is_a_404(self):
        self.assertEqual(self.client.get("/api/image/nope.png").status_code, 404)

    def test_a_japanese_page_name_still_opens(self):
        page_with_text().save(self.backend.pages.root / "第一話.png")
        response = self.client.get("/api/image/第一話.png")
        self.assertEqual(response.status_code, 200)
        response.close()
        self.assertEqual(
            self.backend.pages.output("第一話.png").name, "第一話.png"
        )

    def test_an_upload_keeps_its_suffix_and_its_neighbours(self):
        buffer = io.BytesIO()
        page_with_text().save(buffer, format="PNG")
        for _ in range(2):
            buffer.seek(0)
            body = self.client.post(
                "/api/pages",
                data={"file": (io.BytesIO(buffer.getvalue()), "第一話.png")},
                content_type="multipart/form-data",
            ).json
            self.assertTrue(body["added"][0].endswith(".png"))
        self.assertEqual(len(body["pages"]), 3)

    def test_an_upload_that_is_not_an_image_is_ignored(self):
        body = self.client.post(
            "/api/pages",
            data={"file": (io.BytesIO(b"nope"), "evil.sh")},
            content_type="multipart/form-data",
        ).json
        self.assertEqual(body["added"], [])
        self.assertEqual(body["pages"], ["001.png"])

    def test_detection_is_cached_per_page(self):
        calls = []
        self.backend._detector = lambda image: calls.append(1) or Detection(
            blocks=[], mask=None
        )
        with mock.patch.object(ocr, "read", return_value=""):
            self.client.post("/api/detect", json={"page": "001.png"})
            self.client.post("/api/detect", json={"page": "001.png"})
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
