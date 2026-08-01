# manga-trans

OCR a manga page and get the text back **grouped per text box** — fragments that
sit close together belong to the same speech bubble, fragments separated by a
gap become separate groups.

## Install

```bash
pip install -r requirements.txt
```

Both models download themselves on first run: the CRAFT detector (~80 MB, from
jaided.ai) and `kha-white/manga-ocr-base` (~450 MB, from HuggingFace).

## Usage

```bash
python manga_ocr_groups.py page.jpg
```

```
=== page.jpg - 4 text group(s) ===
[1] bbox=(859,119)-(966,358) fragments=2
    おはようございます
[2] bbox=(241,422)-(344,660) fragments=2
    きょうはいいてんき
...
```

Other options:

```bash
python manga_ocr_groups.py page.jpg --json out.json      # structured output ('-' = stdout)
python manga_ocr_groups.py page.jpg --viz boxes.png      # annotated image, groups numbered
python manga_ocr_groups.py page.jpg --gap 1.6            # merge more aggressively
python manga_ocr_groups.py page.jpg --no-ocr --viz b.png # tune grouping without loading the OCR model
python manga_ocr_groups.py *.jpg --json chapter.json     # several pages at once
python manga_ocr_groups.py page.jpg --cpu                # force CPU
```

`python manga_ocr_groups.py --help` lists every knob.

## How it works

1. **Detect** — EasyOCR's CRAFT detector finds individual text fragments.
   EasyOCR's own line-merging is switched off (its thresholds assume horizontal
   Latin text), so what comes back are raw fragments.
2. **Group** — fragments are clustered by proximity. Two fragments join the same
   group when the **edge-to-edge gap** between them is at most `--gap` × the
   glyph size of the smaller fragment. The threshold is relative to glyph size,
   so small dialogue and big SFX on the same page both group sensibly.
   Clustering is single-linkage, so a chain of nearby fragments forms one box.
3. **Recognise** — each group is cropped as a whole (with a little padding) and
   read by `manga-ocr`, which is trained on complete manga text blocks and
   handles vertical text, multiple lines and furigana itself.

## Tuning the grouping

`--gap` is the knob that matters. Run with `--no-ocr --viz` to see the result
instantly (red = groups, blue = detected fragments) without loading the OCR
model.

| Symptom | Fix |
| --- | --- |
| One bubble split into several groups | raise `--gap` (e.g. `1.5`, `2.0`) |
| Two nearby bubbles merged into one | lower `--gap` (e.g. `0.7`), or cap it with `--max-gap-px` |
| Text missed entirely | lower `--text-threshold` / `--low-text`, or raise `--mag-ratio` for small text |
| Noise picked up as text | raise `--min-fragments` or `--min-group-px` |

`--min-gap-px` / `--max-gap-px` clamp the computed threshold in absolute pixels,
which helps on pages that mix very small and very large lettering.

## JSON output

```json
{
  "pages": [
    {
      "image": "page.jpg",
      "width": 1200,
      "height": 1700,
      "fragments_detected": 7,
      "groups": [
        {
          "bbox": [859, 119, 966, 358],
          "text": "おはようございます",
          "fragments": 2,
          "fragment_boxes": [[859, 119, 909, 358], [916, 119, 966, 313]]
        }
      ]
    }
  ]
}
```

`bbox` is `[x0, y0, x1, y1]` in pixels. Groups are returned in best-effort manga
reading order (top to bottom, right to left); use `--order ltr` or `--order none`
to change that.

## Tests

`test_grouping.py` covers the geometry and clustering logic and needs no models:

```bash
python test_grouping.py        # or: python -m pytest test_grouping.py
```

## Notes

- The detector is a general-purpose text detector, not a speech-bubble detector.
  It works well on dialogue; heavily stylised SFX are hit and miss.
- Recognition quality is whatever `manga-ocr` gives you — it is Japanese-only.
  Pass `--model` to use a fine-tune or a local copy.
- Group crops that accidentally span two bubbles will be read as one run-on
  string; that is a `--gap` tuning problem, check with `--viz`.
