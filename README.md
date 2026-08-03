# manga-trans

OCR a manga page, translate it, and letter the translation back into the
bubbles it came out of — **grouped per speech bubble**, because the bubble is
found first and the text is matched to it, rather than the other way round.

## Quick start

Drop your pages in `pages/` and run:

```bash
./run.sh                          # OCR every image in pages/ -> pages/out/
./run.sh --translate              # ... and translate each bubble with ollama
./run.sh --translate --render     # ... and letter it back onto the page
./run.sh --clean --mask           # or: just cover the Japanese, letter it yourself
```

`run.sh` builds the container image the first time (a few minutes), then uses
it for every run. It picks up podman or docker, whichever you have, mounts
`pages/` and points the container at ollama on your host. Everything after
`run.sh` is passed straight to the script, so `./run.sh --help` lists every
flag. Other entry points:

```bash
./run.sh 001.jpg --save-viz   # a single page, plus an annotated copy
./run.sh test                 # unit tests (no models needed)
./run.sh build                # rebuild the image after changing the code
```

## Install

Needs Python 3.10+.

```bash
pip install -r requirements.txt
```

On Linux this pulls the CUDA build of PyTorch (~4 GB of `nvidia-*` wheels). If
you only run on CPU, install a CPU-only torch first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Both models download themselves on first run: the CRAFT detector (~80 MB, from
the EasyOCR GitHub releases) and `kha-white/manga-ocr-base` (~450 MB, from
HuggingFace).

## How it works

```
detect  ->  segment  ->  match  ->  group  ->  recognise  ->  translate  ->  render
CRAFT       bubbles      text to    one per    manga-ocr      ollama         erase +
            & plates     a shape    utterance                                letter
```

1. **Detect** — EasyOCR's CRAFT detector finds the text fragments (roughly one
   box per column of glyphs). EasyOCR's own line merging is switched off: its
   thresholds assume horizontal Latin text, and so do its defaults —
   `--text-threshold` and `--low-text` are set lower here (0.5 and 0.3 against
   CRAFT's 0.7 and 0.4) because a kana scores lower per character than a Latin
   word does. On the sample pages that finds about a fifth more fragments
   without picking anything up off the artwork.
2. **Segment** — the page is broken into the *shapes* text sits on. See
   [Finding the bubbles](#finding-the-bubbles).
3. **Match** — each fragment is assigned to the smallest shape that covers it,
   so a bubble always beats the pale panel it happens to be drawn on. Fragments
   on no shape at all are judged by what is behind them.
4. **Group** — the fragments of one shape become one text group, split into
   separate utterances by [how the text lines up](#grouping-into-utterances).
5. **Recognise** — each group is cropped as a whole and read by `manga-ocr`,
   which is trained on complete manga text blocks and handles vertical text,
   multiple lines and furigana itself.
6. **Translate / render** — see [Translation](#translation) and
   [Rendering](#rendering-the-translation-onto-the-page).

Everything lives in the `mangatrans` package, one module per stage;
`manga_ocr_groups.py` is the entry point the container runs.

## Finding the bubbles

Asking "how white is it behind this text?" cannot tell a bubble from a sound
effect painted on a pale panel — both are white. What separates them is
**shape**: a bubble is drawn *around* its lettering. So the page is segmented
into candidate shapes first, and a shape is only kept when it

- is an **island** — a connected run of page colour that does not touch the
  page edge;
- is a plausible **size** for a bubble;
- is **compact** — most of its own convex hull (`--min-solidity`), which throws
  out the ragged gaps between pieces of artwork;
- is **flat** — almost none of it mid-grey (`--max-midtone`). Bubble paper is
  paper; a highlight on a face or a sleeve carries shading and screentone;
- is mostly **itself** rather than what it encloses, which rejects the ring an
  outline forms when the page is read the other way up;
- **holds text the detector actually found** (`--contains`);
- and is **no more than a few times the size of that text**
  (`--max-bubble-ratio`). A bubble hugs its lettering; a blank patch of sky
  with a scribble on it does not.

Both polarities are segmented, so an inverted caption box — white text on a
black plate — is found the same way (`--no-inverted` turns that off).

Two smaller things make it hold up on real scans. Thresholds are fractions of
the page's own white point rather than fixed levels, so a dim scan reads the
same as a bright one. And anti-aliased outlines leak: the page colour seeps
through a thin grey line and the bubble drains into the background. The mask is
eroded by a pixel before labelling to seal those (`--seal`) and grown back
afterwards.

What is left over — text on no shape at all — is sorted by what is behind it,
measured on the ring around it so the lettering does not count as its own
background:

| kind | what it is | kept by default |
| --- | --- | --- |
| `bubble` | in a speech bubble, caption box or sign | yes |
| `plate` | free-standing on plain paper: narration, a title | yes |
| `art` | painted over artwork: sound effects, scribbles | no |

```bash
./run.sh --text bubbles     # dialogue only
./run.sh --text page        # default: also narration on plain paper
./run.sh --text all         # also sound effects  (--all-text is an alias)
```

Sound effects are dropped by default because they OCR badly and drag the rest
of the page's translation down with them. Each group's `kind` and the plainness
score behind it land in the JSON, so you can tune against real numbers:

```
processing /pages/004.webp...
  skipping art text at [640, 398, 672, 428] (plain 0.69, 1 fragment(s))
```

## Grouping into utterances

Japanese runs in columns, and the two ways a block of text continues are not
interchangeable:

- **Along** a column it may break for a beat — a short line, an ellipsis, a
  trailing `？` — and still be one sentence. A wide gap there is normal, so
  `--stack-gap` is generous (3× the glyph size).
- **Across** columns the spacing is set by the typesetting and is tight.
  A wide gap means a different block — which is exactly what two bubbles drawn
  overlapping look like, since they share one blob of paper. `--gap` is
  correspondingly tight (1.2×).

Which limit applies is decided by the axis the two fragments line up on. Judging
both directions with one number is what used to make a single bubble split into
two groups while a pair of overlapping bubbles merged into one.

| Symptom | Fix |
| --- | --- |
| One bubble split into several groups | raise `--gap`, or `--stack-gap` if the pieces are stacked |
| Two utterances in one bubble merged | lower `--gap` |
| A real bubble treated as free text | raise `--max-bubble-ratio`, lower `--min-solidity` or raise `--max-midtone` |
| Free text dropped as a sound effect | lower `--plain-threshold`, or `--text all` |
| Text missed entirely | lower `--text-threshold` / `--low-text`, or raise `--mag-ratio` for small text |
| Japanese survives the cover-up | raise `--mask-reach`, then `--erase-pad` |
| The cover-up eats the artwork | lower `--mask-reach` and `--mask-knit` |
| Noise picked up as text | raise `--min-fragments` or `--min-group-px` |
| Run dies with no output (exit 137) | out of memory — see [Large pages](#large-pages) |

Run with `--no-ocr --viz out.png` to see the result instantly without loading
the OCR model. The annotated copy tints each bubble that was used, outlines
fragments in blue, numbers each group in its kind's colour (green `bubble`,
orange `plate`, purple `art`) and marks what was thrown away in red.

## Rendering the translation onto the page

`--render` writes `<name>.render.jpg` next to the JSON: the original Japanese is
removed and the translation lettered into the space it occupied.

```bash
./run.sh --translate --render
```

**Erasing.** Under a bubble's lettering is the bubble, and what the bubble looks
like without its text can be *measured* rather than guessed: a greyscale closing
with a brush wider than one character lifts the strokes out and leaves the
surface they were sitting on — flat white for an opaque bubble, the right shade
for one carrying a wash or with artwork showing through. That surface is painted
back over the whole inside of the bubble, so unboxed furigana and stray specks
go with it.

Only text left standing on **artwork** needs a guess, and there only its
*strokes* are handed to OpenCV's inpainter — a few percent of the area a
box-shaped mask would have covered, which is the difference between a repair and
a smudge.

**Lettering.** A Japanese bubble is tall and narrow because the text runs in
columns; English wants the opposite shape. Since the bubble itself is known, the
type is fitted to the bubble: for a handful of aspect ratios we take the largest
rectangle of that shape that fits wholly inside it, and keep whichever lets the
type be biggest. Two utterances sharing one blob of paper get the half nearest
their own text rather than both claiming all of it. Text with no bubble behind it
falls back to reshaping its own footprint, held to the plain paper it was
lettered on so a long line cannot wander onto the artwork.

Long words are left to overflow rather than hyphenated, which is what forces the
size search down instead of letting it answer every question with a bigger font
and a line of confetti; hyphens appear only when a word cannot fit at any size.

| Flag | Default | |
| --- | --- | --- |
| `--render` | off | write `<out-dir>/<name>.render.jpg` |
| `--render-to` | — | render a single image to this path |
| `--font` | DejaVu Sans Bold | any TTF/OTF; the image bundles `fonts-dejavu-core` |
| `--text-colour` | `black` | lettering colour |
| `--halo-colour` | `white` | outline behind lettering that had to stay on artwork |
| `--line-spacing` | `0.16` | extra gap between lines, in font sizes |
| `--erase-pad` | `0.12` | how far erasing spills past each glyph, in glyph sizes |

`--render` needs `--translate`. Two bubbles drawn overlapping share one blob of
paper; when their text also lines up they are lettered as one block across both
— check with `--save-viz` if something looks off.

## Covering the Japanese to letter it yourself

If you would rather set the English by hand — in Photoshop, in a comic editor,
or from your own script — you do not need the translator or the renderer at all.
`--clean` writes the page with the Japanese covered in flat white, `--mask`
writes the mask that covered it, and neither needs ollama:

```bash
./run.sh --clean --mask               # <name>.clean.png + <name>.mask.png
./run.sh --clean --no-ocr             # fastest: no recognition model either
```

The white goes **exactly over the lettering** rather than over the box around
it. Inside a bubble the shape of the bubble is already known, so the question is
not "which pixels did the detector box?" but "which pixels on this piece of
paper are not paper?" — which finds the furigana beside a kanji and the specks
the detector never boxed, and which stops at the drawn outline, so a cover-up
can never spill onto the artwork. What it finds is then closed up, so a column
of type comes out as one patch to letter onto rather than a constellation of
glyph-shaped holes.

| Flag | Default | |
| --- | --- | --- |
| `--clean` | off | with `--out-dir`, write `<name>.clean.png` |
| `--mask` | off | with `--out-dir`, write `<name>.mask.png` (white = covered) |
| `--clean-to`, `--mask-to` | — | write one image to this path instead |
| `--mask-mode` | `text` | `text`, `bubble` or `auto` — see below |
| `--mask-colour` | `white` | any colour name or `#rrggbb` |
| `--mask-reach` | `1.0` | how far past the detected text to look for leftovers, in glyph sizes |
| `--mask-knit` | `0.25` | how far to close the gaps between strokes (`0` traces each glyph exactly) |
| `--erase-pad` | `0.12` | how far to bleed past each glyph, in glyph sizes |

Three shapes of cover-up, in `--mask-mode`:

- **`text`** — exactly over the lettering. The default here, and what you want
  when the artwork inside the bubble matters.
- **`bubble`** — the whole inside of every bubble that held text. Roughly three
  times the paint, and the most room to fit English into, which is what a
  scanlation usually does. Text with no bubble behind it is still covered
  tightly, because there is nothing there to fill.
- **`auto`** — do not lay down a flat colour at all: measure the surface the
  text was sitting on and paint *that* back, and inpaint sound effects standing
  on artwork. This is what `--render` uses, and it is the one to pick when the
  bubbles carry a wash or a gradient. Passing `--mask-mode` explicitly makes
  `--render` use that mode too.

Each group's JSON gains a `mask_bbox` — the box the white actually covers, and
so the room the English has to play with:

```bash
./run.sh --clean --mask --no-ocr      # then read <name>.json for the boxes
```

Two things are worth knowing before you tune. Screentone is not paper, so a
naive "cover everything that is not paper" swallows a whole dotted panel; tone
dots are dropped by size, which is what tells a speck from a stroke, so a
sound effect lettered over a dot screen is covered and the screen is not. And a
leading `……` or a trailing `？` set in a column of its own is often not boxed by
the detector at all — `--mask-reach` is what picks those up, since inside a
bubble there is nothing to find but lettering. Raise it if stray marks survive;
lower it if the utterance next to it gets caught.

## Container (podman / docker)

CPU-only image with both models baked in and `HF_HUB_OFFLINE=true`, so a run
makes no network calls at all. `run.sh` wraps all of this; the raw commands are
here for when you want to change them.

```bash
podman build -t manga-trans .          # ~7 min, ~4.7 GB image
```

With no arguments every image in the mounted folder is read and one
`<name>.json` per page is written to `pages/out/`:

```bash
cp ~/scans/*.webp pages/

podman run --rm \
  -v ./pages:/pages \
  --userns=keep-id:uid=10001,gid=10001 \
  manga-trans --cpu --save-viz
```

```
=== /pages/001.webp - 14 text group(s) ===
[1] bubble bbox=(859,119)-(966,358) fragments=2
...
wrote /pages/out/001.json
wrote /pages/out/001.boxes.png
```

`--userns=keep-id:uid=10001,gid=10001` maps your host user onto the container
user (rootless podman), so files written to `pages/` are owned by you. Notes:

- SELinux hosts (Fedora/RHEL): mount as `-v ./pages:/pages:Z`.
- docker instead of podman: drop `--userns` and use `--user "$(id -u):$(id -g)"`.
- `--cpu` is optional; the image has CPU-only torch, it just silences the
  "CUDA not available" warning.
- `--build-arg PREFETCH_MODELS=false` leaves the models out (~930 MB smaller)
  and downloads them on first run instead — mount a cache so that happens once:
  `-v manga-models:/opt/models`. That build also sets `HF_HUB_OFFLINE=false`, so
  the download can actually happen.

Or via compose:

```bash
podman compose run --rm manga-trans           # whole pages/ folder
podman compose run --rm manga-trans --gap 1.6 # same, with a flag
```

## Usage

```bash
python manga_ocr_groups.py page.jpg
```

```
=== page.jpg - 4 text group(s) ===
[1] bubble bbox=(859,119)-(966,358) fragments=2
    おはようございます
[2] bubble bbox=(241,422)-(344,660) fragments=2
    きょうはいいてんき
...
```

Folders work too, and one JSON per page can be written with `--out-dir`:

```bash
python manga_ocr_groups.py pages --out-dir pages/out     # every image in pages/
python manga_ocr_groups.py pages --out-dir pages/out --save-viz --recursive
python manga_ocr_groups.py                               # current folder
```

With no arguments the current folder is scanned (override with
`MANGA_TRANS_INPUT`; `MANGA_TRANS_OUT_DIR` sets `--out-dir`, which is how the
container defaults to `/pages` → `/pages/out`). Images are processed in natural
order and the output folder is skipped when scanning, so re-runs stay stable.

Other options:

```bash
python manga_ocr_groups.py page.jpg --json out.json      # all pages in one file ('-' = stdout)
python manga_ocr_groups.py page.jpg --viz boxes.png      # annotated image
python manga_ocr_groups.py page.jpg --text all           # keep sound effects too
python manga_ocr_groups.py page.jpg --no-ocr --viz b.png # tune without the OCR model
python manga_ocr_groups.py *.jpg --json chapter.json     # several pages at once
python manga_ocr_groups.py page.jpg --cpu                # force CPU
```

`python manga_ocr_groups.py --help` lists every knob.

Note that `--no-ocr` still writes JSON when `--out-dir` is active — with empty
`text` fields, overwriting earlier results. Point it at a scratch folder
(`--out-dir /tmp/tune`) while tuning.

## Translation

`--translate` sends each page's bubbles to a local [ollama](https://ollama.com)
in one request, so the model sees the whole page as context, and writes the
result to `<name>.txt` (plus `pages.txt` for the whole run) in reading order.

```bash
ollama serve                                   # if it is not already running
python manga_ocr_groups.py pages --out-dir pages/out --translate
python manga_ocr_groups.py pages --out-dir pages/out --translate \
  --ollama-model qwen3-vl:8b --target-lang Dutch --txt-format translation
```

```
# 006.webp
[1] またがって〜
    -> Climb up~
[2] ゴロン
    -> Goron
```

The default model is `gemma4:12b` — pull it once with `ollama pull gemma4:12b`,
or point `--ollama-model` at whatever `ollama list` shows you have.

| Flag | Default | |
| --- | --- | --- |
| `--ollama-url` | `$OLLAMA_URL`, else `http://localhost:11434` | container default is `http://host.containers.internal:11434` |
| `--ollama-model` | `$OLLAMA_MODEL`, else `gemma4:12b` | `ollama list` shows what you have |
| `--target-lang` | `English` | any language the model knows |
| `--ollama-timeout` | `180` | seconds per request |
| `--txt-format` | `both` | `both`, `translation` or `original` |
| `--txt` | `<out-dir>/pages.txt` | combined text file |

Alignment is enforced with a JSON schema (ollama's structured outputs), so bubble
_n_ always gets translation _n_; a bubble the model skips is retried on its own.
Thinking is switched off (`"think": false`) — on a thinking model that is the
difference between a couple of seconds and a minute or more per page, and models
without a thinking mode are retried automatically without the flag.

## Large pages

Detection memory grows with the resolution the detector runs at, not with the
file on disk: roughly **1.4 kB per pixel of detection canvas**, on top of ~400 MB
for torch and the models. At the usual canvas of 2560 a 2894×4093 scan wants
~4.4 GB, so on a 4 GB machine (a default podman VM, a small container) the
process is simply killed — no traceback, just exit 137.

`--canvas-size auto`, the default, avoids that: before each page it reads the
memory actually available (the cgroup limit inside a container, `MemAvailable`
outside one) and picks the largest canvas that fits, between 640 and 2560.

```
processing jpeg_020.jpg...
  2894x4093 page, detecting at canvas 1280
```

Detected boxes come back in original-image coordinates either way, so the JSON,
the crops and `--viz` are unaffected — only how much fine detail the detector
can see. Notes:

- Pass an explicit `--canvas-size 1600` when you want the same result every run;
  free memory drifts, so `auto` may pick a different canvas on a busy machine.
- Give the container more memory and `auto` uses it (`podman run --memory=8g`).
- `--mag-ratio` and `--canvas-size` are not independent: the magnified long side
  is *clamped* to the canvas, so a magnification bigger than the canvas allows is
  silently thrown away. `auto` sizes the canvas to fit the magnification you
  asked for (memory permitting), so `--mag-ratio 1.5` really does detect at 1.5×.
  With an explicit `--canvas-size`, keep it at or above `mag-ratio × long side`
  or the flag does nothing.
- If small furigana goes missing on a low-memory machine, that is the trade —
  the canvas is the detail budget, and it is bounded by the memory available.

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
          "translation": "Good morning",
          "kind": "bubble",
          "plainness": 1.0,
          "mask_bbox": [848, 100, 985, 375],
          "region": {
            "bbox": [840, 100, 985, 375],
            "polarity": "light",
            "solidity": 0.97,
            "midtone": 0.03
          },
          "fragments": 2,
          "fragment_boxes": [[859, 119, 909, 358], [916, 119, 966, 313]]
        }
      ]
    }
  ]
}
```

`bbox` is `[x0, y0, x1, y1]` in pixels. `region` is the shape the text was found
on, or `null` for free-standing text. `mask_bbox` is what `--clean`/`--mask`
covered, and is `null` unless one of those ran. Groups are returned in best-effort manga
reading order (top to bottom, right to left); use `--order ltr` or `--order none`
to change that.

## Tests

`test_grouping.py` covers the geometry, grouping, segmentation, erasing and
layout logic against a synthetic page. It needs numpy, OpenCV and Pillow, but no
models and no network:

```bash
./run.sh test                  # in the container
python test_grouping.py        # or: python -m pytest test_grouping.py
```

`tests/` holds four real pages to check changes against by eye:

```bash
python manga_ocr_groups.py tests --no-ocr --out-dir /tmp/tune --save-viz
```

## Notes

- The detector is a general-purpose text detector, not a bubble detector; the
  bubbles come from segmenting the page. Heavily stylised SFX are hit and miss,
  which is fine, because they are dropped by default anyway.
- Recognition quality is whatever `manga-ocr` gives you — it is Japanese-only.
  Pass `--model` to use a fine-tune or a local copy.
- A bubble that pokes out of its panel and merges with the page margin is not an
  island any more, so it is classified as free text on plain paper rather than as
  a bubble. It is still kept and still erased cleanly; it just does not get the
  bubble-shaped lettering box.
