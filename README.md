# manga-trans

OCR a manga page and get the text back **grouped per text box** — fragments that
sit close together belong to the same speech bubble, fragments separated by a
gap become separate groups.

## Quick start

Drop your pages in `pages/` and run:

```bash
./run.sh                 # OCR every image in pages/ -> pages/out/
./run.sh --translate     # ... and translate each bubble with ollama
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

## Container (podman / docker)

CPU-only image with both models baked in and `HF_HUB_OFFLINE=true`, so a run
makes no network calls at all — the detector and `manga-ocr` load from the image
in about a second, every time. `run.sh` above wraps all of this; the raw
commands are here for when you want to change them.

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
=== /pages/001.webp - 4 text group(s) ===
...
wrote /pages/out/001.json
wrote /pages/out/001.boxes.png
```

Add `--translate` to also run every bubble through your local ollama:

```bash
podman run --rm -v ./pages:/pages --userns=keep-id:uid=10001,gid=10001 \
  manga-trans --cpu --translate
```

The container reaches ollama on the host at `host.containers.internal:11434`
(already the image default) and uses `gemma4:12b`. Alongside the JSON you get
`<name>.txt` per page and a combined `pages.txt`, both in reading order:

```
# 006.webp
[1] またがって〜
    -> Climb up~
[2] ゴロン
    -> Goron
```

Pages are processed in natural order (`page2` before `page10`), `--save-viz` is
optional, and anything already in `out/` is never picked up as input, so
re-running is safe. Everything after the image name is passed straight to
`manga_ocr_groups.py` — `podman run --rm manga-trans --help` lists every flag,
and a single page still works:

```bash
podman run --rm -v ./pages:/pages --userns=keep-id:uid=10001,gid=10001 \
  manga-trans 001.webp --cpu --json out/001.json --viz out/boxes.png
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
[1] bbox=(859,119)-(966,358) fragments=2
    おはようございます
[2] bbox=(241,422)-(344,660) fragments=2
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
python manga_ocr_groups.py page.jpg --viz boxes.png      # annotated image, groups numbered
python manga_ocr_groups.py page.jpg --gap 1.6            # merge more aggressively
python manga_ocr_groups.py page.jpg --no-ocr --viz b.png # tune grouping without loading the OCR model
python manga_ocr_groups.py *.jpg --json chapter.json     # several pages at once
python manga_ocr_groups.py page.jpg --cpu                # force CPU
```

`python manga_ocr_groups.py --help` lists every knob.

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
Thinking is switched off (`"think": false`) — on a thinking model like gemma4 or
qwen3-vl that is the difference between a couple of seconds and a minute or more
per page, and models without a thinking mode are retried automatically without
the flag.

OCR noise gets translated as noise: if a bubble reads as gibberish, that is a
detection/recognition problem, not a translation one — check it with `--save-viz`.

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
- If small furigana goes missing on a low-memory machine, that is the trade —
  more memory, or a smaller `--canvas-size` with a higher `--mag-ratio`.

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
| Run dies with no output (exit 137) | out of memory — see [Large pages](#large-pages) |

`--min-gap-px` / `--max-gap-px` clamp the computed threshold in absolute pixels,
which helps on pages that mix very small and very large lettering.

Note that `--no-ocr` still writes JSON when `--out-dir` is active — with empty
`text` fields, overwriting earlier results. Point it at a scratch folder
(`--out-dir /tmp/tune`) while tuning.

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

`test_grouping.py` covers the geometry, clustering and canvas-budget logic and
needs no models:

```bash
./run.sh test                  # in the container
python test_grouping.py        # or: python -m pytest test_grouping.py
```

## Notes

- The detector is a general-purpose text detector, not a speech-bubble detector.
  It works well on dialogue; heavily stylised SFX are hit and miss.
- Recognition quality is whatever `manga-ocr` gives you — it is Japanese-only.
  Pass `--model` to use a fine-tune or a local copy.
- Group crops that accidentally span two bubbles will be read as one run-on
  string; that is a `--gap` tuning problem, check with `--viz`.
