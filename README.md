# manga-trans

Find the text on a manga page, check it in the browser, and overlay the
translation on the picture.

The detector is good but not right every time, and a box that clips half a
bubble reads as half a sentence. So nothing is rendered until you have seen it:
every region is drawn on the page, you drag it to the size you actually want,
the crop is read again, and only the regions you approve get overlaid.

## Quick start

Drop your pages in `pages/`, then:

```bash
./run.sh            # builds the image the first time, then serves the GUI
```

Open <http://localhost:8000>. Pick a page on the left; detection, OCR and
translation run on their own. Then:

- **drag a box** to move it, **drag a handle** to resize it — the crop is read
  again the moment you let go, so making a region larger is how you pick up the
  furigana or the second column the detector missed
- **drag on empty picture** to add a region the detector never found
- **edit the English** in the panel on the right, or write it yourself
- **untick approve** for anything you want left alone
- **Overlay approved** covers the Japanese and letters your English into each
  box, writing the page to `pages/out/`

The box you drag is the whole contract: it is what gets read, what gets covered,
and where the English is set. A tall narrow bubble gives tall narrow English —
widen the box and the type grows to match.

## Install

Without a container, needs Python 3.10+:

```bash
pip install -r requirements.txt
python scripts/fetch_models.py     # the detector, and manga-ocr's weights
python -m mangatrans               # http://127.0.0.1:8000
```

On Linux `pip install -r requirements.txt` pulls the CUDA build of PyTorch
(~4 GB of `nvidia-*` wheels) for `manga-ocr`. For CPU only, install torch first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Two models. The detector (`comictextdetector.pt.onnx`, ~95 MB) is fetched by
`scripts/fetch_models.py` into `~/.cache/manga-trans`, because it is published
as a GitHub release asset and no package manager knows how to get it.
`kha-white/manga-ocr-base` (~450 MB) downloads itself from HuggingFace on first
use; `--detector` skips it.

Detection runs on OpenCV's ONNX backend, so it needs no torch and no
onnxruntime. Only recognition does.

## Translation

Translations come from [ollama](https://ollama.com) on your machine, and are
filled in for the whole page in one request so the model has the surrounding
bubbles as context. If ollama is not running the GUI says so and everything else
still works — the text boxes are yours to type into.

```bash
OLLAMA_MODEL=qwen3:8b ./run.sh          # a different model
OLLAMA_URL=http://localhost:11434 python -m mangatrans
```

## How it works

```
detect  ->  read  ->  translate  ->  you  ->  overlay
blocks +    manga     ollama         GUI      cover +
mask        -ocr                              letter
```

1. **Detect** — [comic-text-detector](https://github.com/dmMaze/comic-text-detector),
   trained on ~13k comic pages, returns one box per utterance with a confidence,
   plus a per-pixel mask of the lettering.
2. **Read** — each box is cropped and read by `manga-ocr`, which is trained on
   whole manga text blocks and handles vertical text and furigana.
3. **Translate** — the page goes to ollama in one request.
4. **You** — the part the models cannot do. Boxes under 60% confidence are drawn
   in amber; anything you resize is read again.
5. **Overlay** — the lettering mask says which pixels were ink; a greyscale
   closing measures the paper *under* them and paints it back, so a bubble
   carrying a wash or a screentone comes back carrying it rather than turning
   into a white rectangle. The cover never leaves the box you approved. Then the
   English is wrapped and set at the largest size that fits.

## The code

```
mangatrans/
  detect.py      comic-text-detector on OpenCV's ONNX backend
  ocr.py         manga-ocr, loaded on first use
  translate.py   ollama
  render.py      covering the old lettering, fitting and setting the new
  server.py      the HTTP API the browser talks to
  geometry.py    Box
  web/           the GUI: one page, one stylesheet, one script
```

The API is small enough to drive from anything: `POST /api/detect`,
`/api/read`, `/api/translate` and `/api/render` all take and return JSON, with
boxes as `[x0, y0, x1, y1]` in image pixels.

## Options

Every flag has an environment variable, which is what the container uses:

| flag | variable | default |
| --- | --- | --- |
| `--pages` | `MANGA_TRANS_PAGES` | `pages` |
| `--out` | `MANGA_TRANS_OUT` | `<pages>/out` |
| `--host` | `MANGA_TRANS_HOST` | `127.0.0.1` |
| `--port` | `MANGA_TRANS_PORT` | `8000` |
| `--model` | `MANGA_TRANS_MODEL` | `~/.cache/manga-trans/comictextdetector.pt.onnx` |
| `--font` | `MANGA_TRANS_FONT` | DejaVu Sans Bold, or Pillow's default |
| `--ollama-url` | `OLLAMA_URL` | `http://localhost:11434` |
| `--ollama-model` | `OLLAMA_MODEL` | `gemma4:12b` |

## Tests

```bash
./run.sh test                              # in the container
python -m unittest discover -s tests -t .  # or locally
```

They stub the detector and manga-ocr, so they need no models and no network.
