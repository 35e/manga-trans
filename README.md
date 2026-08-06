# manga-trans

Find the text on a manga page, hide it under white, and set your own text in its
place. This repository is the back end: an HTTP API and nothing else. A front end
talks to it from wherever it likes.

The detector is good but not right every time, and a box that clips half a
bubble hides half a sentence. So the API never renders anything on its own:
hiding and lettering take the boxes they are given back, which is what lets a
front end show the detection, have it corrected, and only then overlay.

Nothing is stored. Every request carries the page it works on and gets its answer
back in the response.

## Run

```bash
podman build -t manga-trans api
podman run --rm --init -p 8000:8000 manga-trans     # http://localhost:8000/api
```

`docker` takes the same two commands.

Without a container, needs Python 3.10+:

```bash
cd api
pip install -r requirements.txt
python -m mangatrans          # http://127.0.0.1:8000/api
```

One model: `comictextdetector.pt.onnx` (~95 MB), downloaded on first use into
`~/.cache/manga-trans` because it is published as a GitHub release asset and no
package manager knows how to get it. The container bakes it in at build time.
Detection runs on OpenCV's ONNX backend, so there is no torch and no onnxruntime.

## The API

Three endpoints. The page goes up as a multipart `image` field every time; boxes
are `[x0, y0, x1, y1]` in image pixels, sent as JSON in a form field beside it.
Every response carries `Access-Control-Allow-Origin`, so a front end on another
port can call it.

| | | |
| --- | --- | --- |
| `POST` | `/api/detect` | `image` → every block of lettering, boxed |
| `POST` | `/api/clean` | `image`, `boxes` → the page with the boxes whited out |
| `POST` | `/api/render` | `image`, `regions` → the same, with text set in them |

```bash
curl -sX POST localhost:8000/api/detect -F image=@001.png
# {"width": 1114, "height": 1600,
#  "regions": [{"box": [812, 96, 949, 324], "confidence": 0.93}, ...]}

curl -X POST localhost:8000/api/clean -F image=@001.png \
     -F 'boxes=[[812,96,949,324]]' -o clean.png

curl -X POST localhost:8000/api/render -F image=@001.png \
     -F 'regions=[{"box":[812,96,949,324],"text":"Good morning!"}]' -o out.png
```

**`/api/detect`** boxes every piece of lettering it finds. A `confidence` under
0.6 is worth a second look. It reads nothing and translates nothing — what the
text says, and what it should say instead, is for the caller to decide.

**`/api/clean`** paints each box white and hands the page back as a PNG.

**`/api/render`** does the same and sets each region's `text` in its box: wrapped
to the width, centred, black, at the largest size that lands inside it. Text too
long for its box is set at the smallest size and left to overrun rather than
dropped — a line that can be read is a line that can be moved. Both endpoints
answer with `image/png`; neither writes anything to disk.

## Options

Everything is set by environment variable:

| variable | default |
| --- | --- |
| `MANGA_TRANS_HOST` | `127.0.0.1` |
| `MANGA_TRANS_PORT` | `8000` |
| `MANGA_TRANS_ORIGIN` | `*` — who may call the API from a browser |
| `MANGA_TRANS_MODEL` | `~/.cache/manga-trans/comictextdetector.pt.onnx` |
| `MANGA_TRANS_FONT` | DejaVu Sans Bold, or Pillow's default |

## The code

```
api/
  mangatrans/
    detect.py     comic-text-detector on OpenCV's ONNX backend
    render.py     whiting out the old lettering, fitting and setting the new
    server.py     the API itself
    geometry.py   Box
  tests/
```

## Tests

From `api/`:

```bash
python -m unittest discover -s tests -t .
```

Or in the container, with the working copy mounted in so a change needs no
rebuild:

```bash
podman run --rm --entrypoint python -w /app \
    -v "$PWD/api/mangatrans:/app/mangatrans:ro" -v "$PWD/api/tests:/app/tests:ro" \
    manga-trans -m unittest discover -s tests -t .
```

They stub the detector, so they need no model and no network.
