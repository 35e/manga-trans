# manga-trans

Find the text on a manga page, read it, hide it under white, and set your own
text in its place. The back end is an HTTP API; `web/` is a dashboard that talks
to it, and any other front end may just as well.

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

Two models, both baked into the container at build time:

- `comictextdetector.pt.onnx` (~95 MB) finds the lettering. It is published as a
  GitHub release asset that no package manager knows how to get, so it is
  downloaded on first use into `~/.cache/manga-trans`. It runs on OpenCV's ONNX
  backend, which needs neither torch nor onnxruntime.
- `kha-white/manga-ocr-base` (~450 MB) reads it. It is a model trained on manga,
  which is why it copes with vertical lines and stylised fonts that general OCR
  does not, and it comes from Hugging Face on first use. It needs torch, and
  torch is why the image is around 2 GB rather than 500 MB. Nothing but
  `/api/read` loads it, so an API only ever asked to detect never pays for it.

## The API

Four endpoints. The page goes up as a multipart `image` field every time; boxes
are `[x0, y0, x1, y1]` in image pixels, sent as JSON in a form field beside it.
Every response carries `Access-Control-Allow-Origin`, so a front end on another
port can call it.

| | | |
| --- | --- | --- |
| `POST` | `/api/detect` | `image` → every block of lettering, boxed |
| `POST` | `/api/letters` | `image` → the lettering itself, pixel by pixel |
| `POST` | `/api/read` | `image`, `boxes` → what each box says |
| `GET` | `/api/models` | → the models Ollama has to translate with |
| `POST` | `/api/translate` | `texts`, `model` → the same lines in another language |
| `POST` | `/api/clean` | `image`, `boxes` and/or `mask` → the page with them whited out |
| `POST` | `/api/render` | `image`, `regions` → the same, with text set in them |

```bash
curl -sX POST localhost:8000/api/detect -F image=@001.png
# {"width": 1114, "height": 1600,
#  "regions": [{"box": [812, 96, 949, 324], "confidence": 0.93}, ...]}

curl -X POST localhost:8000/api/letters -F image=@001.png -o letters.png
curl -X POST localhost:8000/api/letters -F image=@001.png -F grow=6 -o fatter.png

curl -sX POST localhost:8000/api/read -F image=@001.png \
     -F 'boxes=[[812,96,949,324]]'
# {"texts": ["おはようございます"]}

curl -X POST localhost:8000/api/clean -F image=@001.png \
     -F 'boxes=[[812,96,949,324]]' -o clean.png

curl -X POST localhost:8000/api/clean -F image=@001.png \
     -F mask=@mask.png -o clean.png       # white where it should be hidden

curl -X POST localhost:8000/api/render -F image=@001.png \
     -F 'regions=[{"box":[812,96,949,324],"text":"Good morning!"}]' -o out.png
```

**`/api/detect`** boxes every piece of lettering it finds. A `confidence` under
0.6 is worth a second look. It says where the text is, not what it says.

**`/api/letters`** answers with the lettering itself rather than the box around
it: a page-sized PNG, opaque white on the ink and clear everywhere else, which
is the same detector's segmentation head — the one that answers per pixel rather
than per block. Sent on to `/api/clean` it hides the words and leaves the art
they were drawn over, which a rectangle cannot do. `grow` (default 2, up to 50)
is how many pixels to spread the mask by, so no halo is left ringing a letter
that has been hidden; a bolder or blurrier scan wants more.

**`/api/read`** says what it says: one string per box, in the order the boxes
were given, so they line up with the regions `/api/detect` returned. Reading is
a separate call because the boxes are worth correcting first — a box that clips
half a bubble reads half a sentence. A box too small to hold lettering comes
back as `""`, and the box is given a few pixels of air before it goes to the
model. It translates nothing: what the text should say instead is still for the
caller to decide.

**`/api/translate`** is the one thing here that works on words alone — no image
goes with it. `texts` is a JSON list, `model` is one of the names `/api/models`
gives back, and `target` is the language to translate into (English unless
said). The whole page goes over in one request rather than one line at a time:
that is both far quicker and better translation, since a line of manga read on
its own often cannot be translated at all, having no idea who is speaking or
about what. The model is held to a JSON schema so the answers come back
countable, and if it loses count anyway the lines are asked about one at a time,
where it cannot. One translation comes back per text, in order; a text that was
empty stays empty.

The model runs under [Ollama](https://ollama.com) on your own machine, and
nothing is sent anywhere else. Set `MANGA_TRANS_OLLAMA` if it is not on this
one. Both endpoints answer `503` when there is nothing listening there.

**`/api/clean`** paints each box white and hands the page back as a PNG. It also
takes a `mask`: a greyscale image the size of the page, white where the page
should be painted out and black where it should be left alone, sent as a second
file field. The greys between are how much white to lay on, so a brushed edge
comes out soft rather than as a staircase. A mask that is see-through in places
— which is what `/api/letters` hands back — is read by its transparency
instead. Not by merely having an alpha channel: a browser canvas always exports
one, so it has to be a channel some of which is actually clear. A box can only ever say "all of this
rectangle"; a mask can say "this bubble except that corner", which is what a
front end with a brush in it needs. Send boxes, a mask, or both — but not
neither.

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
| `MANGA_TRANS_OCR_MODEL` | `kha-white/manga-ocr-base` |
| `MANGA_TRANS_OLLAMA` | `http://localhost:11434`, or `http://host.containers.internal:11434` in the container |
| `HF_HOME` | where the reader's weights are cached; `/opt/models/hf` in the container |
| `MANGA_TRANS_FONT` | DejaVu Sans Bold, or Pillow's default |

## The code

```
api/
  mangatrans/
    detect.py     comic-text-detector on OpenCV's ONNX backend: blocks, and
                  the per-pixel mask of the lettering inside them
    read.py       manga-ocr, and the cropping that feeds it
    ollama.py     translating a page by a model on this machine
    render.py     whiting out the old lettering, fitting and setting the new
    server.py     the API itself
    geometry.py   Box
  tests/
web/
  src/            the dashboard: drop pages in, detect, read, mask, clean
```

The dashboard puts a page on a board and works it in three tabs. **Inspect**
boxes the lettering and reads it. **Mask** marks the lettering itself for
hiding — not the boxes around it — and that mask can be brushed by hand, drawn
wider or erased back, with blocks worth keeping dropped from it one at a time.
**Translate** sets each translated line back where its original was, in Anime
Ace, in a box that can be dragged about, pulled wider or narrower by its edges,
and sized with the arrow keys. **Apply to image** draws the lot into the page
and saves it — in the browser, with the same font, sizes and wrapping shown on
the board, so what comes out is what was arranged. `/api/render` letters a page
too, and letters it with PIL: it takes boxes and text and finds its own sizes,
which is the endpoint to reach for from something that is not this dashboard.

Translating needs Ollama running with a model pulled:

```bash
ollama pull gemma4:12b
podman run --rm --init -p 8000:8000 manga-trans   # reaches Ollama on the host
```

The dashboard is a Vite app. `cd web && pnpm install && pnpm dev` serves it on
port 5173, pointed at `http://localhost:8000` unless `VITE_API_URL` says
otherwise.

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

They stub the detector and the reader, so they need no model, no network and no
torch.
