# manga-trans

Find the text on a manga page, read it, take it back off the art it was drawn
over, and set your own text in its place. The back end is an HTTP API; `web/` is
a dashboard that talks to it, and any other front end may just as well.

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
| `GET` | `/api/prompt` | → what the model is told, unless told otherwise |
| `POST` | `/api/translate` | `texts`, `model` → the same lines in another language |
| `POST` | `/api/clean` | `image`, `boxes` and/or `mask` → the page with them taken out |
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

curl -X POST localhost:8000/api/clean -F image=@001.png \
     -F mask=@mask.png -F fill=white -o flat.png   # painted out, not filled in

curl -X POST localhost:8000/api/render -F image=@001.png \
     -F 'regions=[{"box":[812,96,949,324],"text":"Good morning!"}]' -o out.png
```

**`/api/detect`** boxes every piece of lettering it finds. A `confidence` under
0.6 is worth a second look. It says where the text is, not what it says.

**`/api/letters`** answers with the lettering itself rather than the box around
it: a page-sized PNG, opaque white on the ink and clear everywhere else, which
is the same detector's segmentation head — the one that answers per pixel rather
than per block. Sent on to `/api/clean` it hides the words and leaves the art
they were drawn over, which a rectangle cannot do. `grow` (default 4, up to 64)
is how many pixels to spread the mask by, so no halo is left ringing a letter
that has been hidden. On clean black-on-white lettering two is already enough at
any size of page; what needs more is scanned material — screentone, the ring
JPEG leaves around a hard edge, the pale rim of an outlined letter — and how
much more depends on the scan, which is why the dashboard puts it next to the
button rather than deciding for you.

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

What the model is told is the caller's too: send `system`, with `{target}`
wherever the language should go, and `GET /api/prompt` hands back the default to
start from. Nothing is kept — every request carries its own — so a front end
that wants its own prompt remembers it and sends it each time, which is what the
dashboard's settings do.

Bear in mind that the answers are held to a JSON schema, and a model under a
schema follows a prompt loosely. Asking for a voice — casual, formal, terse —
comes through; asking it to reformat what it returns mostly does not.

The model runs under [Ollama](https://ollama.com) on your own machine, and
nothing is sent anywhere else. Set `MANGA_TRANS_OLLAMA` if it is not on this
one. All three endpoints answer `503` when there is nothing listening there.

**`/api/clean`** takes what is marked out of the page and hands it back as a
PNG. What is marked can be boxes, a `mask`, or both — but not neither. A mask is
a greyscale image the size of the page, white where the page should be hidden
and black where it should be left alone, sent as a second file field; the greys
between are how much of it to lay on, so a brushed edge comes out soft rather
than as a staircase. A mask that is see-through in places — which is what
`/api/letters` hands back — is read by its transparency instead. Not by merely
having an alpha channel: a browser canvas always exports one, so it has to be a
channel some of which is actually clear. A box can only ever say "all of this
rectangle"; a mask can say "this bubble except that corner", which is what a
front end with a brush in it needs.

What goes where the lettering was is `fill`. By default it is `art`: the page
around the mark is looked at and carried inwards, so the tone a sound effect was
drawn over runs on through it, a panel line that went under a letter comes out
the other side, and the edge of a bubble joins back up. `fill=white` paints it
flat instead, which is the old behaviour and still the right one where the
ground was white to begin with — the inside of a bubble, or a box about to have
new lettering set in it.

Filling is OpenCV's Telea inpainting: the marked pixels are made out of the ones
just outside them, working inwards. It costs nothing that is not already
installed — no model, no network — and around a fifth of a second on a page. It
carries flat tone and it carries a line; what it cannot do is put a screentone's
dots back, so a wide sound effect over a dense tone comes back as the flat
average of it rather than the pattern. That is still a good deal better than a
white hole in the middle of the art, and it is what redrawing by hand is for.

Two pixels around every mark are kept out of what the fill is made of but are
not themselves painted over. Lettering is printed with soft edges, and a mask
that stops at the ink leaves a rim of half-ink just outside it; read as art,
that rim would be carried inwards and the letter put back as a smudge. That is
also why `grow` on `/api/letters` matters more here than it did: what the mask
does not reach is what the fill is made of.

**`/api/render`** does the same and sets each region's `text` in its box: wrapped
to the width, centred, black, at the largest size that lands inside it. Text too
long for its box is set at the smallest size and left to overrun rather than
dropped — a line that can be read is a line that can be moved. It takes `fill`
too, but the default is the other way round — `white` — because a region here is
a rectangle and the text set in it is black, and black lettering wants a ground
that is clear rather than one that is whatever was underneath. Both endpoints
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
    inpaint.py    making what was hidden out of the page around it
    render.py     hiding the old lettering, fitting and setting the new
    server.py     the API itself
    geometry.py   Box
  tests/
web/
  src/            the dashboard: drop pages in, detect, read, mask, clean
```

Pages go in by dropping them, pasting them, or picking them — and a chapter can
go in as a `.zip` or `.cbz`, which is opened in the browser. The pages come out
in the order their names put them, counting properly: page 2 before page 10.
Folders, dotfiles and the `__MACOSX` rubbish a Mac packs in are left behind.

The dashboard puts a page on a board and works it in three tabs. **Inspect**
boxes the lettering and reads it. **Mask** marks the lettering itself for
hiding — not the boxes around it — and that mask can be brushed by hand, drawn
wider or erased back, with blocks worth keeping dropped from it one at a time.
**Hide under** beside the brush is what a clean puts back where the marks were:
the art around them, filled in, or flat white.
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
