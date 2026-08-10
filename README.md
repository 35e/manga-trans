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

The page goes up as a multipart `image` field every time; boxes are
`[x0, y0, x1, y1]` in image pixels, sent as JSON in a form field beside it.
Every response carries `Access-Control-Allow-Origin`, so a front end on another
port can call it.

| | | |
| --- | --- | --- |
| `POST` | `/api/detect` | `image` → every block of lettering, boxed, with its balloon |
| `POST` | `/api/letters` | `image` → the lettering itself, pixel by pixel |
| `POST` | `/api/bubbles` | `image`, `boxes` → the balloon each one is written in |
| `POST` | `/api/read` | `image`, `boxes` → what each box says |
| `GET` | `/api/models` | → the models Ollama has to translate with |
| `GET` | `/api/prompt` | → what the model is told, unless told otherwise |
| `POST` | `/api/translate` | `texts`, `model` → the same lines in another language |
| `POST` | `/api/clean` | `image`, `boxes` and/or `mask` → the page with them taken out |
| `POST` | `/api/render` | `image`, `regions` → the same, with text set in them |

```bash
curl -sX POST localhost:8000/api/detect -F image=@001.png
# {"width": 1114, "height": 1600,
#  "regions": [{"box": [812, 96, 949, 324], "confidence": 0.93,
#               "bubble": [769, 136, 984, 285]}, ...]}

curl -X POST localhost:8000/api/letters -F image=@001.png -o letters.png
curl -X POST localhost:8000/api/letters -F image=@001.png -F grow=6 -o fatter.png

curl -sX POST localhost:8000/api/bubbles -F image=@001.png \
     -F 'boxes=[[812,96,949,324]]'
# {"regions": [{"box": [812, 96, 949, 324], "bubble": [769, 136, 984, 285]}]}

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
0.8 is worth a second look — the dashboard leaves those blocks alone until one
has been. It says where the text is, not what it says. `bubble` comes along with
it, as below, because it is worked out from the page and the boxes and both are
already in hand.

Two balloons that overlap are often boxed as one, and that is worse than it
looks: the block would be read as one string, so two speakers reach the
translator as a single line and the lettering that comes back is set into one
balloon. So a block is cut apart before it is answered with, wherever a run of
blank inside it is wide enough to be a wall rather than the gap between two
lines. The cuts are axis-aligned, since a block is a rectangle. Each piece keeps
the confidence of the block it came from and gets its own `bubble`, and the
pieces are put in reading order along with everything else.

How much blank that takes depends on what the cut stands through, and everything
is measured in characters rather than pixels so it holds at any size the page is
lettered at. A cut crossing several lines at once needs only **0.8 of a
character**: every line it crosses has to fall blank in the same place at the
same time, which lettering inside one balloon does not do. A cut running the
length of a single line has no other line to agree with it — the gap between two
characters is then a candidate, and small kana and punctuation leave most of
their cell empty — so that one needs **1.5**.

Under about 0.8 the line spacing of a generously set balloon starts being read
as a wall, so that is the floor rather than a preference.

A narrower blank than that is still cut on when the lettering either side of it
is **staggered** — shifted the same way at both ends, by half a character or
more. Lines of one block share an edge: vertical Japanese starts every column at
the same height and simply stops early on the last one. Two blocks set beside
each other share nothing. Both ends have to agree before it counts, and that is
what tells a second block from a column that ended early or from two columns
centred against one another — measured, a balloon of two centred columns is out
by 3.0 characters at the start and back 3.0 at the end, where two balloons a
character apart are out by 1.0 at both. Text set at plainly different heights was
never one block, however close together it sits.

Beyond that this is deliberately shy: a balloon left merged can be split by hand
in the dashboard, where a line of dialogue wrongly cut into four cannot be put
back so easily. A block that does not come apart is answered with exactly as the
detector drew it.

Splitting is also why two blocks can come back covering the same lettering: the
head sometimes draws a box around two balloons *and* a box around one of them,
overlapping too little for the detector's own non-maximum suppression to throw
either away, and cutting the first turns that pair into a duplicate. A second
pass drops any block that covers the same lettering as a surer one, since a
duplicate is read twice, translated twice and lettered twice into one place.

Every block also comes back with a margin around it — a quarter of a character,
so it holds at any size — because the block head boxes lettering tightly and
sometimes clips the edge of a glyph. The margin goes on after the split, or it
would close the very gaps the split is measuring.

**`/api/bubbles`** answers with the room each block was written in rather than
the room its words take up: the largest rectangle that fits inside the balloon
around it. This is where a translation goes. Japanese runs down the page, so a
block of it comes back forty pixels across and three hundred tall, and English
set in a column that shape wraps to about a letter a line — which is why a
translated line used to have to be dragged out to its balloon before it could be
read at all.

Nothing is drawn to say where a balloon is, but it is not hard to see: it is a
light shape closed by a dark outline, so the light pixels reachable from the
lettering without crossing that outline are the balloon. The block is painted in
before the flood starts, because a line of Japanese down the middle of a balloon
cuts its ground into a left half and a right half, and a flood started in one of
them measures the gap beside the words rather than the room around them. Dark
balloons with white lettering are found the same way round the other way.

Where several of the boxes turn out to be written in the *same* balloon — which
is what a block cut in two looks like from here, and what a balloon holding two
separate lines of dialogue is — it is shared out between them rather than handed
to each of them whole. They were told apart by the blank between them in the
first place, so the balloon is cut the same way: at the widest blank between
them, on whichever axis that blank is widest, and then each side again until
every block has a piece to itself. Without that, every one of them would be
answered with the same rectangle and their translations lettered one on top of
another.

Cutting once along one axis is not enough, which is why this recurses: four
blocks set two across and two down are not in a row, and one line of cuts hands
the two on the right a left half and a right half of a balloon they are stacked
inside.

What counts as "the same balloon" is that two answers *overlap*, not that they
match. One balloon does not come back as the same rectangle twice: an irregular
one holds a wide short rectangle and a tall narrow one of nearly the same area,
and a pixel of the flood decides which of them a given block is answered with, so
two blocks in one balloon can come back barely half agreeing. A block no balloon
could be made out for is gathered in too, by the box it will be lettered in —
otherwise its neighbour keeps the whole balloon and sets its translation straight
over the top of it.

This is why the endpoint takes a list rather than one box at a time: the answer
for a block depends on which other blocks share its balloon. Ask about a box on
its own and it is handed the whole balloon, since nothing else is known to be in
it — so send every box on the page whenever the answer is going to be used for
lettering.

`bubble` is null where none could be made out, and then the box is all there is
to go on: a sound effect over artwork is in no balloon, a balloon whose outline
a scan has broken cannot be followed, and a balloon drawn no wider than the words
already are has nothing to offer. It is a guess, and saying so beats answering
with a rectangle somewhere in the artwork. No model is involved — this is the one
call on an image that never stands the detector up.

Detection keeps the last page's pass through the network, since the same page
goes through twice as a matter of course: `/api/detect` for the boxes and
`/api/letters` for the mask are one pass, and changing `grow` asks for it again.
The pass is seconds and everything downstream of it is a millisecond, so asking
about a page a second time costs milliseconds rather than repeating it. Only the
last page is kept — they are worked on one at a time.

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
    split.py      cutting a block that holds two balloons back into one
                  block each, by the blank between them
    bubble.py     the balloon a block was written in, which is where a
                  translation goes — the block is only where the words are
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

An archive arrives as a **folder** in the rail rather than as fifty more
thumbnails. Clicking it goes in, and the arrow at the top of the rail comes back
out. The same archive dropped twice fills the same folder rather than putting a
second one beside it, and a page is only ever the duplicate of another in the same
folder: two chapters both hold a `001.png`, and those are two pages. Deleting a
folder deletes its pages, and deleting the last page deletes the folder.

**Clean & translate all**, inside an opened folder, puts every page in it through
the whole of the above — detect, read, hide, letter — one page at a time, since
the API holds one detector and one reader behind a lock each and pages sent
together only queue there anyway. The bar above the rail says which page is in
hand and what is being done to it, and stays where it is when the folder is
closed; clicking the page's name puts that page on the board. **Stop** stops after
the page in hand, a request already sent being left to land. A page that falls over
is named with the reason it gave and the rest carry on. A folder that has been
lettered already asks once before doing it over, since a line moved or rewritten
by hand cannot be got back.

The dashboard puts a page on a board and works it in three tabs. **Inspect**
boxes the lettering and reads it; a block the detector is less than 80% sure of
is read and listed like any other but starts left alone, since a box over half a
bubble or over a piece of artwork does more harm hidden than a real one does
missed. Putting one back is one click, as is dropping one it was too sure of.
**Mask** marks the lettering itself for
hiding — not the boxes around it — and that mask can be brushed by hand, drawn
wider or erased back, with blocks worth keeping dropped from it one at a time.
**Hide under** beside the brush is what a clean puts back where the marks were:
the art around them, filled in, or flat white.
**Translate** sets each translated line in the balloon its original was written
in — not in the box the original came out of, which for a vertical line of
Japanese is a column too narrow to set one word of English across. It is set in
Anime Ace, as large as it will go in that balloon but no larger than the page is
lettered: a balloon is drawn around its words rather than to them, so the largest
type that fits one is far bigger than the type it was drawn around whenever the
line is short, and "OK!" left to fill its balloon comes out four times the height
of the dialogue on either side of it. How large the page is lettered is read off
the original — Japanese is set on a square em, so a block of *n* characters
covering *w × h* was set at about the square root of *wh/n*. **Fit to balloons**
puts every line back where that says, which is what to reach for after drawing a
block by hand, cutting one in two, or dragging a box somewhere it should not have
gone; a block with no balloon around it is left exactly where it is.

The box can be dragged about, pulled wider or narrower by its edges,
sized with the up and down arrows and turned with the left and right ones — or
by the round handle standing above the box, which follows the pointer round it,
holding shift for 15° at a time. Manga letters plenty of things on the slant: a
sound effect running up the page, a shout across a tilted bubble, and a line set
square over one of those reads as a sticker rather than as part of the art. The
box itself stays square to the page and only what sits in it turns, so a line
still wraps to the width it was given; pulling a turned box by an edge pulls it
along its own axes, and the edge that was not pulled stays where it looks.
**Translate again**, beside it, runs the page
over against the blocks as they stand: blocks are added, dropped and put back
after a page has been translated, and this is what brings the lines back into
step with them — one that was added gets a line, one that went away loses its
own. Every line already set is replaced, so it is the whole page again rather
than the gaps in it, which is also the better translation: the model reads the
page as one conversation. **Apply to image** draws the lot into the page
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
