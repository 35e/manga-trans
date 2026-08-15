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

The whole thing — the API and the dashboard beside it on one origin:

```bash
podman compose up --build       # http://localhost:8080
```

`docker compose` takes the same command. The API alone, on its own port:

```bash
podman build -t manga-trans api
podman run --rm --init -p 8000:8000 manga-trans     # http://localhost:8000/api
```

Without a container, needs Python 3.10+:

```bash
cd api
pip install -r requirements.txt
python -m mangatrans          # http://127.0.0.1:8000/api
```

The models are all baked into the container at build time:

- `ogkalu/comic-text-and-bubble-detector` (~44 MB) finds the balloons and the
  lettering in one pass. It is an RT-DETRv2 trained on some 11k manga, webtoon,
  manhua and western comic pages, and it answers with three kinds of thing:
  a balloon, text inside one, and text that is in none. It runs on onnxruntime.
- `comictextdetector.pt.onnx` (~95 MB) is kept for one job the box detector
  cannot do: saying which *pixels* are ink, so a clean can take the words off the
  art rather than painting a rectangle over it. Only its segmentation head is
  used. It is a GitHub release asset that no package manager knows how to get,
  and it runs on OpenCV's ONNX backend, which needs neither torch nor onnxruntime.
- `ogkalu/lama-manga-onnx-dynamic` (~206 MB) fills in what was hidden. It is
  big-LaMa fine-tuned on 300k manga and anime pages, so it continues a line and
  a screentone through a hole rather than smearing colour inwards across it.
  On onnxruntime, and it takes any size that is a multiple of eight.
- `kha-white/manga-ocr-base` (~450 MB) reads Japanese. It is a model trained on
  manga, which is why it copes with vertical lines and stylised fonts that
  general OCR does not, and it comes from Hugging Face on first use. It needs
  torch, and torch is why the image is around 2 GB rather than 500 MB.
- PP-OCR's recognisers, a file of some 10–20 MB per language, read everything
  else — Chinese, Korean, English. They run under
  [RapidOCR](https://github.com/RapidAI/RapidOCR) on onnxruntime and come from
  ModelScope on first use.

Every one of them is stood up on first use and only if it is wanted, so an API
only ever asked to detect never loads the reader or the fill, and one only ever
asked for Korean never imports torch.

onnxruntime goes looking for a GPU as it loads and, in a container that has been
shown a card it cannot read the make of, complains that it cannot — which is
every container on a Mac, and which is not a failure of anything: the reading is
on the CPU either way. The line is dropped rather than left in the log to be
mistaken for the reason a page did not come back.

## Languages

The detector knows nothing about scripts — it was trained on comics rather than
on Japanese — so finding the text, tracing it and hiding it are the same work
whatever the page is in. Everything after that is not, and `language` says which:

| code | | reader |
| --- | --- | --- |
| `ja` | Japanese | manga-ocr |
| `zh` | Chinese (simplified) | PP-OCR |
| `zh-Hant` | Chinese (traditional) | PP-OCR |
| `ko` | Korean | PP-OCR |
| `en` | English | PP-OCR |

`GET /api/languages` hands the list out, so a front end need not hold its own
copy. Send the code to `/api/detect` and `/api/read`; anything that says nothing
is read as Japanese, which is what this was written for and what every request
that predates the field meant.

It decides three things. **Which reader**, as above. **Which way the page is
read** — right to left for Japanese and for the Chinese set in columns, left to
right for Korean and for a webcomic — which is the order `/api/detect` answers
in, and so the order the page reaches the translator as one conversation. And
**what the translator is told the page is in**, which is `source` on
`/api/translate`: the same characters are Japanese or Chinese depending on
nothing a model can see from one line of dialogue, and left to guess it will
translate a Chinese page as though it were Japanese.

Traditional Chinese is the entry for a page set in columns and read right to
left, and simplified for the rows and left-to-right of a webcomic, because that
is how the comics printed in each are usually set. A page that is drawn the other
way round still reads correctly — the blocks come back in the wrong order, and
the dashboard drags them into the right one.

**PP-OCR reads across a page and nothing else**, so a balloon is taken apart
before it goes over: cut into its lines, and — where those lines are columns —
each column cut at the gaps between its characters and set out left to right, so
the model is handed the line that column would be rather than one it has to read
sideways. Which way the lettering runs is measured off the ink of each block
rather than taken from the language, since a page of Korean carries a sound
effect written down the side of it just as a page of Japanese does. A column set
so solid that no gap can be found in it is cut on its own width instead: CJK is
set on a square em, so that is where the characters are whether or not the ink
says so.

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
| `GET` | `/api/languages` | → the languages a page can be read in |
| `GET` | `/api/models` | → the models Ollama has to translate with |
| `GET` | `/api/prompt` | → what the model is told, unless told otherwise |
| `POST` | `/api/survey` | `pages`, `model` → what the chapter is, read before any of it is translated |
| `POST` | `/api/translate` | `texts`, `model` → the same lines in another language, and the terms it named |
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

curl -sX POST localhost:8000/api/read -F image=@001.png -F language=ko \
     -F 'boxes=[[812,96,949,324]]'
# {"texts": ["안녕하세요"]}

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
it, as below, because both are already in hand. `language` changes nothing about
what is found, only the order it comes back in.

`kind` comes with it too: `speech` where the lettering is inside a balloon and
`free` where it is not — a sound effect, a caption, a sign. Nothing about
finding, tracing or hiding a block does anything differently with it. A
translation must: a sound effect rendered as dialogue is a character shouting
"thud". Send it back on `/api/translate` as `kinds`.

Every block comes back with a small margin around it, and duplicates are dropped.

**`/api/bubbles`** answers with the room each block was written in rather than
the room its words take up: the largest rectangle that fits inside the balloon
*around that block*. This is where a translation goes. Japanese runs down the
page, so a block of it comes back forty pixels across and three hundred tall, and
English set in a column that shape wraps to about a letter a line.

**The endpoint takes a list rather than one box at a time, and the answer for a
block depends on which others share its balloon.** Where several are in one
balloon their answers are cut back to a cell each; a box asked about alone is
handed the whole balloon. So send every box on the page whenever the answer is
going to be used for lettering.

`bubble` is null where the block is in no balloon — a sound effect over artwork
is in none — and then the box is all there is to go on. Saying so beats answering
with a rectangle somewhere in the artwork.

*How the room is measured, and what a flood fill got wrong, is in `DOCS.md`.*

**`/api/letters`** answers with the lettering itself rather than the box around
it: a page-sized PNG, opaque white on the ink and clear everywhere else. Sent on
to `/api/clean` it hides the words and leaves the art they were drawn over, which
a rectangle cannot do.

`grow` (default 4, up to 64) is how far to spread the mask, so no halo is left
ringing a hidden letter. Two is enough on clean black-on-white lettering; what
needs more is scanned material, and how much more depends on the scan — which is
why the dashboard puts it next to the button rather than deciding for you. **It
is measured in the detector's pixels rather than the page's**, and so means the
same thing at any resolution the page was scanned at.

**`/api/read`** says what it says: one string per box, in the order the boxes
were given, so they line up with the regions `/api/detect` returned. Reading is
a separate call because the boxes are worth correcting first — a box that clips
half a bubble reads half a sentence. A box too small to hold lettering comes
back as `""`. It translates nothing. `language` is which reader stands up, and
only the one asked for ever does.

**`/api/survey`** is the one call here about a chapter rather than a page, and it
is the answer to the thing a page-at-a-time run cannot do for itself. A chapter is
translated in order, so page three is translated long before page forty has been
read — and page forty is usually where a chapter gets round to saying who someone
is. Send the whole chapter's lettering through this first and every page can be
translated against all of it.

`pages` is a JSON list of lists: one list of lines per page, in reading order, as
`/api/read` gave them. A page with nothing on it is an empty list rather than a
page left out — the answer is one `beat` per page and they are positional, so a
page dropped on the way in puts every beat after it one place wrong.

A chapter of raw lettering does not fit a context window, so this takes a
windowful at a time. Send a few pages; send what came back as `chapter` with the
next few, and `first` saying which page the window starts at. The last window is
the one that has read the lot, and its answer is the one to keep — there is no
separate consolidation pass.

`chapter` comes back as `{synopsis, register, beats, cast, terms}`. `synopsis` is
the chapter whole, `register` is how it is written, and `beats` is one line per
page of *this* window, so a caller lays them down at `first`. `cast` and `terms`
are the same shapes `/api/translate` answers with. Everything is capped — a
synopsis to 1200 characters, a register to 200, a beat to 160, the cast to twelve
with 200 characters describing each of them, and the terms to forty — and cut
rather than refused.

The beats are counted the way the translations are, and a second miscount hands
back *no* beats rather than beats a page out. What the window said about the
chapter is kept either way. Nothing is kept here: the caller carries the chapter
from window to window, and back in on `/api/translate`.

**`/api/translate`** is the one thing here that works on words alone — no image
goes with it. `texts` is a JSON list, `model` is one of the names `/api/models`
gives back, `target` is the language to translate into (English unless said) and
`source` the one the page was lettered in (Japanese unless said). Both are words
rather than codes: they are only ever words in a prompt, and a caller may well be
translating something this API has no reader for. The whole page goes over in one
request rather than one line at a time, held to a JSON schema so the answers come
back countable; a page that miscounts is shown its own answer and asked again,
and only a second miscount falls back to one line at a time. One translation
comes back per text, in order; a text that was empty stays empty.

`kinds` and `budgets` are what a caller knows about each line that the model
cannot see. `kinds` is `speech` or `free` from `/api/detect`, or `""` for a block
nothing classified. `budgets` is roughly how many characters fit where the
translation is going to be lettered. Both are optional; both are one per text in
the same order, and a list that is not is **refused** rather than lined up wrong.
Neither is ever enforced — a line over its budget is still lettered, smaller.

What the model is told is the caller's too: send `system`, with `{target}` and
`{source}` wherever the languages should go, and `GET /api/prompt` hands back the
default to start from — as `prompt`, alongside `survey`, which is the one
`/api/survey` uses. Nothing is kept; every request carries its own. The two are
not interchangeable: a survey briefed to translate translates its window instead
of reading it.

`terms` comes back beside `texts`: the names, places, honorifics and coinages
this page introduced, each with the wording just used for it. Send them back as
`glossary` — a JSON list of `{"source": …, "target": …}`, optionally with a
`note` — with the next page, and the model is told to render them the same way
again. That is what keeps a chapter consistent across pages no model ever sees
together, and it rides on the request that was going anyway.

`story` comes back the same way and goes out again as `previously`: `scene`, a
sentence or two on where the chapter has got to, and `cast`, who it is going on
between. The scene is *rewritten* each page rather than added to, so a chapter of
forty pages carries two sentences rather than eighty.

A cast entry is `{"name": …, "gender": "male"|"female"|"unknown", "note": …}`.
The `note` is where the character is: their part in the story, who they are to
the others, and how they speak. It is all a page being translated will be told
about anyone on it, so it is carried to 200 characters rather than the 80 a
term's note gets. **`unknown` is the answer wanted until the chapter has actually
shown otherwise**:
a guess made on page one is read back as established fact by every page after it.
The cast is named in the page's own script (`先輩`, not "the senior") so the name
is a key that holds from page to page, and each page is asked at the foot of its
own text whether anything on it settles whoever is still unknown.

Send `"settled": ["gender"]` on an entry and the model is told that one is not
its to change. That is how a caller says what it already knows.

`chapter` is the other half, and the half that reaches forwards: what
`/api/survey` made of the whole chapter, with `page` saying which of its pages
this is, counting from zero. It does not come back. Send the bible whole — the
API windows the beats around the page itself. Its `cast` and `terms` are ignored
here: they ride as `previously` and `glossary`.

The risk in that is worth stating plainly: **a model shown the ending writes
towards it.** A note goes over with the chapter drawing the line — use it for who
people *are*, not for what *happens* — and the page is told under its own text
how far the reader has got.

The same line twice on one page is asked about once and lettered twice, keyed on
the words *and* the `kind`. Where the two blocks have different room, the tighter
budget is what is sent. For the same reason the repetition penalty is turned off:
a page repeats itself on purpose.

*The measured findings behind all of this — the 先輩 experiment, why the notes are
appended rather than written into the prompt, why the context window is asked for
— are in `DOCS.md`.*

The request asks for a context window rather than taking the one it is given:
Ollama's own default is 4096 tokens and it drops what will not fit rather than
saying so, front first — which is the briefing and the glossary, in a request
that is otherwise a page of dialogue. What that costs is invisible from the
outside: the terms quietly stop being honoured, and a briefing half gone comes
back as a miscounted page.

The model runs under [Ollama](https://ollama.com) on your own machine, and
nothing is sent anywhere else. In a container that machine is not `localhost`,
and what it *is* called is the runtime's to say: Docker answers to
`host.docker.internal`, Podman to `host.containers.internal`. So each is tried
in turn and whichever answers is used, which is why nothing about the host is
baked into the image. Set `MANGA_TRANS_OLLAMA` to say where it is instead — for
Docker on Linux, where neither name resolves unless the container was started
with `--add-host=host.docker.internal:host-gateway`. All three endpoints answer
`503` when nothing is listening at any of them, and say where they looked.

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
drawn over runs on through it. `fill=white` paints it flat instead, which is the
right one where the ground was white to begin with. `fill=telea` is OpenCV's
inpainting — no weights, a tenth of a second, and a legible ghost over anything
hatched. `fill=art` quietly falls back to it when LaMa's weights cannot be
loaded, which an image built with `--build-arg PREFETCH_MODEL=false` may not have.

Filling is LaMa, fine-tuned on 300k manga and anime pages, run on crops around
each mark rather than on the whole page. A 300 dpi page cleans in about fifteen
seconds under three and a half gigabytes.

Two pixels around every mark are kept out of what the fill is made of but are not
themselves painted over — which is also why `grow` on `/api/letters` matters more
here than it did: what the mask does not reach is what the fill is made of.

*Why crops rather than pages, and the memory measurements behind the megapixel
cap, are in `DOCS.md`.*

**`/api/render`** does the same and sets each region's `text` in its box: wrapped
to the width, centred, black, at the largest size that lands inside it. Text too
long for its box is set at the smallest size and left to overrun rather than
dropped. It takes `fill` too, but the default is the other way round — `white` —
because black lettering wants a clear ground. Both endpoints answer with
`image/png`; neither writes anything to disk.

## Options

Everything is set by environment variable:

| variable | default |
| --- | --- |
| `MANGA_TRANS_HOST` | `127.0.0.1` |
| `MANGA_TRANS_PORT` | `8000` |
| `MANGA_TRANS_ORIGIN` | `*` — who may call the API from a browser. Nothing needs it under `compose`, where the dashboard is on the same origin |
| `MANGA_TRANS_REGIONS` | unset — `ogkalu/comic-text-and-bubble-detector`, from the Hugging Face cache |
| `MANGA_TRANS_LAMA` | unset — `ogkalu/lama-manga-onnx-dynamic`, from the same cache |
| `MANGA_TRANS_MODEL` | `~/.cache/manga-trans/comictextdetector.pt.onnx` — the ink mask |
| `MANGA_TRANS_OCR_MODEL` | `kha-white/manga-ocr-base` — the Japanese reader |
| `MANGA_TRANS_OCR_MODELS` | `~/.cache/manga-trans/ppocr` — where every other reader's weights go |
| `MANGA_TRANS_OLLAMA` | unset — `localhost`, `host.docker.internal` and `host.containers.internal` on port 11434 are tried in that order |
| `HF_HOME` | where the reader's weights are cached; `/opt/models/hf` in the container |
| `MANGA_TRANS_FONT` | DejaVu Sans Bold, or Pillow's default |

## The code

```
api/
  mangatrans/
    detect.py     the balloons and the lettering (RT-DETRv2, onnxruntime),
                  and the per-pixel mask of the ink (comic-text-detector's
                  segmentation head, on OpenCV's ONNX backend)
    bubble.py     the room inside a balloon a block was written in, which is
                  where a translation goes — the block is only where the
                  words are
    languages.py  what a page can be written in, and what that means for
                  reading it: which reader, which way round, how it joins
    read.py       manga-ocr and PP-OCR, and the cropping that feeds them
    ollama.py     translating a page by a model on this machine
    inpaint.py    making what was hidden out of the page around it: LaMa,
                  and Telea where there are no weights for it
    render.py     hiding the old lettering, fitting and setting the new
    server.py     the API itself
    geometry.py   Box
  tests/
web/
  src/            the dashboard: drop pages in, detect, read, mask, clean
  Dockerfile      built once, then served by nginx with /api proxied through
  default.conf.template  nginx, resolving the API afresh rather than once
docker-compose.yml  both halves, on one origin
```

`DOCS.md` is the section-per-module reference: how each of these works and why it
is that way. `CLAUDE.md` is the short list of what must not be broken.

## The dashboard

Pages go in by dropping them, pasting them, or picking them — and a chapter can
go in as a `.zip` or `.cbz`, which is opened in the browser and comes out in the
order the names put it, counting properly: page 2 before page 10.

An archive arrives as a **folder** in the rail. The same archive dropped twice
fills the same folder, and pages are only duplicates of each other within one.
**New folder** starts an empty one, which pages dropped while it is open land in;
an archive still makes its own. Deleting a folder deletes its pages, and deleting
the last page deletes an archive's folder but not a hand-made one.

A folder is run in two steps, because a chapter is worth reading before it is
translated. **Read chapter** puts every page through detect and read, then reads
the chapter itself out of what those pages said. Nothing is hidden yet: hiding is
the slow half, and this way what the chapter turns out to be — including who is
in it — arrives without waiting on it. **Clean & translate chapter** then hides
the words on every page and letters it against the whole of it — pressed without
having read the chapter it still works, each page knowing only the pages before
it. Either asks once before doing a folder that has already been lettered.

With no model picked the first reads the pages but not the chapter, and the
second cleans without lettering — which is the whole of what **Read all** and
**Clean all** are.

The bar above the rail says which step is running, which page is in hand and what
is being done to it; clicking the page's name puts it on the board. **Stop** stops
after the page in hand. A page that falls over is named with its reason and the
rest carry on.

The gap between the two steps is the point of splitting them, and splitting them
at the clean is what makes the gap cheap to reach: what the chapter turned out to
be is shown under the folder's buttons and can be put right there, and a name
corrected *there* is corrected on every page. So is the cast — who the chapter is
about, what they are to each other and how they speak — which is what a page
being translated knows about anyone on it. The page-by-page beats are shown too,
and are the one part that is not editable. Delete or add a page after reading the
chapter and the beats no longer line up, so the rail says so and translating
falls back to knowing only what came before.

A run also builds the folder's **chapter terms**, so a character keeps one
spelling across pages the model never sees together; the first rendering of a term
wins. **Forget** starts it over.

**Download chapter** hands the whole folder back as one archive, the way it
arrived: `ch01.cbz` comes back as `ch01-english.cbz`, or `ch01-cleaned.cbz` where
it was only cleaned. *Every* page goes in, at the best state it reached, under the
name it came in under. It can be pressed whenever there is something to save, and
pressed again — a chapter only read has nothing to save yet, so it waits for the
second step.

A page is worked on the board in three tabs. **Inspect** boxes the lettering and
reads it; **Page is in** beside the blocks is the language, remembered for next
time. A block the detector is less than 80% sure of starts left alone. **Mask**
marks the lettering itself for hiding, brushable by hand, with **Hide under**
choosing between the art, no model, and flat white. **Translate** sets each line
in the balloon its original was written in, in Anime Ace, as large as it will go
there but no larger than the page is lettered; the box can be dragged, pulled,
sized with the up and down arrows and turned with the left and right ones or by
the handle above it. **Fit** sets one line at the largest size that lands in its
box. **Translate again** runs the page over against the blocks as they stand.
**Apply to image** draws the lot into the page and saves it, with the same font,
sizes and wrapping shown on the board.

*What each of those decisions costs, and why, is in `DOCS.md`.*

Translating needs Ollama running with a model pulled:

```bash
ollama pull gemma4:12b
podman run --rm --init -p 8000:8000 manga-trans   # finds Ollama on the host
docker run --rm --init -p 8000:8000 manga-trans   # so does this
```

The dashboard is a Vite app. `cd web && pnpm install && pnpm dev` serves it on
port 5173, pointed at `http://localhost:8000` unless `VITE_API_URL` says
otherwise.

## Tests

In the container, with the working copy mounted in so a change needs no rebuild:

```bash
podman run --rm --network none --entrypoint python -w /app \
    -v "$PWD/api/mangatrans:/app/mangatrans:ro" -v "$PWD/api/tests:/app/tests:ro" \
    manga-trans -m unittest discover -s tests -t .
```

They stub every model — the two detectors, the reader and the fill — so they need
no weights, no network and no torch, which is what `--network none` above is
there to keep true.
