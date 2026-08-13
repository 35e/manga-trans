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
it, as below, because it is worked out from the page and the boxes and both are
already in hand. `language` changes nothing about what is found, only the order
it comes back in.

`kind` comes with it too: `speech` where the lettering is inside a balloon and
`free` where it is not — a sound effect, a caption, a sign, a shout across the
art. The model is asked both questions in the one pass, so it costs nothing here,
and nothing about finding, tracing or hiding a block does anything differently
with it. A translation must: a sound effect rendered as dialogue is a character
shouting "thud". Send it back on `/api/translate` as `kinds`.

Two balloons that overlap used to be boxed as one, and that is worse than it
looks: the block is read as one string, so two speakers reach the translator as a
single line and the lettering that comes back is set into one balloon. The model
boxes by *region* rather than by lettering — it is asked where the balloons are
at the same time as where the words are — so two balloons are two answers, and
the gap-measuring that used to cut a merged block back apart is gone with the
problem it was for.

The detector still sometimes draws two boxes over the same lettering. A second
pass drops any block covering the same words as a surer one, since a duplicate is
read twice, translated twice and lettered twice into one place.

Every block comes back with a small margin around it, because the head boxes
lettering tightly and sometimes clips the edge of a glyph — enough to hold the
whole of what it found and to cover the letter when a block is cleaned by its box
rather than by its traced ink.

**`/api/bubbles`** answers with the room each block was written in rather than
the room its words take up: the largest rectangle that fits inside the balloon
*around that block*. This is where a translation goes. Japanese runs down the
page, so a block of it comes back forty pixels across and three hundred tall, and
English set in a column that shape wraps to about a letter a line — which is why
a translated line used to have to be dragged out to its balloon before it could
be read at all.

The balloon is *detected*, not guessed at. That is the difference this turns on.
It used to be flooded outwards from the words — a balloon is a light shape closed
by a dark outline, so the light pixels reachable from the lettering are the
balloon — and that works until the outline has a tail to escape down, or a scan
has broken it into the panel beside it, at which point the answer is a rectangle
somewhere else on the page. Now the model says which shape is the balloon, and
the measuring only ever happens inside it.

Inside it, the balloon's own box is still not the answer: a balloon is an oval,
and a line set to the corners of its bounding box runs outside the outline. So
the inside is thresholded within that box and the largest rectangle in it that
still holds the block is what comes back. The block is painted in first, because
a line of Japanese down the middle of a balloon cuts its ground into a left half
and a right half and a measurement started in one of them describes the gap
beside the words rather than the room around them. Dark balloons with white
lettering are found the same way round the other way.

Around the block, not simply the largest rectangle in the balloon: an answer
always holds every pixel of the box it was asked about, and only ever says how
much wider or taller the room around the words runs. A translation set anywhere
else is one the reader has to go looking for.

Where several blocks turn out to be in the *same* balloon — a balloon holding two
separate lines of dialogue — their answers overlap, and each is cut back to its
own side rather than left running across the others. The page is cut at the
widest blank between them, on whichever axis that blank is widest, and then each
side again until every block has a cell to itself; every answer is then cropped
to the cell of the block it was measured around. Without that, two of them would
be lettered one on top of another. Cutting once along one axis is not enough,
which is why this recurses: four blocks set two across and two down are not in a
row, and one line of cuts hands the two on the right a left half and a right half
of a balloon they are stacked inside.

Each block keeps its own answer through this, cropped — never a share of a
neighbour's. Handing a whole group one balloon and cutting *that* up is what
sends a translation to the far side of the page.

This is why the endpoint takes a list rather than one box at a time: the answer
for a block depends on which other blocks share its balloon. Ask about a box on
its own and its answer is left uncropped, since nothing else is known to be in
the balloon with it — so send every box on the page whenever the answer is going
to be used for lettering.

`bubble` is null where the block is in no balloon, and then the box is all there
is to go on: a sound effect over artwork is in none, and a balloon drawn no wider
than the words already are has nothing to offer. Saying so beats answering with a
rectangle somewhere in the artwork.

This call used to stand no model up, and now it stands the region detector up:
where a balloon is is something a model answers, and the boxes sent in are only
asked which of those balloons hold them.

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
is how far to spread the mask, so no halo is left ringing a letter that has been
hidden. On clean black-on-white lettering two is already enough; what needs more
is scanned material — screentone, the ring JPEG leaves around a hard edge, the
pale rim of an outlined letter — and how much more depends on the scan, which is
why the dashboard puts it next to the button rather than deciding for you.

It is measured in the *detector's* pixels rather than the page's, and so means
the same thing at any resolution the page was scanned at. The mask is worked out
on a 1024-square canvas and stretched to the page, so its edge is only ever
accurate to a canvas pixel — one page pixel on a small page, three and a half on
an A4 scan at 300 dpi. Counted in page pixels the same `grow` would be a
generous allowance on the one and almost none on the other, which comes back as
the feet of every letter still on the page after a clean.

**`/api/read`** says what it says: one string per box, in the order the boxes
were given, so they line up with the regions `/api/detect` returned. Reading is
a separate call because the boxes are worth correcting first — a box that clips
half a bubble reads half a sentence. A box too small to hold lettering comes
back as `""`, and the box is given a few pixels of air before it goes to the
model. It translates nothing: what the text should say instead is still for the
caller to decide. `language` is which reader stands up — see above — and only
the one asked for ever does.

**`/api/translate`** is the one thing here that works on words alone — no image
goes with it. `texts` is a JSON list, `model` is one of the names `/api/models`
gives back, `target` is the language to translate into (English unless said) and
`source` the one the page was lettered in (Japanese unless said). Both are words
rather than codes: they are only ever words in a prompt, and a caller may well be
translating something this API has no reader for. The whole page goes over in one
request rather than one line at a time:
that is both far quicker and better translation, since a line of manga read on
its own often cannot be translated at all, having no idea who is speaking or
about what. The model is held to a JSON schema so the answers come back
countable. A page that comes back miscounted anyway is shown its own answer and
asked again — still the whole page, still every line read against the ones around
it — and only a second miscount falls back to one line at a time, which cannot
miscount and is the worst translation this can produce. One translation comes
back per text, in order; a text that was empty stays empty.

`kinds` and `budgets` are what a caller knows about each line that the model
cannot see. `kinds` is `speech` or `free` from `/api/detect`, or `""` for a block
nothing classified. `budgets` is roughly how many characters fit where the
translation is going to be lettered — worth saying while the words are still
being chosen, since a line too long for its balloon is not refused anywhere, it
is set smaller until it fits, and a page of that is a page nobody can read. Both
are optional; both are one per text in the same order, and a list that is not is
refused rather than lined up wrong.

What the model is told is the caller's too: send `system`, with `{target}` and
`{source}` wherever the languages should go, and `GET /api/prompt` hands back the
default to start from. Nothing is kept — every request carries its own — so a front end
that wants its own prompt remembers it and sends it each time, which is what the
dashboard's settings do.

`terms` comes back beside `texts`: the names, places, honorifics and coinages
this page introduced, each with the wording just used for it. Send them back as
`glossary` — a JSON list of `{"source": …, "target": …}` — with the next page, and
the model is told to render them the same way again. That is what keeps a chapter
consistent across pages no model ever sees together, and it rides on the request
that was going anyway rather than a second one, which over forty pages would be
forty more calls. Nothing is kept here either: the caller collects the terms and
sends them on, the same as the prompt.

Bear in mind that the answers are held to a JSON schema, and a model under a
schema follows a prompt loosely. Asking for a voice — casual, formal, terse —
comes through; asking it to reformat what it returns mostly does not. That is why
the sentence asking for `terms` is added by the API to whatever `system` says
rather than living in the default prompt: replace the prompt and the glossary
goes on working.

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
drawn over runs on through it, a panel line that went under a letter comes out
the other side, and the edge of a bubble joins back up. `fill=white` paints it
flat instead, which is the old behaviour and still the right one where the
ground was white to begin with — the inside of a bubble, or a box about to have
new lettering set in it.

Filling is LaMa, fine-tuned on 300k manga and anime pages. It has seen line art,
so it continues a hatched edge and a screentone through the hole rather than
averaging what surrounds it — which is the difference between a clean and a
smudge on anything that was drawn rather than flat.

`fill=telea` is OpenCV's inpainting, which is what this used to do: the marked
pixels are made out of the ones just outside them, working inwards, with no
notion of structure. It costs nothing that is not already installed and runs in
about a tenth of a second against LaMa's few seconds, and on flat tone the two
are hard to tell apart. Over anything hatched it leaves a legible ghost of the
lettering and breaks every line it crosses. It is kept for comparison, and
because an image built with `--build-arg PREFETCH_MODEL=false` may not have
LaMa's weights — `fill=art` quietly falls back to it rather than failing when
they cannot be loaded.

LaMa is not run over the whole page. A page is mostly art that is staying, so the
mask is cut into the pieces that actually have marks in them, each taken with
enough of the art around it to be made out of, and each is put through on its
own. Marks close together go through as one piece: two letters of a word are not
worth two passes, and the art around one of them would otherwise hold the other
as material to copy it from.

A piece larger than about a megapixel is worked out smaller and stretched back.
LaMa's cost is in its Fourier layers, which hold whole feature maps, so it grows
with the area it is given: a balloon on an A4 page scanned at 300 dpi would ask
for something like 9 GB on its own, and the process is killed rather than
answering. Working smaller costs almost nothing here — only the marked pixels are
kept, lettering is thin, and what replaces it is the tone and the lines around it
rather than any detail of its own. A 300 dpi page cleans in about fifteen seconds
under three and a half gigabytes.

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

**New folder** starts an empty one, for pages that did not arrive as a chapter.
It is named as it is made, there being nothing that renames one afterwards, and
opens straight away — because pages dropped while a folder is open land *in* it,
which is the whole point of having made it. An archive is the exception and still
makes a folder of its own: a chapter inside a chapter means nothing here. A
hand-made folder is kept when its last page is deleted, having been made empty in
the first place, and the first archive dropped into one tells it what to come back
out as.

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

A run builds the folder's **chapter terms** as it goes: each page hands back the
names, places and coinages it introduced, and every page after it is translated
against them, so a character keeps one spelling across pages the model never sees
together. They are listed under the folder's buttons — the first page to name
someone settles it for the rest of the chapter, and there would otherwise be
nothing saying why a later page came out as it did. **Forget** starts the list
over; pages already lettered are left as they are. The list belongs to the folder,
is capped at forty, and like everything else here is gone when the tab is.

**Download chapter**, beside it and on the card the run leaves behind, hands the
whole folder back as one archive — the way it arrived. A chapter dropped in as
`ch01.cbz` comes back as `ch01-english.cbz`, named for whatever it was translated
into; one that was only cleaned comes back `ch01-cleaned.cbz`.

*Every* page goes in, at the best state it reached: lettered where it was
translated, cleaned where it was only cleaned, and exactly as it arrived where it
was neither — a page that fell over is still part of the story, and leaving it
out renumbers everything after it. Pages keep the names they came in under, so a
comic reader puts them back in the same order. Anything drawn on comes out as a
PNG; anything untouched is passed through byte for byte rather than re-encoded.

It can be pressed whenever there is something to save — after a whole run, after
one stopped part way, or later in the session — and pressed again as often as you
like, since the pages are drawn afresh each time from what is on screen. Nothing
is compressed on the way in: these are PNGs and JPEGs already, and deflating them
a second time costs seconds over a chapter to save about a percent.

The dashboard puts a page on a board and works it in three tabs. **Inspect**
boxes the lettering and reads it. **Page is in**, beside the blocks, is the
language it is lettered in: it picks the reader, it puts the blocks in the order
that language is read in, and it is what the translator is told the page is in.
It sits on this tab rather than in the settings because this is the step it bears
on, and it is remembered for next time, a chapter being all one language. A block
the detector is less than 80% sure of
is read and listed like any other but starts left alone, since a box over half a
bubble or over a piece of artwork does more harm hidden than a real one does
missed. Putting one back is one click, as is dropping one it was too sure of.
**Mask** marks the lettering itself for
hiding — not the boxes around it — and that mask can be brushed by hand, drawn
wider or erased back, with blocks worth keeping dropped from it one at a time.
**Hide under** beside the brush is what a clean puts back where the marks were:
**The art**, filled in by a model that has seen line art, so a screentone and a
hatched edge carry on through; **No model**, which is the same idea done by
OpenCV and much faster, and fine over flat tone but a smear over anything drawn;
or **White**, flat.
**Translate** sets each translated line in the balloon its original was written
in — not in the box the original came out of, which for a vertical line of
Japanese is a column too narrow to set one word of English across. It is set in
Anime Ace, as large as it will go in that balloon but no larger than the page is
lettered: a balloon is drawn around its words rather than to them, so the largest
type that fits one is far bigger than the type it was drawn around whenever the
line is short, and "OK!" left to fill its balloon comes out four times the height
of the dialogue on either side of it. How large the page is lettered is read off
the original — Japanese is set on a square em, so a block of *n* characters
covering *w × h* was set at about the square root of *wh/n*.

The box can be dragged about, pulled wider or narrower by its edges,
sized with the up and down arrows and turned with the left and right ones — or
by the round handle standing above the box, which follows the pointer round it,
holding shift for 15° at a time. Manga letters plenty of things on the slant: a
sound effect running up the page, a shout across a tilted bubble, and a line set
square over one of those reads as a sticker rather than as part of the art. The
box itself stays square to the page and only what sits in it turns, so a line
still wraps to the width it was given; pulling a turned box by an edge pulls it
along its own axes, and the edge that was not pulled stays where it looks.
**Fit**, against each line in the translations list, sets that one line at the
largest size that lands in its box, held to the size the page is lettered at —
which is the size the arrows deliberately go past.
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
podman run --rm --init -p 8000:8000 manga-trans   # finds Ollama on the host
docker run --rm --init -p 8000:8000 manga-trans   # so does this
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
podman run --rm --network none --entrypoint python -w /app \
    -v "$PWD/api/mangatrans:/app/mangatrans:ro" -v "$PWD/api/tests:/app/tests:ro" \
    manga-trans -m unittest discover -s tests -t .
```

They stub every model — the two detectors, the reader and the fill — so they need
no weights, no network and no torch, which is what `--network none` above is
there to keep true.
