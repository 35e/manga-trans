# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`README.md` documents every endpoint, its form fields and its behaviour, plus the
full environment-variable table. Read it before changing anything in the API's
surface — it is kept accurate and should stay that way.

## Commands

Python runs in a container, never on the host. The image bakes the code in, so a
running container ignores edits until it is rebuilt — mount the working copy
instead when iterating.

```bash
podman compose up --build                             # both halves, localhost:8080
podman build -t manga-trans api                       # ~2.5 GB, bakes every model in
podman run --rm --init -p 8000:8000 manga-trans       # http://localhost:8000/api

# Tests, with the working copy mounted so no rebuild is needed:
podman run --rm --entrypoint python -w /app \
    -v "$PWD/api/mangatrans:/app/mangatrans:ro" -v "$PWD/api/tests:/app/tests:ro" \
    manga-trans -m unittest discover -s tests -t .

# One class or one test — same mounts, last argument changes:
    manga-trans -m unittest tests.test_mangatrans.TestOllama
    manga-trans -m unittest tests.test_mangatrans.TestApi.test_clean_hides_the_boxes
```

`docker` takes the same commands. Tests stub every model — `StubRegions`,
`StubLetters` and `StubReader` patched onto `server.Regions`/`server.Letters`/
`server.Reader`, `StubLama` patched onto `inpaint.Lama` for the whole module (a
patch around building the client would lift before the request that reaches for
it), and `mock.patch.object(ollama, "ask", ...)` for translation — so they need
no model, no network and no torch.

Front end, from `web/`:

```bash
pnpm install
pnpm dev        # 5173, talks to http://localhost:8000 unless VITE_API_URL says otherwise
pnpm build      # tsc -b && vite build
pnpm lint       # oxlint
```

There are no front-end tests.

## Architecture

Two halves that only meet over HTTP: `api/` is a stateless Flask service, `web/`
is a Vite/React dashboard that holds all the state. Nothing is persisted
anywhere — every request carries the page it works on, and the browser is the
only place a page, its blocks, its mask or its translations live.

The API deliberately never renders end-to-end. Detection is imperfect, so
`/api/detect` (where the text is), `/api/bubbles` (what room it was written in),
`/api/read` (what it says), `/api/clean` (hide it) and `/api/render` (set new
text) are separate calls taking boxes back in, which is what lets the dashboard
show the detection, have it corrected by hand, and only then act on it.

Under `docker-compose.yml` the two are served on **one origin**: nginx hands out
the built dashboard and proxies `/api` through to the API container, so
`VITE_API_URL` builds as `''` and no browser ever makes a cross-origin call.
`MANGA_TRANS_ORIGIN` and the CORS headers still exist for anything talking to
port 8000 directly. Keep `lib/api.ts`'s fallback (`?? 'http://localhost:8000'`)
nullish-coalescing rather than `||`, or the empty base collapses back to the
absolute one and same-origin stops working. nginx needs `client_max_body_size`
raised past its 1 MB default, and its proxy timeouts raised past 60 s: a page and
its mask together run to tens of megabytes, and reading one takes seconds.

**nginx must re-resolve the API, and that is why its config is a template.**
Written `proxy_pass http://api:8000`, nginx looks the name up once as it loads
and keeps that address for the life of the process. The API container gets a new
address every time it is recreated — a crash, a `compose up`, a restart — so from
then on nginx holds a dead one and answers **everything** with 502, long after
the API is back and healthy; measured, 10.89.0.4 to 10.89.0.183 across one
restart, and every call 502 until nginx itself was restarted. A name held in a
variable (`set $upstream`) is looked up per request instead, which needs a
`resolver`, which is why `default.conf.template` goes to `/etc/nginx/templates/`
and the image's entrypoint fills `${NGINX_LOCAL_RESOLVERS}` in from the
container's own `/etc/resolv.conf`. That entrypoint script returns early unless
`NGINX_ENTRYPOINT_LOCAL_RESOLVERS` is set, and `NGINX_ENVSUBST_FILTER` keeps
envsubst off nginx's own `$host` and `$remote_addr`. A `proxy_pass` with a
variable in it does not pass the path on by itself, hence the explicit
`$request_uri`. Do not "simplify" this back to a plain hostname.

### API (`api/mangatrans/`)

`server.py` is the whole HTTP surface; the rest are libraries it composes.
Models load lazily behind a lock on first use and are shared thereafter — an API
only ever asked to detect never stands the OCR reader (and torch) up at all.
`Regions`, `Letters`, `Reader` and `Lama` each hold their own lock because
neither an OpenCV net, an onnxruntime session nor torch generation is reentrant.

**`Regions.run` and `Letters.run` each keep the last page's pass** behind that
same lock. The ink pass is ~2.5 s and everything downstream of it is under a
millisecond, so the dashboard's ordinary flow — `/api/detect` then
`/api/letters`, then `/api/letters` again on every change of spread — is one pass
of each rather than several. Keep any new work that needs the segmentation on
this path rather than adding a pass.

- `detect.py` — two models, for two different questions.
  `Regions` is comic-text-and-bubble-detector, an RT-DETRv2 on onnxruntime:
  one pass, three classes (`bubble`, `text_bubble`, `text_free`), ~0.2 s. Its
  export carries RT-DETR's own postprocessing, so it answers with `labels`,
  `boxes` and `scores` — already corner-to-corner and already in the page's
  pixels — rather than logits to threshold. It takes `orig_target_sizes` as
  **(width, height)**; the other way round returns the same balloons transposed.
  `Letters` is comic-text-detector, kept for its segmentation head *alone* — the
  per-pixel ink map behind `/api/letters`, which is what lets a clean take the
  words off the art. Its block head is not used.
- `split.py` — **not in the pipeline.** Kept, with its tests, until real chapters
  confirm the region detector never runs two balloons together; see its docstring.
- `bubble.py` — the room inside the balloon a block was written in, as the
  largest rectangle in it *around that block*. Pure OpenCV on the greyscale page,
  no model of its own: threshold inside the balloon the detector found, paint the
  block in (or vertical Japanese cuts the ground in two), fill the holes, erode a
  margin, then search out from the block on a 128-pixel grid (`holding`). `None`
  is a real answer. `rooms()` then cuts one balloon into a cell per block where
  several blocks share it. `/api/detect` includes it because it is nearly free
  there.
- `languages.py` — the one table of what a page can be written in: which reader
  reads it, which way round it is read, whether its script stacks into columns
  and whether its words are spaced. Both ends look languages up here, the
  dashboard through `/api/languages` rather than by holding a copy.
- `read.py` — manga-ocr for Japanese, PP-OCR (RapidOCR on onnxruntime) for
  everything else, behind one `Reader` that stands up only the language it is
  asked for. **The only thing that imports torch, and it does so inside
  `Reader.load()`.** Keep that import deferred, keep `rapidocr` deferred the same
  way inside `Ppocr.load()`, and keep both out of every other module.
- `ollama.py` — the whole page goes over in one request, held to a JSON schema;
  if the model returns the wrong number of lines it falls back to one line at a
  time. Prompts are never stored: callers send `system` each time.
- `inpaint.py` / `render.py` — hiding and lettering. `render.marked()` turns
  boxes + mask into one greyscale page (white hidden), `render.hidden()` picks
  fill.
- `geometry.py` — `Box`, x1/y1 exclusive, corners normalised on construction.

### Web (`web/src/`)

`App.tsx` owns the state and the async orchestration, keyed by page id:
`analyses` and `lettering` live there, masks, cleaned pages and traced bitmaps
come from hooks. It holds no logic of its own — every edit to a page goes
through `lib/`.

- `lib/regions.ts` / `lib/lettering.ts` — **all** the per-block editing, as pure
  transforms on one `Analysis` / one `Lines`. Add a block operation here, not in
  a component.
- `lib/fit.ts` (measuring and wrapping), `lib/order.ts` (reading order),
  `lib/mask.ts` (the brushed mask), `lib/compose.ts` (the canvas letterer),
  `lib/zip.ts` (a chapter in and back out), `lib/chapter.ts` (which version of a
  page goes into the archive, and what the archive is called),
  `lib/api.ts` (the client, and every shared type).
- `components/Board.tsx` draws the page and switches tools by `mode`; the tool
  rows are `InspectTools`, `MaskTools` and `TranslateTools`, and the overlays
  over it are `RegionsLayer`, `DrawRegion`, `MaskCanvas`, `TranslationLayer` and
  `ViewBar`, one per thing that can be done to a page.
- `components/Sidebar.tsx` is the rail: it owns which folder is open, and holds
  `Gallery` (folder cards, then pages), the folder's own bar and
  `BatchProgress`.
- `components/icons.tsx` holds every line icon; `components/ui.tsx` every
  control.
- Hooks own one concern each: `useImageLibrary`, `useBatch`, `useMasks`,
  `useLetterMasks`, `useObjectUrls`, `useOllama`, `usePrompt`, `useLanguage`,
  `useBoardView`, `useBoardKeys`, `useBoxDrag`, `useFileDrop`,
  `useLetteringFont`.

## Invariants worth knowing before editing

**Per-page arrays are positionally aligned.** `analysis.detection.regions`,
`analysis.texts`, `analysis.excluded` (indices) and `lettering[pageId]` are all
indexed by the same block position. Anything that inserts, moves or splits a
block must carry every one of them plus `selected`. Do not write that by hand:
`lib/regions.ts` and `lib/lettering.ts` are the only places that edit these, and
they are paired — `blocks.inserted` goes with `lines.inserted`, `blocks.moved`
with `lines.moved`, `blocks.split` with `lines.split`. Anything **asynchronous**
must instead re-find its block by `region.id` when the answer comes back
(`blocks.withReading`), because the list may have been reordered while the
request was in flight.

**A block is not where the translation goes.** `region.box` is where the
Japanese is; `region.bubble` is the room it was written in, and lettering uses
`bubble ?? box` (`lines.roomFor`). Anything that changes a box by hand must drop
the bubble with it — a stale one letters into the balloon the block came from —
and ask `/api/bubbles` again if it is going to the API anyway. `blocks.withBox`
already does the dropping. Size is capped at `originalSize` (`lib/fit.ts`): a
balloon is drawn around its words, so filling one sets a short line four times
too large.

**Reading order is defined twice and must agree.** Down the page, then across it
the way the language is read — `(y0, -x1)` right to left, `(y0, x0)` left to
right — in `detect.py` (`blocks.sort`, from `Detector.__call__`'s `rtl`) and in
`lib/order.ts` (`key`, from the same flag). Both take it from
`languages.Language.rtl`, the API directly and the dashboard through
`useLanguage`. A hand-drawn block is inserted where that rule puts it, not
appended, because the order is also the order the page is translated in as one
conversation.

**The language decides three things and nothing else.** Which reader stands up;
which way the blocks are sorted; and what `/api/translate` is told the page is in
(`source`, a word rather than a code — a caller may be translating something
there is no reader for). Detection, the segmentation mask, the balloon-finding
and the splitting are all language-blind and must stay that way: the detector was
trained on comics rather than on a script, and `split.py` measures in characters,
which holds for lines running down a page and across it alike.

**PP-OCR reads across a page and nothing else**, so `read.pieces` takes a
balloon apart before it goes over — into lines, or into columns each set out as a
line by `read.unstacked`. Which way the lettering runs is measured off the ink of
the block (`read.upright`, the shape of what was set) rather than taken from the
language: a page of Korean carries a sound effect written down the side of it
just as a page of Japanese does. The gaps look like the better signal — line
spacing against character spacing — and are not: CJK is set solid both ways and
the air between two columns is the letterer's taste. `read.inked` decides ink
from ground at the *edge* of the crop rather than by taking the rarer of the two,
which gets a heavy sound effect exactly backwards.

**`rapidocr` asks for `opencv-python`, the GUI build.** It is the same cv2 as the
headless one everything here uses, linked against a libGL no server has, and
being installed second it wins. The Dockerfile drops it and lays the headless
build back down afterwards; do not "simplify" those three lines into one `pip
install`.

**onnxruntime hunts for a GPU as it loads**, and handed a card whose make it
cannot read — which is every container on a Mac — says so at warning level, from
C++, straight to the descriptor. There is no logger to turn down and no env var
for it: the severity can only be raised from Python once the environment exists,
by which time the line has been written. So `read.quieted()` catches fd 2 for the
length of a reader's load and writes back everything that is *not* that line. Not
a blanket silencer, and it must not become one: a reader that cannot find its
weights says so the same way, and that has to get out.

**Ollama runs on the machine, not in the container, and Docker and Podman
disagree about what that machine is called** — `host.docker.internal` against
`host.containers.internal`. Naming either one in the image leaves the other
unresolvable, which shows up as a page that reads perfectly and then will not
translate, so `ollama.answering()` tries each in turn and keeps the first that
answers. Keep the host out of the Dockerfile. A miss is deliberately not
remembered: Ollama is as often started after the dashboard as before it.

**There are two independent letterers.** `/api/render` sets text with PIL
(`render.py`: binary search for the largest fitting size, greedy wrap). The
dashboard's "Apply to image" sets it with canvas (`lib/compose.ts`), sharing
`lib/fit.ts` with the on-board preview so what is arranged is what comes out —
same font, sizes, wrapping and hyphenation. They are not expected to produce
identical output; `/api/render` is for callers that are not this dashboard.
`lib/fit.ts` must be awaited via `ready()` before measuring, or it measures the
fallback font.

**Masks.** Greyscale, page-sized, white hidden, greys partial. `server.mask_in`
believes an alpha channel only when some of it is actually clear, since a
browser canvas always exports one — do not "simplify" that check. `lib/mask.ts`
exports white-on-black for exactly this reason.

**A block is marked into a mask by the lettering in it, never by its box.**
`mask.mark` falls back to stamping the whole rectangle when it is handed no
tracing, and that fallback is for the "Blocks" button and for a tracing that
failed — not for the ordinary path. So anything marking a block after the mask
has been seeded goes through `App.markLetters`, which asks for the tracing if it
is not already held: a spread changed since, or a page whose tracing was dropped
at the end of a folder run, otherwise leaves a block put back by hand or one
drawn where the detector missed one cleaning out the whole square that was drawn.

**`fill` defaults differ by endpoint**: `art` for `/api/clean`, `white` for
`/api/render`. `art` is LaMa, `telea` is the same idea without a model, and
`art` falls back to `telea` when LaMa's weights cannot be loaded — `server.optionally`
remembers the miss rather than retrying per request, because it is a missing file
rather than anything that might be there next time. The painter is handed *in* to
`render.hidden`, the same way the detector and the reader are, so standing it up
stays the caller's business.

**A folder run goes one page at a time, and the page in hand is the page named.**
`useBatch` sends one page through `App.pipeline` and waits. Sending five at once
buys nothing — `Detector` and `Reader` hold a lock each, so they queue at the API
regardless — and coming back in no particular order leaves the progress bar with
nothing true to say. Which is also why board actions are shut while a run is going
(`Board`'s `runningFolder`): a second page in the air would have the bar naming
whatever was asked for last.

`pipeline` therefore takes the page rather than reading the active one, and moves
the step tabs and clears the selection **only** for the page on the board — a run
must not drag the view about under someone reading another page. It hands back why
it gave up rather than only leaving it in the banner (`App.lastFailure`, set by
`during`), because a run has to say which page that was. Not every empty answer is
a refusal: `translatePage` also comes back false for a page whose every block was
left alone, so the reason is what decides, not the boolean.

**A tracing is dropped as soon as its clean lands, unless its page is on the
board.** It is a page-sized `ImageBitmap` worked out from the page and so can be
had again; fifty pages run through would otherwise hold fifty of them alongside
fifty masks and fifty cleaned pages. The page being brushed keeps its own, that
being the one that will want it again.

**A chapter is packed on the click, not during the run.** `App.downloadFolder`
draws every page afresh out of `lettering` and `cleanedPages`, which is what lets
it be pressed after a whole run, after one stopped part way, or an hour later,
and pressed again — and it means a run itself holds nothing extra. Pages go
through **one at a time**: each is composed at the page's own resolution, and
forty of those in flight together is forty page-sized canvases for no gain, since
the work is the same either way.

**Every page goes into the archive, at whatever state it reached**
(`chapter.finished`): lettered, else cleaned, else the original bytes under the
original name. A page that will not compose falls back to its original rather
than being skipped — a chapter with a gap in it is not a chapter, and dropping a
page renumbers every page after it. Anything drawn on is a PNG and says so;
anything untouched is passed through byte for byte, since re-encoding a JPEG
nobody touched costs quality for nothing.

**Nothing in the archive is compressed** (`zip.pack`, `level: 0`). Every entry is
already a PNG or a JPEG: deflating one a second time takes seconds over a chapter
and gives back about a percent. `pack` also numbers a repeated name rather than
letting it overwrite — pages are told apart by name *and* size *and* date, so one
folder really can hold two files called `001.png`, and a record keyed by name
would silently ship one of them.

**`GalleryFolder.archive` is the only record of what a chapter arrived as.** The
folder is named after the archive's *stem*, so without it nothing says whether it
was a `.zip` or a `.cbz`, and `archiveName` could not hand back the kind that came
in. `folderFor` still matches on the stem, so re-dropping fills the same folder
and the extension that named it first is the one that sticks.

**A folder is named after its archive, and dedupe is scoped to the folder.**
Re-dropping an archive fills the same folder rather than making a second one, which
is what makes `fingerprint(file, folder)` right both ways round: the same zip twice
is the same five pages, but two chapters each holding a `001.png` hold two pages.

**Traced masks are cached per `(pageId, spread)`** (`useLetterMasks`), and the
`ImageBitmap`s are explicitly `close()`d on removal — a different `grow`/`spread`
is a different tracing, not the same one again. That hook writes its ref before
its state so a tracing can be marked into a mask the moment it arrives.

**Dockerfile layer order is load-bearing.** Only `__init__.py`, `geometry.py`,
`languages.py`, `detect.py`, `read.py` and `inpaint.py` are copied before the
model prefetch step; editing any of those six invalidates ~810 MB of baked
weights and forces a re-download on the next build. Everything else is copied
after. `languages.py` is in that set because `read.ensure_readers` needs it to
know what to fetch, and `inpaint.py` because it fetches LaMa's — both honest
enough, since adding a language or a fill is adding weights.

Keep the heavy imports deferred to first use for the same reason they always
were: `onnxruntime` inside `Regions.__init__` and `Lama.__init__`, torch inside
`Reader.load()`, `rapidocr` inside `Ppocr.load()`, `huggingface_hub` inside the
`ensure_*` functions. A top-level import of any of them puts the cost on every
request, including the ones that never touch that model.

**Debian's package index stalls in a container that carries small files fine** —
a VM on a Mac, most proxies — because it is ~12 MB in one response, and apt
reports the stall as a mirror failure. `/etc/apt/apt.conf.d/99robust` in the
Dockerfile sets retries *and* turns HTTP pipelining off; retries alone do not fix
it, since the retry hits the same stall. `detect.ensure_model` retries its 95 MB
GitHub download for the same reason, and waits between tries: five attempts in a
row all land in the same bad few seconds and buy nothing.

**A merged block is wrong for everything downstream** — it is read as one string,
translated as one line, and lettered into one balloon. That is what `split.py`
was for, and what boxing by *region* rather than by lettering now prevents: the
model is asked where the balloons are at the same time as where the words are, so
two balloons are two answers. If real pages show it merging after all, the fix
belongs in the detector or back in `split.py`, not in the dashboard.

**Order in `Regions.__call__` is load-bearing**: decode, then split the classes,
then `suppressed`, then pad (`PAD`), then sort. Padding before `suppressed` could
make two neighbours look like one; sorting before padding is harmless but sorting
before the classes are split is not, since balloons and blocks are ordered
separately and only the blocks are read.

**Duplicates still have to be dropped.** RT-DETR matches one query to one object
and needs no NMS, but it does sometimes put two boxes over the same lettering.
`detect.suppressed` drops the less sure of any pair covering the same words, and
runs per class — a balloon and the text filling it cover each other almost
entirely and are not duplicates of one another.

**A room always holds the block it is for.** `bubble.holding` searches out from
the block rather than for the largest rectangle in the balloon, so an answer is
the words plus whatever room is around them and never a rectangle somewhere else.
Without that, the largest rectangle is in the wrong part of the balloon as often
as the right one: a balloon with a tail, or one drawn round two lines with the
words in one of them. Everything downstream leans on this — the sharing out below
only crops, `lines.roomFor` letters into whatever comes back, and there is
nothing else on the page saying where the words belong.

**The balloon's own box is not the room.** A balloon is an oval and a line set to
the corners of its bounding box runs outside the outline, so `bubble.inside`
thresholds the interior *within* the detected box and measures the largest
rectangle in that. The block is painted in first (`shrunk`), or a column of
Japanese down the middle cuts the ground in two and the answer is the gap beside
the words. Light ground first, then inverted, for a shout set white on black.

**Blocks are grouped by the balloon they are in, not by whether their answers
collide.** That collision test existed because a balloon was flooded out from the
words and one balloon came back as a different rectangle from each block in it —
measured, two blocks in one balloon agreed only 0.54. The detector now says which
balloon is which, so `bubble.assigned` asks the far simpler question: which
balloon holds this block (`HELD`), smallest first so a shout drawn inside a
thought wins over the one around it. A block no balloon holds keeps its own box.

Where several blocks share one balloon, `bubble.divided` cuts a cell apiece and
each answer is cropped to its own. That division **recurses** — cut at the widest
blank on whichever axis it is widest, then each side again — because blocks set
two across and two down are not in a row, and one line of cuts gives the two on
the right a left and a right half of a balloon they are stacked inside. Every
block keeps *its own* answer cropped, never a share of a neighbour's: handing the
group one balloon and cutting that up puts a translation on the far side of the
page from its Japanese. A cropped answer that no longer holds its block is
refused; that only happens where the blocks themselves overlap, and there the box
is the honest answer.

**Cleaning is LaMa, and the seam is not.** `inpaint.fill` grows the hole by `EDGE`
for *sampling* only and then alpha-composites the result back through the
caller's ungrown, greyscale mask, so a soft brushed edge blends rather than steps
and the rim of half-ink just outside a letter is never read as art. That is
independent of which painter made the pixels, and it stays that way. A page
marked all over short-circuits to white **before** the painter: there is nothing
left to make a fill out of, and a model handed a page that is entirely hole does
not say so, it invents one. LaMa is run on crops around each mark rather than on
the page (`inpaint.patches`) — a page is mostly art that is staying — and marks
closer than `APART` go through together, or the context around one letter holds
the next as material to copy it from. Its input must be a whole multiple of
`BLOCK`; a size that is not fails inside the graph rather than being padded.

**A crop over `inpaint.LARGEST` is worked out smaller and stretched back**, and
this is what keeps the API alive rather than what makes it quick. LaMa's cost is
in its Fourier layers, which hold whole feature maps: measured, a 0.6 MP crop
peaks near 2.2 GB and a 1.5 MP one near 3.9 GB, so a balloon on an A4 page
scanned at 300 dpi (8.7 MP — an ordinary chapter) asks for something like 9 GB
and the kernel kills the process. There is no exception to catch and nothing in
the API's own log; the container simply goes and a front end sees a **502**, and
`restart: unless-stopped` brings it back so cleanly that `RestartCount` is still
0. If a big page ever 502s again, look for `oom-kill` in `podman machine ssh
sudo dmesg` before anything else. Working smaller costs almost nothing: only the
marked pixels are kept, lettering is thin, and the fill is the tone and lines
around it rather than any detail of its own.

**`grow` is in the detector's pixels, not the page's** (`detect.GROW`,
`page_mask`). The mask is worked out on a 1024-square canvas and stretched, so
its edge is only accurate to a canvas pixel — one page pixel on a small page,
three and a half on a 300 dpi scan. Held in page pixels the same `grow` is three
canvas pixels of allowance on one and barely one on the other: measured, 100% of
the lettering covered at 1000x1400 against 94.5% at 2480x3508, which comes out as
the feet of every letter still on the page after a clean. Anything else measured
against the mask belongs in the same units.

**An answer from `/api/bubbles` depends on which other boxes were asked about**,
so anything whose answer will be lettered with must send *every* box on the page,
not just the ones that changed. A box asked about alone is handed the whole
balloon — correct in isolation, and on top of its neighbours in practice. This is
why `reread` in `App.tsx` reads only the changed blocks (the slow half) but asks
for balloons for all of them, and applies the result to every region by id.

## Style

Comments are for *why*, and they are short. A module or function gets a line or
two saying what it is for and what would go wrong done the obvious way; anything
longer has to be earning it — the non-obvious threshold, the ordering that
matters, the alternative that was tried. Do not restate the code, and do not
explain the domain here: that is what this file is for. Names carry the rest.

Python tests are named as statements of behaviour
(`test_the_far_edge_is_exclusive`).
