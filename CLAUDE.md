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
podman build -t manga-trans api                       # ~2 GB, bakes both models in
podman run --rm --init -p 8000:8000 manga-trans       # http://localhost:8000/api

# Tests, with the working copy mounted so no rebuild is needed:
podman run --rm --entrypoint python -w /app \
    -v "$PWD/api/mangatrans:/app/mangatrans:ro" -v "$PWD/api/tests:/app/tests:ro" \
    manga-trans -m unittest discover -s tests -t .

# One class or one test — same mounts, last argument changes:
    manga-trans -m unittest tests.test_mangatrans.TestOllama
    manga-trans -m unittest tests.test_mangatrans.TestApi.test_clean_hides_the_boxes
```

`docker` takes the same commands. Tests stub the detector and reader
(`StubDetector`/`StubReader`, patched onto `server.Detector`/`server.Reader`) and
`mock.patch.object(ollama, "ask", ...)` for translation, so they need no model,
no network and no torch.

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

### API (`api/mangatrans/`)

`server.py` is the whole HTTP surface; the rest are libraries it composes.
Models load lazily behind a lock on first use and are shared thereafter — an API
only ever asked to detect never stands the OCR reader (and torch) up at all.
`Detector` and `Reader` each hold their own lock because neither an OpenCV net
nor torch generation is reentrant.

**`Detector.run` keeps the last page's pass** behind that same lock. The forward
pass is ~2.7 s and everything downstream of it is under a millisecond, so the
dashboard's ordinary flow — `/api/detect` then `/api/letters`, then `/api/letters`
again on every change of spread — went from two-plus passes a page to one. Keep
any new work that needs the segmentation on this path rather than adding a pass.

- `detect.py` — comic-text-detector on OpenCV's ONNX backend. Two heads off one
  pass: block boxes, and the per-pixel segmentation mask behind `/api/letters`.
  Fetches its own weights on first use.
- `split.py` — a block holding two overlapping balloons, cut back into one block
  each. Pure numpy/OpenCV on the segmentation mask, which `Detector.__call__`
  already has from the same pass. Thresholds are in characters, not pixels; see
  the note on the deferred import below.
- `bubble.py` — the balloon a block was written in, as the largest rectangle
  inside it *around that block*. Pure OpenCV on the greyscale page, no model:
  flood the light ground from the block (painted in first, or vertical Japanese
  cuts the balloon in two), fill its holes, erode a margin, then search out from
  the block on a 128-pixel grid (`holding`). Every answer is checked and `None`
  is a real answer. `bubbles()` then cuts overlapping answers back to a cell each
  when several blocks turn out to share a balloon. `/api/detect` includes it
  because it is free there.
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
`/api/render`. `inpaint.fill` grows the hole by `EDGE` for *sampling* only, and
still paints just what was marked — the soft rim around lettering must not
become the material the fill is made of.

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

**A folder is named after its archive, and dedupe is scoped to the folder.**
Re-dropping an archive fills the same folder rather than making a second one, which
is what makes `fingerprint(file, folder)` right both ways round: the same zip twice
is the same five pages, but two chapters each holding a `001.png` hold two pages.

**Traced masks are cached per `(pageId, spread)`** (`useLetterMasks`), and the
`ImageBitmap`s are explicitly `close()`d on removal — a different `grow`/`spread`
is a different tracing, not the same one again. That hook writes its ref before
its state so a tracing can be marked into a mask the moment it arrives.

**Dockerfile layer order is load-bearing.** Only `__init__.py`, `geometry.py`,
`languages.py`, `detect.py` and `read.py` are copied before the model prefetch
step; editing any of those five invalidates ~550 MB of baked weights and forces a
re-download on the next build. Everything else is copied after. `languages.py` is
in that set because `read.ensure_readers` needs it to know what to fetch, which
is honest enough — adding a language is adding weights.

This is why `detect.py` imports `split` **inside** `Detector.__call__` rather
than at the top: `split.py` is copied after the prefetch, so a top-level import
would break the build, and moving `split.py` before the prefetch would make every
threshold tweak cost 550 MB. Keep that import deferred.

**A merged block is wrong for everything downstream**, which is why splitting
happens in `Detector.__call__` and not in the dashboard: the block is read as one
string, translated as one line, and lettered into one balloon. `split.pieces`
hands a block back **unchanged** when it does not come apart — it must never
re-box a block it did not cut, or every block on the page would shift.

**The split threshold depends on what the cut stands through**, and that is what
makes a gap this small safe. A cut crossing several lines at once (`GAP`, 0.8
characters) is a strong signal — every line has to fall blank in the same place
at once. A cut along a lone line (`GAP_ALONE`, 1.5) has only that line's own
character gaps to go on, and a column of small kana and punctuation reaches 1.1.
`LINES` (2.5) is which of the two applies. A single threshold cannot do this: at
1.6 it misses balloons a character apart, and anything below ~1.1 shatters that
punctuation column. Tuned by grid search over 21 rendered cases — the binding
constraint at the low end is a balloon whose own lines are set far apart, which
breaks below 0.8. Err shy: a merge can be split by hand in the dashboard, but
there is no way to put a wrongly-cut block back together.

**A narrower gap is still cut on when the two sides are staggered**
(`split.staggered`, `STAGGER`) — shifted the same way at *both* ends. Requiring
both is the whole of it: "the tops do not line up" on its own is worse than
useless, because a balloon of two columns centred against each other is out by
3.0 characters where two genuinely separate balloons are out by 1.0. Nested
means one block (a column that stopped early, columns centred); shifted the same
way at both ends means two. Do not weaken this to a start-only test.

**Order in `Detector.__call__` is load-bearing**: split on tight boxes and the
ungrown mask, then `suppressed`, then pad (`PAD`, a quarter of a character), then
sort. Padding before the split would close the gaps it measures; padding before
`suppressed` could make two neighbours look like one; sorting first would leave
the halves of a cut block out of reading order.

**Splitting a block has two consequences that must be handled together with it**,
and both show up as translations lettered on top of each other:

- *Duplicates.* NMS runs on what the head said, before anything is cut, so it
  never sees the pieces. The head often draws a box round two balloons and
  another round one of them — under the NMS threshold, so both survive — and
  cutting the first makes an exact duplicate of the second. `detect.suppressed`
  drops the less sure of any pair covering the same lettering.
- *A shared balloon.* Two blocks inside one balloon each ask `bubble.around`
  what room they are in and are answered with overlapping pieces of that one
  balloon. `bubble.bubbles` therefore gathers them (`bubble.sharing`) and
  `bubble.divided` cuts the page into a cell apiece, which each answer is then
  cropped to. That division **recurses** — cut at the widest blank on whichever
  axis it is widest, then each side again — because blocks set two across and
  two down are not in a row, and one line of cuts gives the two on the right a
  left and a right half of a balloon they are stacked inside.

**A room always holds the block it is for.** `bubble.holding` searches out from
the block rather than for the largest rectangle in the balloon, so an answer is
the words plus whatever room is around them and never a rectangle somewhere else
on the page. Without that, the largest rectangle is in the wrong place as often
as it is in the right one: a balloon with a tail, one drawn round two lines with
the words in one of them, one whose outline a scan has broken into the panel
beside it — measured, that last one answers with a rectangle holding *none* of
the block it was asked about. Everything downstream leans on this: the sharing
out below only crops, `lines.roomFor` letters into whatever comes back, and there
is nothing else on the page saying where the words belong.

**`sharing` gathers blocks whose answers collide, not blocks whose answers
agree**, and the difference is the whole of it. `around` does not answer with the
same rectangle twice for one balloon: an irregular balloon holds a wide short
rectangle and a tall narrow one of nearly the same area, and which one a block
comes back with depends on where in it that block sits — measured, two blocks in
one balloon come back agreeing 0.54, where the old test wanted 0.7. Anything
keyed on agreement leaves those two lettered on top of each other. `sharing` is
also given the box of any block `around` answered `None` for, because that box is
where the block is lettered and so is what a neighbour's balloon collides with;
and it gathers **transitively**, since A over B and B over C is one balloon
holding three blocks however little A and C touch.

Every block in such a group keeps *its own* answer, cropped to its own cell —
never a share of a neighbour's. Handing the group one balloon and cutting that up
is what puts a translation on the far side of the page from its Japanese: the
odd one out is in a different balloon, and its piece of this one is nowhere near
its words. Only a block `around` answered `None` for borrows (`bubble.borrowed`),
and only from an answer that holds it (`HELD`) — a balloon merely reaching over a
sound effect beside it is not the room that was written in, and refusing leaves
the block exactly where it already was. A cropped answer that no longer holds its
block is refused for the same reason; that only happens where the blocks
themselves overlap, and there the box is the honest answer.

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
