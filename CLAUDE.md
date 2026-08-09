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

- `detect.py` — comic-text-detector on OpenCV's ONNX backend. Two heads off one
  pass: block boxes, and the per-pixel segmentation mask behind `/api/letters`.
  Fetches its own weights on first use.
- `bubble.py` — the balloon a block was written in, as the largest rectangle
  inside it. Pure OpenCV on the greyscale page, no model: flood the light ground
  from the block (painted in first, or vertical Japanese cuts the balloon in
  two), fill its holes, erode a margin, then a stack search for the largest
  rectangle on a 128-pixel grid. Every answer is checked and `None` is a real
  answer. `/api/detect` includes it because it is free there.
- `read.py` — manga-ocr. **The only thing that imports torch, and it does so
  inside `Reader.load()`.** Keep that import deferred and keep torch out of
  every other module.
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
- `components/Board.tsx` draws the page and switches tools by `mode`; the
  overlays over it are `RegionsLayer`, `DrawRegion`, `MaskCanvas`,
  `TranslationLayer` and `ViewBar`, one per thing that can be done to a page.
- `components/icons.tsx` holds every line icon; `components/ui.tsx` every
  control.
- Hooks own one concern each: `useImageLibrary`, `useMasks`, `useLetterMasks`,
  `useObjectUrls`, `useOllama`, `usePrompt`, `useBoardView`, `useBoardKeys`,
  `useBoxDrag`, `useFileDrop`, `useLetteringFont`.

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

**Reading order is defined twice and must agree.** `(y0, -x1)` — down the page,
then right to left — in `detect.py` (`blocks.sort`) and in `lib/order.ts`
(`key`). A hand-drawn block is inserted where that rule puts it, not appended,
because the order is also the order the page is translated in as one
conversation.

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

**`fill` defaults differ by endpoint**: `art` for `/api/clean`, `white` for
`/api/render`. `inpaint.fill` grows the hole by `EDGE` for *sampling* only, and
still paints just what was marked — the soft rim around lettering must not
become the material the fill is made of.

**Traced masks are cached per `(pageId, spread)`** (`useLetterMasks`), and the
`ImageBitmap`s are explicitly `close()`d on removal — a different `grow`/`spread`
is a different tracing, not the same one again. That hook writes its ref before
its state so a tracing can be marked into a mask the moment it arrives.

**Dockerfile layer order is load-bearing.** Only `__init__.py`, `geometry.py`,
`detect.py` and `read.py` are copied before the model prefetch step; editing any
of those four invalidates ~550 MB of baked weights and forces a re-download on
the next build. Everything else is copied after.

## Style

Comments are for *why*, and they are short. A module or function gets a line or
two saying what it is for and what would go wrong done the obvious way; anything
longer has to be earning it — the non-obvious threshold, the ordering that
matters, the alternative that was tried. Do not restate the code, and do not
explain the domain here: that is what this file is for. Names carry the rest.

Python tests are named as statements of behaviour
(`test_the_far_edge_is_exclusive`).
