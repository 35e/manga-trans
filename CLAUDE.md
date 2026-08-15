# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

It holds only what must not be broken. **`DOCS.md` is the reasoning under it** —
how each module works, what was tried instead, and the measurements behind the
thresholds. `README.md` documents every endpoint, its form fields and its
behaviour, plus the environment-variable table; read it before changing anything
in the API's surface, and keep it accurate.

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
`/api/detect`, `/api/bubbles`, `/api/read`, `/api/clean` and `/api/render` are
separate calls taking boxes back in, which is what lets the dashboard show the
detection, have it corrected by hand, and only then act on it.

`DOCS.md` has the module map for both halves. In brief: `api/mangatrans/server.py`
is the whole HTTP surface and the rest are libraries it composes; `web/src/App.tsx`
owns the state and the orchestration, and every edit to a page goes through
`web/src/lib/`.

Under `docker-compose.yml` the two are served on **one origin**: nginx hands out
the built dashboard and proxies `/api` through to the API container, so
`VITE_API_URL` builds as `''`. Keep `lib/api.ts`'s fallback
(`?? 'http://localhost:8000'`) nullish-coalescing rather than `||`, or the empty
base collapses back to the absolute one and same-origin stops working.

**nginx must re-resolve the API per request, and that is why its config is a
template.** A plain `proxy_pass http://api:8000` is looked up once at load, so
after the API container is recreated nginx holds a dead address and 502s
everything. Do not "simplify" `set $upstream` / `$request_uri` /
`${NGINX_LOCAL_RESOLVERS}` back to a plain hostname.

## Invariants worth knowing before editing

**Per-page arrays are positionally aligned.** `analysis.detection.regions`,
`analysis.texts`, `analysis.excluded` (indices) and `lettering[pageId]` are all
indexed by the same block position. `lib/regions.ts` and `lib/lettering.ts` are
the only places that edit these, and they are paired — `blocks.inserted` with
`lines.inserted`, `blocks.moved` with `lines.moved`, `blocks.split` with
`lines.split`. Anything **asynchronous** must re-find its block by `region.id`
(`blocks.withReading`), because the list may have been reordered in flight.

**A block is not where the translation goes.** `region.box` is where the Japanese
is; `region.bubble` is the room it was written in, and lettering uses
`bubble ?? box` (`lines.roomFor`). Anything that changes a box by hand must drop
the bubble with it and ask `/api/bubbles` again — `blocks.withBox` does the
dropping. Size is capped at `originalSize` (`lib/fit.ts`).

**Reading order is defined twice and must agree.** Down the page, then across it
the way the language is read — `(y0, -x1)` right to left, `(y0, x0)` left to
right — in `detect.py` (`blocks.sort`) and in `lib/order.ts` (`key`), both from
`languages.Language.rtl`. A hand-drawn block is inserted where that rule puts it,
not appended: the order is also the order the page is translated in.

**The language decides three things and nothing else.** Which reader stands up;
which way the blocks are sorted; and what `/api/translate` is told the page is in
(`source`, a word rather than a code). Detection, the segmentation mask, the
balloon-finding and the splitting are all language-blind and must stay that way.

**PP-OCR reads across a page and nothing else**, so `read.pieces` takes a balloon
apart before it goes over. Which way the lettering runs is measured off the ink
(`read.upright`) rather than taken from the language, and `read.inked` decides ink
from ground at the *edge* of the crop rather than by taking the rarer of the two.

**`rapidocr` asks for `opencv-python`, the GUI build**, and installed second it
wins. The Dockerfile drops it and lays the headless build back down afterwards; do
not "simplify" those three lines into one `pip install`.

**onnxruntime hunts for a GPU as it loads** and complains from C++ straight to
fd 2, so `read.quieted()` catches the descriptor for the length of a reader's load
and writes back everything that is *not* that line. Not a blanket silencer, and it
must not become one: a reader that cannot find its weights says so the same way.

**Ollama runs on the machine, not in the container, and Docker and Podman disagree
about what that machine is called.** `ollama.answering()` tries each in turn and
keeps the first that answers; keep the host out of the Dockerfile. A miss is
deliberately not remembered.

**There are two independent letterers.** `/api/render` sets text with PIL
(`render.py`); the dashboard's "Apply to image" sets it with canvas
(`lib/compose.ts`), sharing `lib/fit.ts` with the on-board preview. They are not
expected to produce identical output. `lib/fit.ts` must be awaited via `ready()`
before measuring, or it measures the fallback font.

**Masks.** Greyscale, page-sized, white hidden, greys partial. `server.mask_in`
believes an alpha channel only when some of it is actually clear, since a browser
canvas always exports one — do not "simplify" that check. `lib/mask.ts` exports
white-on-black for exactly this reason.

**A line is sent with what it is and how long it may be, and both are positional
with `texts`.** `kinds` and `budgets` (`lettering.budgetFor`) are the two things a
caller has and the model cannot see. `ollama.translate` drops empty texts and
renumbers what is left, so it must carry both along with them. `server.beside`
refuses a list that is not one per text for the same reason. Both are optional and
neither is ever enforced. `KINDS_NOTE` and `BUDGET_NOTE` are appended by
`ollama.told` only when the lines carry them, for the same reason `TERMS_NOTE` is:
the prompt is the caller's to replace.

**`KINDS_NOTE` has to insist on an answer for every line.** A model told a line is
a sound effect may decide it needs no translating and answer for the other
nineteen, which is the one failure the whole schema exists to prevent.

**The context window is asked for, not taken** (`ollama.CONTEXT`). Ollama's default
is 4096 tokens and it truncates silently, front first — the briefing and the
glossary. The symptom is a page that comes back miscounted.

**A miscounted page is asked again before it is taken apart.** `ollama.corrected`
puts the model's own reply back in front of it with the counts. The
one-line-at-a-time pass cannot miscount and is the worst translation this
produces. Terms are kept from whichever answer named any (`terms or noted(again)`).

**A chapter is read before it is translated, and that is what makes page three
translatable.** `/api/survey` sweeps the whole chapter's already-read text first,
`SURVEY_PAGES` at a time with the running answer handed back in, and only then is
anything translated.

**A model shown the ending writes towards it, and `ollama.CHAPTER_NOTE` is the
whole of what stops it.** The note draws one line: use the chapter for who someone
*is*, not for what *happens*. It is appended by `told` only where a chapter is
given, and `ollama.placed` says how far the reader has got **under the page**
rather than in the briefing.

**The beats are indexed by a page's place in its folder.** One line per page,
positional — so `App.placeOf` and the list a folder run works through must agree,
and `lib/bible.fits` refuses a bible whose beat count no longer matches. Where it
does not fit, translate against no chapter at all. The API keeps a count contract
on the way in for the same reason (`ollama.beaten`): a window that miscounts twice
hands back **no** beats rather than beats a page out.

**A survey's terms are seeded once, at the end of the sweep — never window by
window.** `useGlossary` keeps the first rendering of a term and never moves it, so
folding each window in as it arrived would freeze the readings of the windows that
had not yet read the ending. Inside `lib/bible.folded` the rule is the opposite way
round and last wins.

**A chapter carries three things, and they are kept three different ways round.**
The bible is *replaced* per window and the **last** window wins. The glossary
accumulates and the **first** rendering of a term wins (`useGlossary`). The story's
`scene` is **replaced** and the **last** page wins (`useStory`, via `lib/story.ts`).
Everything is capped on the way in and out (`SCENE_LIMIT`, `CAST_LIMIT`,
`NOTE_LIMIT`, `CAST_NOTE_LIMIT`, `GLOSSARY_LIMIT`, `SYNOPSIS_LIMIT`,
`REGISTER_LIMIT`, `BEAT_LIMIT`, `BEATS_LIMIT`): it all rides on *every* page of a
folder run. An empty answer leaves what was held.

**A cast note is not a term note and is not capped like one.** `NOTE_LIMIT` (80)
says what decides a wording; `CAST_NOTE_LIMIT` (200) is the character, and it is
all a page is told about anyone on it. Both `SURVEY_NOTE` and `STORY_NOTE` ask for
it at that length — a page told to write "a few words" would shorten what the
survey wrote, since `lib/story.ts` lets any non-empty note replace the held one.

**A survey's cast is seeded unsettled.** `settled` means "set by hand", which is
what `described` tells the model, so the sweep must not claim it. Hand corrections
stay immovable through both runs.

**`unknown` is a real answer about the cast, and the schema is what makes it
cheap.** `gender` is an enum of `male`/`female`/`unknown`, and the merge rules
(`lib/story.ts`) are asymmetric: **`unknown` never overwrites something known**,
any other known value replaces what was held, and a fact in `settled` is not moved
by either side. Same shape for `note`: empty never clears.

**The cast is keyed on the name, so the name has to be the page's own.**
`STORY_NOTE` asks for `先輩`, not "the senior": an English label drifts between
pages and every drift is a second person in a cast of twelve. `ollama.NOT_A_NAME`
drops the placeholder a model reaches for when nobody is named.

**Whoever is still unknown is asked about under the page, not in the briefing**
(`ollama.asking`). Measured: standing instructions read as a description of the
job, where a question under the text reads as being about the text.

**The same line twice on one page is one question** (`ollama.asked_once`), keyed on
the words *and* the kind, answered once, and lettered into every block it came
from. Where those blocks have different room the **tightest** budget is sent. For
the same reason `repeat_penalty` is 1.0 against a default of 1.1 — a page repeats
itself on purpose — and `num_predict` (`PREDICT`) bounds the looping instead.

**A chapter's glossary rides on the translate call, and the browser keeps it.**
Four things about it are load-bearing: `terms` is **not** in `SCHEMA["required"]`,
so a page that names nobody cannot break the count contract; a **miscounted** page
still keeps the terms of its spoiled reply; the note asking for terms is appended
by `ollama.told` rather than written into `SYSTEM_DEFAULT`; and in `useGlossary`
the **first** rendering of a term wins and is never overwritten, capped at `LIMIT`.

**A folder made by hand outlives its pages.** `useImageLibrary.remove` deletes an
archive's folder when its last page goes, and must not do that to a `manual` one.
That flag is held rather than read off a missing `archive`, because the first zip
dropped into such a folder fills `archive` in — matched by name, which is also why
`makeFolder` refuses a name already taken rather than making it unique.

**A block is marked into a mask by the lettering in it, never by its box.**
`mask.mark` falls back to stamping the whole rectangle only for a tracing that
failed. Anything marking a block after the mask has been seeded goes through
`App.markLetters`, which asks for the tracing if it is not already held.

**`fill` defaults differ by endpoint**: `art` for `/api/clean`, `white` for
`/api/render`. `art` falls back to `telea` when LaMa's weights cannot be loaded —
`server.optionally` remembers the miss rather than retrying per request. The
painter is handed *in* to `render.hidden`.

**A folder run goes one page at a time, and the page in hand is the page named.**
`useBatch` sends one page through a step and waits: `Detector` and `Reader` hold a
lock each, so pages sent together queue at the API regardless and come back in no
particular order. Board actions are shut while a run is going (`Board`'s
`runningFolder`) for the same reason.

**A folder is run twice, and the step is an argument rather than the hook's.**
`App.examine` finds and reads a page; `App.render` hides, translates and letters
one. Between them the chapter is read, which is why `useBatch.start` also takes an
`after` — run inside the same run so there is one card, one **Stop** and one thing
that is true. **The clean is in the second run, not the first**, so a chapter can
be read and put right before LaMa is paid for: `render` hides a page that was read
and not hidden (`cleanedNow`) and skips one that has been, which is also why a mask
brushed after a clean is re-cleaned from the board rather than by running the folder
again. Both steps take the page rather than reading the active one, and move the
step tabs and clear the selection **only** for the page on the board — reading
lands on **mask**, hiding on **translate**, which is where the page-change rule
would put each anyway. Both hand back why they gave up (`App.lastFailure`), because
a run has to say which page that was; not every empty answer is a refusal, so the
reason decides rather than the boolean. `App.render` takes its analysis **from the
step that found it** rather than reading it back out of state.

**A tracing is dropped as soon as its clean lands, unless its page is on the
board.** Traced masks are cached per `(pageId, spread)` (`useLetterMasks`) and the
`ImageBitmap`s are explicitly `close()`d on removal; that hook writes its ref
before its state so a tracing can be marked into a mask the moment it arrives.

**A chapter is packed on the click, not during the run.** `App.downloadFolder`
draws every page afresh out of `lettering` and `cleanedPages`, one at a time.

**Every page goes into the archive, at whatever state it reached**
(`chapter.finished`): lettered, else cleaned, else the original bytes under the
original name. A page that will not compose falls back to its original rather than
being skipped — dropping a page renumbers every page after it. Anything drawn on is
a PNG; anything untouched is passed through byte for byte.

**Nothing in the archive is compressed** (`zip.pack`, `level: 0`). `pack` also
numbers a repeated name rather than letting it overwrite — pages are told apart by
name *and* size *and* date, so one folder really can hold two files called
`001.png`.

**`GalleryFolder.archive` is the only record of what a chapter arrived as**, the
folder being named after the archive's *stem*. `folderFor` still matches on the
stem, so re-dropping fills the same folder and the extension that named it first is
the one that sticks. Dedupe is scoped to the folder (`fingerprint(file, folder)`).

**Dockerfile layer order is load-bearing.** Only `__init__.py`, `geometry.py`,
`languages.py`, `detect.py`, `read.py` and `inpaint.py` are copied before the model
prefetch step; editing any of those six invalidates ~810 MB of baked weights.
Everything else is copied after. This is also why `bubble.cropped` lives in
`bubble.py` rather than on `Box`.

Keep the heavy imports deferred to first use: `onnxruntime` inside
`Regions.__init__` and `Lama.__init__`, torch inside `Reader.load()`, `rapidocr`
inside `Ppocr.load()`, `huggingface_hub` inside the `ensure_*` functions.

**Debian's package index stalls in a container that carries small files fine**, and
apt reports the stall as a mirror failure. `/etc/apt/apt.conf.d/99robust` sets
retries *and* turns HTTP pipelining off; retries alone do not fix it.
`detect.ensure_model` retries its 95 MB GitHub download for the same reason, and
waits between tries.

**A merged block is wrong for everything downstream** — read as one string,
translated as one line, lettered into one balloon. Boxing by *region* rather than
by lettering is what prevents it. If real pages show it merging after all, the fix
belongs in the detector or back in `split.py`, not in the dashboard. `split.py` is
kept, out of the pipeline, until that is settled.

**Order in `Regions.__call__` is load-bearing**: decode, then split the classes,
then `suppressed`, then pad (`PAD`), then sort. Padding before `suppressed` could
make two neighbours look like one.

**Duplicates still have to be dropped.** RT-DETR needs no NMS but does sometimes put
two boxes over the same lettering. `detect.suppressed` runs per class — a balloon
and the text filling it cover each other almost entirely and are not duplicates.

**A room always holds the block it is for.** `bubble.holding` searches out from the
block rather than for the largest rectangle in the balloon. Everything downstream
leans on this.

**The balloon's own box is not the room.** `bubble.inside` thresholds the interior
*within* the detected box. The block is painted in first (`shrunk`), or a column of
vertical text down the middle cuts the ground in two. Light ground first, then
inverted, for a shout set white on black.

**Blocks are grouped by the balloon they are in, not by whether their answers
collide** (`bubble.assigned`, `HELD`, smallest first so a shout inside a thought
wins). Where several share one balloon, `bubble.divided` cuts a cell apiece, and
that division **recurses**. Every block keeps *its own* answer cropped, never a
share of a neighbour's. A cropped answer that no longer holds its block is refused.

**Cleaning is LaMa, and the seam is not.** `inpaint.fill` grows the hole by `EDGE`
for *sampling* only and then alpha-composites through the caller's ungrown mask.
That is independent of which painter made the pixels. A page marked all over
short-circuits to white **before** the painter. LaMa runs on crops
(`inpaint.patches`), and marks closer than `APART` go through together. Its input
must be a whole multiple of `BLOCK`.

**A crop over `inpaint.LARGEST` is worked out smaller and stretched back**, and this
is what keeps the API alive rather than what makes it quick: over it, the kernel
kills the process, there is no exception to catch, and a front end sees a **502**.
If a big page ever 502s again, look for `oom-kill` in `podman machine ssh sudo
dmesg` before anything else.

**`grow` is in the detector's pixels, not the page's** (`detect.GROW`, `page_mask`).
The mask is worked out on a 1024-square canvas and stretched, so its edge is only
accurate to a canvas pixel. Anything else measured against the mask belongs in the
same units.

**An answer from `/api/bubbles` depends on which other boxes were asked about**, so
anything whose answer will be lettered with must send *every* box on the page. This
is why `reread` in `App.tsx` reads only the changed blocks but asks for balloons for
all of them, and applies the result to every region by id.

## Style

Comments are minimal. A module or function gets one line saying what it is for,
where the name does not carry it, and a line or two more only where a reader who
did not know would change the line — an ordering that matters, a library's
misbehaviour, a threshold with a trap in it, the derivation of an expression that
is otherwise unreadable.

Everything longer goes in `DOCS.md`: how a module works, what was tried instead,
the measurements behind a constant. Do not explain the domain in a comment, and do
not restate the code. Names carry the rest.

Python tests are named as statements of behaviour
(`test_the_far_edge_is_exclusive`), which is why they need no docstring.
