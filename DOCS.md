# DOCS.md

How the code works and why it is the way it is, a section per module.

`README.md` is the API surface — what to send and what comes back. `CLAUDE.md` is
the short list of what must not be broken. This is the reasoning under both, and
the place a comment goes when it is longer than a line.

## api/mangatrans

Two halves meet over HTTP and nothing is persisted anywhere: every request
carries the page it works on, and the browser is the only place a page, its
blocks, its mask and its translations live.

### server.py

The whole HTTP surface; everything else is a library it composes. Models load
lazily behind a lock on first use and are shared thereafter, so an API only ever
asked to detect never stands the OCR reader — or torch — up at all. `Regions`,
`Letters`, `Reader` and `Lama` hold a lock each: neither an OpenCV net, an
onnxruntime session nor torch generation is reentrant.

The API deliberately never renders end to end. Detection is imperfect, so
`/api/detect`, `/api/bubbles`, `/api/read`, `/api/clean` and `/api/render` are
separate calls that take boxes back in. That is what lets the dashboard show the
detection, have it corrected by hand, and only then act on it.

Reading a request is stricter than reading a model's answer. `ollama.noted` and
its neighbours are lenient because they are parsing something generated;
`terms_in`, `story_in` and `chapter_in` refuse what they do not recognise,
because a caller sending nonsense wants to be told.

`optionally` remembers a miss rather than retrying per request — LaMa's absence
is a missing file, not something that might be there next time.

### detect.py

Two models, for two different questions.

`Regions` is comic-text-and-bubble-detector, an RT-DETRv2 on onnxruntime: one
pass, three classes (`bubble`, `text_bubble`, `text_free`), about 0.2 s. Its
export carries RT-DETR's own postprocessing, so it answers with `labels`, `boxes`
and `scores` — already corner to corner and already in the page's pixels — rather
than logits to threshold. It is fed a plain resize to a square rather than a
letterbox, because the graph is told the page's shape separately and puts it back
itself; the preprocessing comes straight from the model's own
`preprocessor_config.json`.

The two text classes are kept apart as `Block.kind` (`speech`/`free`) and carried
to the translator. Everything else on the page treats them alike and always will
— finding, tracing and cleaning do not care — but a translation must, and the
model is asked both questions in the one pass, so keeping the answer costs
nothing and cannot be recovered later.

`Letters` is comic-text-detector, kept for its segmentation head alone: the
per-pixel ink map behind `/api/letters`, which is what lets a clean take the words
off the art. Its block head is not used. Boxing by region beats boxing by
lettering, because a region detector does not run two balloons together the way a
lettering detector does.

**`grow` is measured in the detector's pixels, not the page's.** The mask is
worked out on a 1024-square canvas and stretched, so its edge is only accurate to
a canvas pixel — one page pixel on a small page, three and a half on a 300 dpi
scan. Held in page pixels the same `grow` is three canvas pixels of allowance on
one and barely one on the other: measured, 100% of the lettering covered at
1000×1400 against 94.5% at 2480×3508, which comes out as the feet of every letter
still on the page after a clean.

RT-DETR matches one query to one object and needs no NMS, but it does sometimes
put two boxes over the same lettering, so `suppressed` drops the less sure of any
pair covering the same words. It runs per class: a balloon and the text filling it
cover each other almost entirely and are not duplicates of one another.

The 95 MB download retries with a wait between tries. A container network that
carries small files fine — a VM on a Mac, most proxies — drops one large response
part way through, and whatever drops one attempt drops the next few as well, so
five tries in a row all land in the same bad seconds and buy nothing. The last
wait is long enough to outlast the minute-scale throttle a release asset gets when
it is fetched repeatedly. `urlretrieve` cannot set headers and the default
`Python-urllib/3.x` is turned away by enough of the internet — GitHub's release
assets among them, which close the connection rather than saying so — that an
agent string is worth setting.

### bubble.py

The room inside the balloon a block was written in, as the largest rectangle in it
*around that block*. Pure OpenCV on the greyscale page, no model of its own:
threshold inside the balloon the detector found, paint the block in, fill the
holes, erode a margin, then search out from the block on a 128-pixel grid. `None`
is a real answer, and the right one for a sound effect over artwork.

**The balloon's own box is not the room.** A balloon is an oval and a line set to
the corners of its bounding box runs outside the outline. The block is painted in
first, or a column of vertical Japanese down the middle cuts the ground in two and
the answer is the gap beside the words. Light ground first, then inverted, for a
shout set white on black.

**A room always holds the block it is for.** `holding` searches out from the block
rather than for the largest rectangle in the balloon, so an answer is the words
plus whatever room is around them and never a rectangle somewhere else. Without
that, the largest rectangle is in the wrong part of the balloon as often as the
right one — a balloon with a tail, or one drawn round two lines with the words in
one of them.

Blocks are grouped by the balloon they are in, not by whether their answers
collide. That collision test existed because a balloon used to be flooded out from
the words, and one balloon came back as a different rectangle from each block in
it: measured, two blocks in one balloon agreed only 0.54. The detector now says
which balloon is which, so `assigned` asks the far simpler question — which
balloon holds this block, smallest first, so a shout drawn inside a thought wins
over the one around it.

Where several blocks share one balloon, `divided` cuts a cell apiece. That
division **recurses** — cut at the widest blank on whichever axis it is widest,
then each side again — because blocks set two across and two down are not in a
row, and one line of cuts gives the two on the right a left and a right half of a
balloon they are stacked inside. Every block keeps *its own* answer cropped, never
a share of a neighbour's: handing the group one balloon and cutting that up puts a
translation on the far side of the page from its Japanese.

`cropped` lives here rather than on `Box` because `geometry.py` is copied into the
image before the models are baked in — see *Infrastructure*.

### read.py

manga-ocr for Japanese, PP-OCR (RapidOCR on onnxruntime) for everything else,
behind one `Reader` that stands up only the language it is asked for. manga-ocr is
trained on manga specifically — vertical lines, stylised fonts, furigana — and is
why it manages what general OCR does not; there is no manga-ocr for Korean, and
using the Japanese one on Chinese only gets Japanese back.

This is the only module that imports torch, inside `Reader.load()`, and `rapidocr`
is deferred the same way inside `Ppocr.load()`.

**PP-OCR reads across a page and nothing else**, so `pieces` takes a balloon apart
before it goes over — into lines, or into columns each set out as a line by
`unstacked`. Which way the lettering runs is measured off the ink of the block
(`upright`, the shape of what was set) rather than taken from the language: a page
of Korean carries a sound effect written down the side of it just as a page of
Japanese does. The gaps look like the better signal — line spacing against
character spacing — and are not: CJK is set solid both ways, and the air between
two columns is the letterer's taste rather than anything to measure against.

`inked` decides ink from ground at the *edge* of the crop rather than by taking
the rarer of the two, which would do for ordinary dialogue and get a heavy sound
effect exactly backwards.

**onnxruntime hunts for a GPU as it loads**, and handed a card whose make it
cannot read — which is every container on a Mac — says so at warning level, from
C++, straight to the descriptor. There is no logger to turn down and no env var
for it: the severity can only be raised from Python once the environment exists,
by which time the line has been written. So `quieted()` catches fd 2 for the
length of a reader's load and writes back everything that is *not* that line. It
is not a blanket silencer and must not become one — a reader that cannot find its
weights says so the same way, and that has to get out.

### ollama.py

The whole page goes over in one request, held to a JSON schema. A line of manga
read on its own often cannot be translated at all, having no idea who is speaking
or about what.

The one-line-at-a-time fallback is the worst translation this produces — every
line loses the page it was to be read against — which is why a miscount is asked
again first. `corrected` shows the model its own reply with the counts, the count
being the one thing it cannot see from the request.

`CONTEXT` is asked for because Ollama's default is 4096 tokens and it truncates
silently, front first — the briefing and the glossary. What that costs is
invisible: the terms quietly stop being honoured, and what it looks like from
outside is a good model that has started miscounting. It went from 8192 to 12288
when the survey went in.

`repeat_penalty` is 1.0 against a default of 1.1. A page repeats itself on purpose
— the same shout in two balloons, a catchphrase, a row of `……` — and a penalty on
saying a thing twice is a push to render the second one differently for no reason
but that it came second. What it is nominally there to stop is a model looping,
which `PREDICT` bounds without touching the wording.

**A chapter is read before it is translated, and that is what makes page three
translatable.** The glossary and the story are both built forwards, so page three
would otherwise be translated with no idea what page forty reveals — and in manga
that is precisely where the pronouns, the honorifics and the name reveals are
settled. The early windows not having read the end does not matter: nothing is
lettered until all of it has been read.

The risk that opens is that **a model shown the ending writes towards it**. Page
three comes back heavy with a significance it has not earned, a line left vague
comes back settled, and people address each other as what they will turn out to
be — worse than not having read the chapter, being wrong in a way that reads as
deliberate. `CHAPTER_NOTE` is the whole of what stops it, and it holds because it
asks for a voice rather than states a rule, which is how a model under a schema
takes instruction at all.

**Whoever is still unknown is asked about under the page, not in the briefing.**
Measured with gemma4:12b: told only in the system message to correct what it is
given, the model translates 「先輩は僕の兄です」 — *senpai is my older brother* —
faithfully, and hands the cast straight back with 先輩 still unknown. Moved to a
line under the page naming who is waiting and what counts as evidence, it corrects.
Standing instructions read as a description of the job; a question under the text
reads as being about the text. `placed` says how far the reader has got under the
page for the same reason.

`NOT_A_NAME` drops the placeholder a model reaches for when nobody is named: filed
under "unknown", an unnamed character collides with the next one *and* turns the
question below into `Still unknown: 先輩, unknown`. Measured — and it appears to
be what stopped the model answering that question at all.

`SURVEY_NOTE` asks for a `note` on each of the cast, and for a long time did not:
the field was in `CAST_SCHEMA` from the start and the survey listed the cast by
name and gender alone, so a chapter came back with twelve names and nothing said
about any of them. What a page is told about someone is `described(person)`, which
is that note — so a survey that skipped it left the cast doing nothing but keeping
pronouns straight. It is capped at `CAST_NOTE_LIMIT` rather than `NOTE_LIMIT`
because it is a different thing from a term's note, and `STORY_NOTE` asks for the
same length for a reason that is easy to miss: a page's answer replaces what is
held (`lib/story.ts`), so a page still told "a few words" would quietly file the
survey's sentence down over the length of a chapter.

`beats` is a bare list counted against the pages rather than each beat carrying its
page number. Numbered, a page the model passed over would leave a gap instead of
shifting the rest — but that only moves the failure: a model that numbers from one
when the window started at seventeen loses the whole window, and there is no honest
way to tell that from a chapter that really does start there.

Some Ollama builds put the whole answer under `thinking` rather than `content`, so
`answered` reads both.

### inpaint.py

**Cleaning is LaMa, and the seam is not.** `fill` grows the hole by `EDGE` for
*sampling* only and then alpha-composites the result back through the caller's
ungrown, greyscale mask, so a soft brushed edge blends rather than steps and the
rim of half-ink just outside a letter is never read as art. That is independent of
which painter made the pixels.

A page marked all over short-circuits to white **before** the painter: there is
nothing left to make a fill out of, and a model handed a page that is entirely hole
does not say so, it invents one.

LaMa runs on crops around each mark rather than on the page — a page is mostly art
that is staying — and marks closer than `APART` go through together, or the context
around one letter holds the next as material to copy it from. Each crop is read
from the page rather than from the output, so two crops that overlap are not made
out of each other. The padding is a reflection of the crop rather than black: an
edge invented out of nothing is an edge the model tries to continue.

Input must be a whole multiple of `BLOCK`; measured, a size that is not a multiple
of eight fails inside a `Mul` rather than being padded.

**A crop over `LARGEST` is worked out smaller and stretched back, and this is what
keeps the API alive rather than what makes it quick.** LaMa's cost is in its
Fourier layers, which hold whole feature maps: measured, a 0.6 MP crop peaks near
2.2 GB and a 1.5 MP one near 3.9 GB, so a balloon on an A4 page scanned at 300 dpi
— 8.7 MP, an ordinary chapter — asks for something like 9 GB and the kernel kills
the process. There is no exception to catch and nothing in the API's own log; the
container simply goes, a front end sees a **502**, and `restart: unless-stopped`
brings it back so cleanly that `RestartCount` is still 0. If a big page ever 502s
again, look for `oom-kill` in `podman machine ssh sudo dmesg` before anything else.
Working smaller costs almost nothing: only the marked pixels are kept, lettering is
thin, and the fill is the tone and lines around it rather than any detail of its
own.

Anything the mark touched at all is kept as mark — rounding a thin stroke *out* of
the hole leaves that stroke behind.

### render.py

Hiding and lettering with PIL. `marked()` turns boxes and mask into one greyscale
page (white hidden, and where the two overlap the stronger wins, so a box is not
thinned by a mask brushed lightly over it); `hidden()` picks the fill. `fit` binary
-searches for the largest size that fits and wraps greedily; text too long for its
box is still drawn, overrunning, because a line that can be read and then moved
beats a bubble left empty.

The painter is handed *in* to `hidden`, the same way the detector and the reader
are, so standing it up stays the caller's business.

`ART` fills from the page around it; `WHITE_OUT` paints flat, which is right when
the ground has to be clear for new lettering and wrong almost everywhere else;
`TELEA` is `ART` without a model, kept because it needs no weights and because a
fill worth comparing against is worth having.

### languages.py

The one table of what a page can be written in: which reader reads it, which way
round it is read, whether its script stacks into columns and whether its words are
spaced. Both ends look languages up here, the dashboard through `/api/languages`
rather than by holding a copy.

**The language decides three things and nothing else.** Which reader stands up;
which way the blocks are sorted; and what `/api/translate` is told the page is in
(`source`, a word rather than a code — a caller may be translating something there
is no reader for). Detection, the segmentation mask, the balloon-finding and the
splitting are all language-blind and must stay that way: the detector was trained
on comics rather than on a script.

Traditional Chinese is the entry for a page set in columns and read right to left,
since that is how the comics printed in it are set; simplified is the entry for the
rows-and-left-to-right of a webcomic. Which way round a particular page is drawn is
still the reader's to correct.

### geometry.py

`Box`, x1/y1 exclusive, corners normalised on construction. `covers` is how much of
the smaller of two boxes lies inside the other rather than intersection over union,
which reads low for a small box wholly inside a large one — exactly the pair worth
catching.

### split.py

**Not in the pipeline.** The detector boxes by region, so it answers with one block
per balloon and there is nothing left to cut apart. It is kept, with its tests,
until enough real chapters have been through the new detector to say it never runs
two balloons together. Delete it once that is settled.

Its thresholds were tuned by grid search over 21 rendered cases and are not cheap
to work out again, which is the other reason it is still here. The measurements
behind them:

- The widest gap measured through a whole balloon was **0.37 of a character**, so
  barely more than a line gap is already worth cutting on where a cut stands
  through several lines at once. Every line it crosses has to fall blank in the
  same place at the same time, which lettering set inside one balloon does not do.
- A single column needs far more, because small kana and punctuation leave most of
  their cell empty: あっ、、、そうっ、か has gaps of **1.1 characters** in it and is
  still one line of one balloon.
- Lines of one block share an edge across a cut. Measured, a balloon of two centred
  columns is out by **3.0 characters** at the start and back 3.0 at the end, where
  two balloons a character apart are out by **1.0** at both.
- The character size is a mean rather than a median: punctuation and small kana are
  a large enough minority of the marks in a line to drag one down — on a real
  column the median says **23 px** where the characters are **42**.

Everything is measured in characters rather than pixels, since a page may be
lettered at any size. `pieces` must be handed the *ungrown* mask: growing it to
cover the halo around a letter also closes the gaps this exists to measure.

## web/src

A Vite/React dashboard that holds all the state. `App.tsx` owns it and the async
orchestration, keyed by page id; it holds no logic of its own — every edit to a
page goes through `lib/`.

The two folder runs are split at the clean rather than at the translate. Reading a
page is a detection and an OCR pass; hiding it is LaMa on every balloon, and over a
chapter that is most of the wait. Putting the clean in the second run is what makes
the synopsis, the beats and the cast arrive early enough to be worth correcting,
and correcting them is the whole point of there being a gap between the runs at
all. It was the other way round first — the first run cleaned as it read — and
what that cost was not the minutes themselves but where they fell: in front of the
one answer worth looking at before anything is committed to. The second run hides
what the first only read and skips a page already hidden, so the expensive half is
still paid for once.

Everything that keeps this half correct — the positional arrays, `bubble ?? box`,
the reading order defined twice, the three merge directions, the folder-run rules
— is in `CLAUDE.md`. What follows is only the map.

### lib/

- `api.ts` — the client and every shared type. Blocks are named with an id here,
  once, as they arrive: the API knows nothing of ids and answers the same page the
  same way twice.
- `regions.ts` / `lettering.ts` — all the per-block editing, as pure transforms on
  one `Analysis` / one `Lines`. A block operation goes here, not in a component.
- `story.ts` — every rule about which of two answers about a chapter's cast wins.
  Here rather than in `useStory` because it is the whole of what decides whether a
  chapter's facts are right.
- `bible.ts` — the same, for what a survey made of a chapter: how one window's
  answer is folded into the running one, and `fits`. `SURVEY_PAGES` is what to
  lower when a chapter will not survey, rather than raising the API's window.
- `fit.ts` — measuring and wrapping, shared with the canvas letterer so what is
  arranged is what comes out. `originalSize` is `sqrt(width × height / n)`, CJK
  being set on a square em. `roomInCharacters` is an estimate on purpose — what
  really fits depends on which letters — and nothing is held to it.
- `turn.ts` — the arithmetic of a box whose contents are turned. Writing a drawn
  point as `c + R(p − c)`, the shift that holds the untouched edge still is
  `(I − R)(c − c′)`: the two middles alone, so one correction serves every handle.
- `order.ts` (reading order), `mask.ts` (the brushed mask), `compose.ts` (the
  canvas letterer), `zip.ts` (a chapter in and back out), `chapter.ts` (which
  version of a page goes into the archive, and what the archive is called),
  `images.ts`, `dom.ts`.

### hooks/

One concern each: `useImageLibrary`, `useBatch`, `useMasks`, `useLetterMasks`,
`useObjectUrls`, `useOllama`, `usePrompt`, `useLanguage`, `useGlossary`,
`useStory`, `useChapter`, `useBoardView`, `useBoardKeys`, `useBoxDrag`,
`useFileDrop`, `useLetteringFont`.

Two patterns recur and are worth knowing. **Chapter state is read through a ref**
(`useGlossary`, `useStory`, `useChapter`), so translating a page does not make the
callback depend on the record. And **nothing that makes or frees a resource
happens inside a state updater** (`useObjectUrls`, `useImageLibrary`): those run
twice under StrictMode, and a URL made twice is one leaked.

`useBoardView` is the fiddliest. The board is a scroll box, so panning is the
browser's own; what is added is the zoom and the arithmetic that keeps the point
under the pointer still. Two things there are not obvious: the drawn page size is
**floored**, because a page that rounds its way past the board puts scrollbars up,
which shrinks the board and refits, forever; and the pan is caught in the
**capture** phase, or a pan across the page reaches the brush and draws instead.

### components/

`Board.tsx` draws the page and switches tools by `mode`; the tool rows are
`InspectTools`, `MaskTools` and `TranslateTools`, and the overlays are
`RegionsLayer`, `DrawRegion`, `MaskCanvas`, `TranslationLayer` and `ViewBar`, one
per thing that can be done to a page. `Sidebar.tsx` is the rail: it owns which
folder is open and holds `Gallery`, the folder's own bar and `BatchProgress`.
`icons.tsx` holds every line icon; `ui.tsx` every control — a button in the mask
toolbar and a button in the header are the same button.

## Infrastructure

`README.md` walks through the dashboard; `CLAUDE.md` holds the rules about layer
order, packing and the nginx template. What follows is the reasoning under them.

**Debian's package index stalls in a container that carries small files fine** — a
VM on a Mac, most proxies — because it is ~12 MB in one response, and apt reports
the stall as a mirror failure rather than a stall. Retries alone do not fix it,
since the retry hits the same stall; the pipelining has to go too.

**On x86_64 the torch wheel on PyPI drags the whole CUDA stack in** — gigabytes of
it, for a card this image will never see. PyTorch's CPU index has the same torch
without any of that, so it goes in first and pip treats the requirement as met.

**`rapidocr` asks for `opencv-python`, the GUI build.** It is the same cv2 as the
headless one everything here uses, linked against a libGL no server has, and being
installed second it wins. Hence the uninstall-and-reinstall.

The image runs as uid 10001, group 0 so it also runs under
`--user $(id -u):$(id -g)`. The models are fetched as that user into a directory it
already owns — chowning half a gigabyte of weights afterwards would copy every one
into a second layer. `--build-arg PREFETCH_MODEL=false` skips the prefetch.

**nginx holds a resolved address for the life of the process**, which is why the
upstream is a variable and the config is a template. The API container gets a new
address every time it is recreated — a crash, a `compose up`, a restart — and from
then on a plain `proxy_pass http://api:8000` answers **everything** with 502, long
after the API is back and healthy. Measured: 10.89.0.4 to 10.89.0.183 across one
restart, and every call 502 until nginx itself was restarted.

A variable needs a `resolver`, which is why `default.conf.template` goes to
`/etc/nginx/templates/` and the image's entrypoint fills `${NGINX_LOCAL_RESOLVERS}`
in from the container's own `/etc/resolv.conf`. That entrypoint returns early
unless `NGINX_ENTRYPOINT_LOCAL_RESOLVERS` is set, and `NGINX_ENVSUBST_FILTER` keeps
envsubst off nginx's own `$host` and `$remote_addr`. A `proxy_pass` with a variable
in it does not pass the path on by itself, hence the explicit `$request_uri`.

`VITE_API_URL` is set to the **empty string** rather than left unset, because
`lib/api.ts` falls back to `http://localhost:8000` only when it is nullish. An
empty base is what makes every call same-origin. `MANGA_TRANS_ORIGIN` and the CORS
headers still exist for anything talking to port 8000 directly.

**Ollama runs on the machine, not in the container, and Docker and Podman disagree
about what that machine is called** — `host.docker.internal` against
`host.containers.internal`. Naming either in the image leaves the other
unresolvable, which shows up as a page that reads perfectly and then will not
translate. A miss is deliberately not remembered: Ollama is as often started after
the dashboard as before it.
