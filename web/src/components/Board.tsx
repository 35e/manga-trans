import { useEffect, useRef, useState } from 'react'
import { useBoxDrag } from '../hooks/useBoxDrag'
import type {
  Analysis,
  BoardMode,
  Box,
  Detection,
  Fill,
  Lettering,
  Stage,
} from '../lib/api'
import { UNSURE } from '../lib/api'
import type { GalleryImage } from '../lib/images'
import type { Brush, Mask, Point } from '../lib/mask'
import { BoxGrips } from './BoxGrips'
import { MaskTools } from './MaskTools'
import { Steps } from './Steps'
import { TranslateTools } from './TranslateTools'
import { TranslationLayer } from './TranslationLayer'

/** Everything the translate tab needs, kept together rather than spread out. */
export type Translating = {
  models: string[]
  model: string
  onModel: (model: string) => void
  target: string
  onTarget: (target: string) => void
  onTranslate: () => void
  onFitAll: () => void
  lettering: (Lettering | null)[]
  onBox: (index: number, box: Box) => void
  onSize: (index: number, by: number) => void
  onApply: () => void
  applying: boolean
  note: string | null
}

type Props = {
  image: GalleryImage | null
  analysis: Analysis | null
  mask: Mask | null
  cleaned: string | null
  stage: Stage | null
  error: string | null
  selected: number | null
  onSelect: (index: number | null) => void
  onDetect: () => void
  onClean: (mask: Blob) => void
  onRunAll: () => void
  onAddRegion: (box: Box) => void
  /** A block's box while it is being dragged: every frame of it. */
  onRegionBox: (index: number, box: Box) => void
  /** The same drag, once it is over, with the box as it was before. */
  onRegionSettled: (index: number, was: Box) => void
  onToggleExcluded: (index: number) => void
  /** The traced lettering for this page, once it has been asked for. */
  letters: ImageBitmap | null
  onTrace: () => Promise<ImageBitmap | null>
  /** How far past the ink a tracing reaches, in page pixels. */
  spread: number
  onSpread: (spread: number) => void
  /** What a clean puts where the lettering was. */
  fill: Fill
  onFill: (fill: Fill) => void
  mode: BoardMode
  onMode: (mode: BoardMode) => void
  /** Whether the board is showing the cleaned page or the one that came in. */
  showCleaned: boolean
  onShowCleaned: (showing: boolean) => void
  translating: Translating
}

export function Board({
  image,
  analysis,
  mask,
  cleaned,
  stage,
  error,
  selected,
  onSelect,
  onDetect,
  onClean,
  onRunAll,
  onAddRegion,
  onRegionBox,
  onRegionSettled,
  onToggleExcluded,
  letters,
  onTrace,
  spread,
  onSpread,
  fill,
  onFill,
  mode,
  onMode,
  showCleaned,
  onShowCleaned,
  translating,
}: Props) {
  const surface = useRef<HTMLDivElement>(null)
  const overlay = useRef<HTMLCanvasElement>(null)
  const cursor = useRef<HTMLDivElement>(null)
  const drawing = useRef<Point | null>(null)

  const page = useFittedPage(surface, image)
  const [brush, setBrush] = useState<Brush>({ radius: 16, erase: false })
  const [showBoxes, setShowBoxes] = useState(true)
  const [adding, setAdding] = useState(false)
  const [drawn, setDrawn] = useState<Box | null>(null)
  const drawnFrom = useRef<Point | null>(null)
  // Brushing draws straight onto the canvas; this is only so the buttons that
  // care whether anything is marked catch up when a stroke ends.
  const [edits, setEdits] = useState(0)

  const detection = analysis?.detection ?? null
  const busy = stage !== null
  const masking = mode === 'mask' && !showCleaned
  const marked = Boolean(mask && !mask.empty)

  // Where this page has got to, which is what the steps show.
  const read = analysis?.texts != null
  const lettered = translating.lettering.some(Boolean)

  // The array changes identity only when a block is dropped or put back, which
  // is what the effects below want to hear about.
  const excludedList = analysis?.excluded
  const excluded = new Set(excludedList)
  const toClean = (regions: Detection['regions']) =>
    regions.filter((_, index) => !excluded.has(index)).map((region) => region.box)

  // The canvas is blank whenever it is remounted or resized, so the mask is
  // drawn back on after anything that could have done either — including a
  // block being dropped from the list, which erases its box from the mask.
  useEffect(() => {
    if (overlay.current && mask) mask.showOn(overlay.current)
  }, [mask, page, edits, masking, analysis])

  // These are read through refs rather than depended on: dropping a block from
  // a mask that was deliberately cleared out should not seed the whole page
  // again, and the tracing callback changes identity on every keystroke of
  // state above.
  const excludedNow = useRef(excludedList)
  excludedNow.current = excludedList
  const lettersNow = useRef(letters)
  lettersNow.current = letters
  const traceNow = useRef(onTrace)
  traceNow.current = onTrace

  // Coming to the mask with blocks already found and nothing marked yet: mark
  // the lettering itself, which is what wants hiding — the boxes around it are
  // only the fallback for when tracing fails.
  useEffect(() => {
    if (!masking || !mask || !mask.empty || !detection) return
    let dropped = false

    void (async () => {
      const traced = lettersNow.current ?? (await traceNow.current())
      // Someone may have left, or started brushing, while that was in the air.
      if (dropped || !mask.empty) return
      const skip = new Set(excludedNow.current)
      const boxes = detection.regions
        .filter((_, index) => !skip.has(index))
        .map((region) => region.box)
      if (traced) mask.letters(traced, boxes)
      else mask.boxes(boxes)
      setEdits((count) => count + 1)
    })()

    return () => {
      dropped = true
    }
  }, [masking, mask, detection])

  // The keys that work on whichever block is picked out: delete drops it from
  // the clean, and on the translate tab the arrows set its type larger and
  // smaller. Neither fires while something is being typed into.
  const nudge = translating.onSize
  useEffect(() => {
    if (selected === null) return

    const onKey = (event: KeyboardEvent) => {
      const typing = event.target as HTMLElement | null
      if (typing && ['INPUT', 'TEXTAREA', 'SELECT'].includes(typing.tagName)) return

      if (mode === 'translate') {
        if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return
        event.preventDefault()
        const step = event.shiftKey ? 5 : 1
        nudge(selected, event.key === 'ArrowUp' ? step : -step)
        return
      }

      if (event.key !== 'Delete' && event.key !== 'Backspace') return
      event.preventDefault()
      onToggleExcluded(selected)
    }

    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [selected, mode, onToggleExcluded, nudge])

  const at = (event: React.PointerEvent<HTMLCanvasElement>): Point => {
    const rect = event.currentTarget.getBoundingClientRect()
    return {
      x: ((event.clientX - rect.left) / rect.width) * (image?.width ?? 0),
      y: ((event.clientY - rect.top) / rect.height) * (image?.height ?? 0),
    }
  }

  const moveDot = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const dot = cursor.current
    if (!dot || !image) return
    const rect = event.currentTarget.getBoundingClientRect()
    const size = brush.radius * 2 * (rect.width / image.width)
    dot.style.width = `${size}px`
    dot.style.height = `${size}px`
    dot.style.transform = `translate(${event.clientX - rect.left - size / 2}px, ${
      event.clientY - rect.top - size / 2
    }px)`
    dot.style.opacity = '1'
  }

  const repaint = () => {
    if (overlay.current && mask) mask.showOn(overlay.current)
  }

  /** A box dragged out on the page, in the page's own pixels, corners in order. */
  const between = (from: Point, to: Point): Box => [
    Math.max(0, Math.round(Math.min(from.x, to.x))),
    Math.max(0, Math.round(Math.min(from.y, to.y))),
    Math.min(image?.width ?? 0, Math.round(Math.max(from.x, to.x))),
    Math.min(image?.height ?? 0, Math.round(Math.max(from.y, to.y))),
  ]

  return (
    <section className="flex min-h-0 min-w-0 flex-1 flex-col">
      <div className="flex flex-wrap items-center gap-3 border-b border-slate-200 bg-white px-4 py-2.5 dark:border-white/10 dark:bg-slate-950">
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium text-slate-900 dark:text-white">
            {image ? image.name : 'No page on the board'}
          </p>
          <p className="text-xs text-slate-500 tabular-nums dark:text-slate-400">
            {image
              ? `${image.width} × ${image.height}${
                  page ? ` · ${Math.round(page.scale * 100)}%` : ''
                }`
              : 'Pick one from the gallery'}
          </p>
        </div>

        {image && (
          <Steps
            current={mode}
            onPick={onMode}
            steps={[
              { id: 'inspect', label: 'Text', done: read, open: true },
              { id: 'mask', label: 'Clean', done: Boolean(cleaned), open: read },
              {
                id: 'translate',
                label: 'Translate',
                done: lettered,
                open: read,
              },
            ]}
          />
        )}

        {image && (
          <div className="flex shrink-0 items-center gap-2">
            <button
              type="button"
              onClick={onRunAll}
              disabled={busy}
              title="Detect the text, hide it, and letter the page"
              className="inline-flex items-center gap-2 rounded-lg border border-slate-300 px-3 py-2 text-sm font-medium text-slate-700 transition-colors hover:bg-slate-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-500 disabled:cursor-not-allowed disabled:opacity-40 dark:border-white/15 dark:text-slate-200 dark:hover:bg-white/10"
            >
              Do all three
            </button>

            {mode === 'inspect' && (
              <Action onClick={onDetect} disabled={busy} stage={stage}>
                {detection ? 'Read again' : 'Detect text'}
              </Action>
            )}

            {mode === 'mask' && (
              <Action
                onClick={() => {
                  if (mask && !mask.empty) mask.toBlob().then(onClean)
                }}
                disabled={busy || !marked}
                stage={stage}
                title={marked ? undefined : 'Mark something to hide first'}
              >
                {cleaned ? 'Clean again' : 'Clean page'}
              </Action>
            )}

            {mode === 'translate' && (
              <Action
                onClick={lettered ? translating.onApply : translating.onTranslate}
                disabled={
                  busy ||
                  translating.applying ||
                  (!lettered && !(translating.model && read))
                }
                stage={stage}
                title={
                  lettered
                    ? 'Set the lettering into the page and save it'
                    : undefined
                }
              >
                {translating.applying
                  ? 'Applying…'
                  : lettered
                    ? 'Apply to image'
                    : 'Translate page'}
              </Action>
            )}
          </div>
        )}
      </div>

      {mode === 'inspect' && detection && (
        <div className="flex flex-wrap items-center gap-3 border-b border-slate-200 bg-slate-50 px-4 py-2 dark:border-white/10 dark:bg-white/5">
          <label className="flex cursor-pointer items-center gap-2 text-xs font-medium text-slate-600 select-none dark:text-slate-300">
            <input
              type="checkbox"
              checked={showBoxes}
              onChange={(event) => setShowBoxes(event.target.checked)}
              className="size-3.5 accent-indigo-600"
            />
            Show the boxes
          </label>
          <button
            type="button"
            onClick={() => setAdding((armed) => !armed)}
            aria-pressed={adding}
            title="Draw a block the detector missed"
            className={`rounded-lg border px-2.5 py-1.5 text-xs font-medium transition-colors ${
              adding
                ? 'border-indigo-600 bg-indigo-600 text-white'
                : 'border-slate-300 text-slate-700 hover:bg-white dark:border-white/15 dark:text-slate-200 dark:hover:bg-white/10'
            }`}
          >
            {adding ? 'Drawing a block…' : 'Add a block'}
          </button>

          <span className="text-xs text-slate-500 dark:text-slate-400">
            {adding
              ? 'Drag across the bubble it missed. It is read and put in reading order.'
              : 'Click one to pick it out, drag it to move, pull an edge to resize; delete drops it from the clean.'}
          </span>
        </div>
      )}

      {masking && (
        <MaskTools
          brush={brush}
          onBrush={setBrush}
          onMarkBlocks={() => {
            if (!mask || !detection) return
            mask.boxes(toClean(detection.regions))
            setEdits((count) => count + 1)
          }}
          onMarkLetters={() => {
            if (!mask || !detection) return
            void (async () => {
              const traced = lettersNow.current ?? (await traceNow.current())
              if (!traced) return
              mask.letters(traced, toClean(detection.regions))
              setEdits((count) => count + 1)
            })()
          }}
          canMark={Boolean(
            mask && detection && toClean(detection.regions).length > 0,
          )}
          tracing={stage === 'tracing'}
          onClear={() => {
            if (!mask) return
            mask.clear()
            setEdits((count) => count + 1)
          }}
          canClear={marked}
          spread={spread}
          onSpread={onSpread}
          fill={fill}
          onFill={onFill}
          note={read ? null : 'find the text first, or brush the page by hand'}
        />
      )}

      {mode === 'translate' && image && (
        <TranslateTools
          models={translating.models}
          model={translating.model}
          onModel={translating.onModel}
          target={translating.target}
          onTarget={translating.onTarget}
          onFitAll={translating.onFitAll}
          canFit={lettered}
          note={translating.note}
        />
      )}

      {cleaned && (
        <div className="flex flex-wrap items-center gap-3 border-b border-emerald-200 bg-emerald-50 px-4 py-2 text-sm text-emerald-900 dark:border-emerald-500/30 dark:bg-emerald-500/10 dark:text-emerald-200">
          <span className="font-medium">The page has been cleaned.</span>
          <div className="flex rounded-lg border border-emerald-300 p-0.5 dark:border-emerald-500/40">
            <Segment
              label="Original"
              active={!showCleaned}
              onClick={() => onShowCleaned(false)}
            />
            <Segment
              label="Cleaned"
              active={showCleaned}
              onClick={() => onShowCleaned(true)}
            />
          </div>
          <a
            href={cleaned}
            download={`${image?.name.replace(/\.[^.]+$/, '') ?? 'page'}-clean.png`}
            className="ml-auto rounded-lg bg-emerald-600 px-3 py-1.5 text-xs font-medium text-white transition-colors hover:bg-emerald-500"
          >
            Download
          </a>
        </div>
      )}

      {error && (
        <p
          role="alert"
          className="border-b border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-300"
        >
          {error}
        </p>
      )}

      <div
        ref={surface}
        onPointerDown={(event) => {
          // Anywhere on the board that is not a box puts down whichever box is
          // held: picking one out is only ever meant to last as long as it is
          // being worked on.
          if (selected === null) return
          if (!(event.target as Element).closest('[data-box]')) onSelect(null)
        }}
        className="board relative min-h-0 flex-1 overflow-hidden bg-slate-100 p-6 dark:bg-slate-900"
      >
        {image && page && (
          <div
            className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 shadow-2xl ring-1 ring-black/10 dark:ring-white/10"
            style={{ width: page.width, height: page.height }}
          >
            <img
              src={showCleaned && cleaned ? cleaned : image.url}
              alt={image.name}
              className="block h-full w-full select-none"
              draggable={false}
            />

            {!showCleaned &&
              showBoxes &&
              mode === 'inspect' &&
              detection?.regions.map((region, index) => (
                <RegionBox
                  key={region.id}
                  region={region}
                  index={index}
                  page={detection}
                  scale={page.scale}
                  text={analysis?.texts?.[index] ?? null}
                  excluded={excluded.has(index)}
                  active={selected === index}
                  onSelect={() => onSelect(selected === index ? null : index)}
                  onBox={(box) => onRegionBox(index, box)}
                  onSettled={(was) => onRegionSettled(index, was)}
                />
              ))}

            {mode === 'inspect' && adding && (
              // Over the blocks, not under them: while a block is being drawn
              // the whole page is the drawing surface.
              <div
                className="absolute inset-0 z-30 cursor-crosshair touch-none"
                onPointerDown={(event) => {
                  event.currentTarget.setPointerCapture(event.pointerId)
                  const rect = event.currentTarget.getBoundingClientRect()
                  drawnFrom.current = {
                    x: ((event.clientX - rect.left) / rect.width) * image.width,
                    y: ((event.clientY - rect.top) / rect.height) * image.height,
                  }
                }}
                onPointerMove={(event) => {
                  const from = drawnFrom.current
                  if (!from) return
                  const rect = event.currentTarget.getBoundingClientRect()
                  setDrawn(
                    between(from, {
                      x: ((event.clientX - rect.left) / rect.width) * image.width,
                      y: ((event.clientY - rect.top) / rect.height) * image.height,
                    }),
                  )
                }}
                onPointerUp={(event) => {
                  if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                    event.currentTarget.releasePointerCapture(event.pointerId)
                  }
                  drawnFrom.current = null
                  // A stray click is not a block: it takes a real drag.
                  if (drawn && drawn[2] - drawn[0] > 6 && drawn[3] - drawn[1] > 6) {
                    onAddRegion(drawn)
                  }
                  setDrawn(null)
                }}
              >
                {drawn && (
                  <span
                    aria-hidden="true"
                    style={{
                      left: `${(drawn[0] / image.width) * 100}%`,
                      top: `${(drawn[1] / image.height) * 100}%`,
                      width: `${((drawn[2] - drawn[0]) / image.width) * 100}%`,
                      height: `${((drawn[3] - drawn[1]) / image.height) * 100}%`,
                    }}
                    className="absolute border-2 border-dashed border-indigo-500 bg-indigo-500/20"
                  />
                )}
              </div>
            )}

            {mode === 'translate' && (
              <TranslationLayer
                page={{ width: image.width, height: image.height }}
                scale={page.scale}
                lettering={translating.lettering}
                selected={selected}
                onSelect={onSelect}
                onBox={translating.onBox}
              />
            )}

            {mode === 'mask' && !showCleaned && (
              <canvas
                ref={overlay}
                width={image.width}
                height={image.height}
                className={`absolute inset-0 h-full w-full ${
                  masking ? 'cursor-none touch-none' : 'pointer-events-none'
                }`}
                onPointerDown={(event) => {
                  if (!masking || !mask) return
                  event.currentTarget.setPointerCapture(event.pointerId)
                  const point = at(event)
                  drawing.current = point
                  mask.dot(point, brush)
                  repaint()
                }}
                onPointerMove={(event) => {
                  if (!masking) return
                  moveDot(event)
                  if (!drawing.current || !mask) return
                  const point = at(event)
                  mask.stroke(drawing.current, point, brush)
                  drawing.current = point
                  repaint()
                }}
                onPointerUp={(event) => {
                  if (!drawing.current) return
                  drawing.current = null
                  if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                    event.currentTarget.releasePointerCapture(event.pointerId)
                  }
                  setEdits((count) => count + 1)
                }}
                onPointerLeave={() => {
                  if (cursor.current) cursor.current.style.opacity = '0'
                }}
              />
            )}

            {masking && (
              <div
                ref={cursor}
                aria-hidden="true"
                className="pointer-events-none absolute top-0 left-0 rounded-full border-2 border-white opacity-0 ring-1 ring-slate-900/70"
              />
            )}
          </div>
        )}

        {!image && <BoardEmpty />}
      </div>
    </section>
  )
}

/** What each stage is called while it is happening. */
const LABELS: Record<Stage, string> = {
  detecting: 'Detecting…',
  reading: 'Reading…',
  tracing: 'Tracing…',
  cleaning: 'Cleaning…',
  translating: 'Translating…',
}

/** The one button that does the thing this step is for. */
function Action({
  onClick,
  disabled,
  stage,
  title,
  children,
}: {
  onClick: () => void
  disabled: boolean
  stage: Stage | null
  title?: string
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      className="inline-flex shrink-0 items-center gap-2 rounded-lg bg-indigo-600 px-3.5 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-500 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-500 disabled:cursor-not-allowed disabled:opacity-40"
    >
      {stage ? (
        <>
          <Spinner />
          {LABELS[stage]}
        </>
      ) : (
        children
      )}
    </button>
  )
}

function Segment({
  label,
  active,
  onClick,
}: {
  label: string
  active: boolean
  onClick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
        active
          ? 'bg-indigo-600 text-white'
          : 'text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-white/10'
      }`}
    >
      {label}
    </button>
  )
}

function Spinner() {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 24 24"
      className="size-4 animate-spin"
      fill="none"
    >
      <circle
        cx="12"
        cy="12"
        r="9"
        stroke="currentColor"
        strokeOpacity="0.3"
        strokeWidth="3"
      />
      <path
        d="M21 12a9 9 0 0 0-9-9"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="round"
      />
    </svg>
  )
}

/**
 * One block the detector found, over the lettering it found.
 *
 * Draggable and pullable by its edges, because the detector runs two speech
 * bubbles together often enough to matter: the fix is to pull this one back off
 * the second and draw a block around what is left.
 */
function RegionBox({
  region,
  index,
  page,
  scale,
  text,
  excluded,
  active,
  onSelect,
  onBox,
  onSettled,
}: {
  region: Detection['regions'][number]
  index: number
  page: { width: number; height: number }
  scale: number
  text: string | null
  excluded: boolean
  active: boolean
  onSelect: () => void
  onBox: (box: Box) => void
  onSettled: (was: Box) => void
}) {
  const drag = useBoxDrag({ box: region.box, page, scale, onBox, onSettled })
  const [x0, y0, x1, y1] = region.box
  const unsure = region.confidence < UNSURE

  return (
    <div
      data-box
      style={{
        left: `${(x0 / page.width) * 100}%`,
        top: `${(y0 / page.height) * 100}%`,
        width: `${((x1 - x0) / page.width) * 100}%`,
        height: `${((y1 - y0) / page.height) * 100}%`,
      }}
      className={`absolute ${active ? 'z-20' : 'z-10'}`}
    >
      <button
        type="button"
        onPointerDown={drag.grab}
        onPointerMove={drag.shift}
        onPointerUp={drag.release}
        onPointerCancel={drag.release}
        onClick={() => {
          // The click that ends a drag is not a click on the box.
          if (drag.dragged.current) {
            drag.dragged.current = false
            return
          }
          onSelect()
        }}
        title={excluded ? `${text ?? ''} — left alone`.trim() : text || undefined}
        aria-label={
          excluded
            ? `Block ${index + 1}, left alone`
            : text
              ? `Block ${index + 1}: ${text}`
              : `Text block ${index + 1}`
        }
        aria-pressed={active}
        className={`h-full w-full cursor-move touch-none border transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white ${
          excluded
            ? 'border-dashed border-slate-400/60 hover:border-slate-400'
            : active
              ? 'border-indigo-500 bg-indigo-500/8'
              : unsure
                ? 'border-amber-500/60 hover:border-amber-500 hover:bg-amber-500/8'
                : 'border-indigo-500/40 hover:border-indigo-500 hover:bg-indigo-500/8'
        }`}
      >
        <span
          className={`absolute -top-px -left-px rounded-br px-1 text-[9px] leading-4 font-medium text-white tabular-nums transition-colors ${
            excluded
              ? 'bg-slate-500/70 line-through'
              : active
                ? 'bg-indigo-500'
                : unsure
                  ? 'bg-amber-500/80'
                  : 'bg-indigo-500/60'
          }`}
        >
          {index + 1}
        </span>
      </button>

      {active && <BoxGrips drag={drag} />}
    </div>
  )
}

function BoardEmpty() {
  return (
    <div className="pointer-events-none flex h-full items-center justify-center">
      <div className="text-center">
        <svg
          aria-hidden="true"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinejoin="round"
          className="mx-auto size-12 text-slate-300 dark:text-slate-700"
        >
          <rect x="4" y="3" width="16" height="18" rx="2" />
          <path d="M8 8h5M8 12h8M8 16h6" />
        </svg>
        <p className="mt-4 text-sm font-medium text-slate-600 dark:text-slate-300">
          Click a page in the gallery to put it on the board
        </p>
        <p className="mt-1 text-sm text-slate-400 dark:text-slate-500">
          Then: find the text, hide it, letter it back in.
        </p>
      </div>
    </div>
  )
}

type FittedPage = { width: number; height: number; scale: number }

/**
 * The size the page is drawn at: as large as the board allows without cropping
 * it or blowing it up past its own pixels. Measured rather than left to
 * `object-contain`, because the boxes and the mask are laid over this exact
 * rect.
 */
function useFittedPage(
  surface: React.RefObject<HTMLDivElement | null>,
  image: GalleryImage | null,
): FittedPage | null {
  const [page, setPage] = useState<FittedPage | null>(null)
  const width = image?.width
  const height = image?.height

  useEffect(() => {
    const element = surface.current
    if (!element || !width || !height) {
      setPage(null)
      return
    }

    const fit = (available: { width: number; height: number }) => {
      if (available.width === 0 || available.height === 0) return
      const scale = Math.min(
        available.width / width,
        available.height / height,
        1,
      )
      setPage({ width: width * scale, height: height * scale, scale })
    }

    // ResizeObserver reports the box once on observe, so there is no separate
    // first measurement — and none taken a different way, past the padding.
    const observer = new ResizeObserver(([entry]) => fit(entry.contentRect))
    observer.observe(element)
    return () => observer.disconnect()
  }, [surface, width, height])

  return page
}
