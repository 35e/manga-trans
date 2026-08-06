import { useEffect, useRef, useState } from 'react'
import type { Analysis, Detection, Stage } from '../lib/api'
import { UNSURE } from '../lib/api'
import type { GalleryImage } from '../lib/images'
import type { Brush, Mask, Point } from '../lib/mask'
import { MaskTools } from './MaskTools'

type Mode = 'inspect' | 'mask'

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
  onToggleExcluded: (index: number) => void
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
  onToggleExcluded,
}: Props) {
  const surface = useRef<HTMLDivElement>(null)
  const overlay = useRef<HTMLCanvasElement>(null)
  const cursor = useRef<HTMLDivElement>(null)
  const drawing = useRef<Point | null>(null)

  const page = useFittedPage(surface, image)
  const [mode, setMode] = useState<Mode>('inspect')
  const [brush, setBrush] = useState<Brush>({ radius: 16, erase: false })
  const [showBoxes, setShowBoxes] = useState(true)
  const [showCleaned, setShowCleaned] = useState(false)
  // Brushing draws straight onto the canvas; this is only so the buttons that
  // care whether anything is marked catch up when a stroke ends.
  const [edits, setEdits] = useState(0)

  const detection = analysis?.detection ?? null
  const busy = stage !== null
  const masking = mode === 'mask' && !showCleaned
  const marked = Boolean(mask && !mask.empty)

  // The array changes identity only when a block is dropped or put back, which
  // is what the effects below want to hear about.
  const excludedList = analysis?.excluded
  const excluded = new Set(excludedList)
  const toClean = (regions: Detection['regions']) =>
    regions.filter((_, index) => !excluded.has(index)).map((region) => region.box)

  useEffect(() => setShowCleaned(false), [image?.id])
  useEffect(() => {
    if (cleaned) setShowCleaned(true)
  }, [cleaned])

  // The canvas is blank whenever it is remounted or resized, so the mask is
  // drawn back on after anything that could have done either — including a
  // block being dropped from the list, which erases its box from the mask.
  useEffect(() => {
    if (overlay.current && mask) mask.showOn(overlay.current)
  }, [mask, page, edits, masking, analysis])

  // Coming to the mask with blocks already found and nothing marked yet: start
  // from the blocks, minus any that have been left alone. The list of those is
  // read through a ref rather than depended on: dropping a block from a mask
  // deliberately cleared out should not seed the whole page again.
  const excludedNow = useRef(excludedList)
  excludedNow.current = excludedList

  useEffect(() => {
    if (!masking || !mask || !mask.empty || !detection) return
    const skip = new Set(excludedNow.current)
    mask.boxes(
      detection.regions
        .filter((_, index) => !skip.has(index))
        .map((region) => region.box),
    )
    setEdits((count) => count + 1)
  }, [masking, mask, detection])

  // A block picked out on the board is dropped with the delete key.
  useEffect(() => {
    if (selected === null) return
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== 'Delete' && event.key !== 'Backspace') return
      const typing = event.target as HTMLElement | null
      if (typing && ['INPUT', 'TEXTAREA', 'SELECT'].includes(typing.tagName)) return
      event.preventDefault()
      onToggleExcluded(selected)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [selected, onToggleExcluded])

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
          <div className="flex rounded-lg border border-slate-300 p-0.5 dark:border-white/15">
            <Segment
              label="Inspect"
              active={mode === 'inspect'}
              onClick={() => setMode('inspect')}
            />
            <Segment
              label="Mask"
              active={mode === 'mask'}
              onClick={() => setMode('mask')}
            />
          </div>
        )}

        {detection && mode === 'inspect' && (
          <label className="flex cursor-pointer items-center gap-2 text-xs font-medium text-slate-600 select-none dark:text-slate-300">
            <input
              type="checkbox"
              checked={showBoxes}
              onChange={(event) => setShowBoxes(event.target.checked)}
              className="size-3.5 accent-indigo-600"
            />
            Boxes
          </label>
        )}

        <button
          type="button"
          onClick={onDetect}
          disabled={!image || busy}
          className="inline-flex shrink-0 items-center gap-2 rounded-lg border border-slate-300 px-3 py-2 text-sm font-medium text-slate-700 transition-colors hover:bg-slate-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-500 disabled:cursor-not-allowed disabled:opacity-40 dark:border-white/15 dark:text-slate-200 dark:hover:bg-white/10"
        >
          {stage === 'detecting' && <Spinner />}
          {stage === 'reading' && <Spinner />}
          {stage === 'detecting'
            ? 'Detecting…'
            : stage === 'reading'
              ? 'Reading…'
              : detection
                ? 'Read again'
                : 'Detect text'}
        </button>

        <button
          type="button"
          onClick={() => {
            if (mask && !mask.empty) mask.toBlob().then(onClean)
          }}
          disabled={!image || busy || !marked}
          title={marked ? undefined : 'Mark something to hide first'}
          className="inline-flex shrink-0 items-center gap-2 rounded-lg bg-indigo-600 px-3.5 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-500 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-500 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {stage === 'cleaning' && <Spinner />}
          {stage === 'cleaning' ? 'Cleaning…' : 'Clean page'}
        </button>
      </div>

      {masking && (
        <MaskTools
          brush={brush}
          onBrush={setBrush}
          onFill={() => {
            if (!mask || !detection) return
            mask.boxes(toClean(detection.regions))
            setEdits((count) => count + 1)
          }}
          canFill={Boolean(
            mask && detection && toClean(detection.regions).length > 0,
          )}
          onClear={() => {
            if (!mask) return
            mask.clear()
            setEdits((count) => count + 1)
          }}
          canClear={marked}
        />
      )}

      {cleaned && (
        <div className="flex flex-wrap items-center gap-3 border-b border-emerald-200 bg-emerald-50 px-4 py-2 text-sm text-emerald-900 dark:border-emerald-500/30 dark:bg-emerald-500/10 dark:text-emerald-200">
          <span className="font-medium">The page has been cleaned.</span>
          <div className="flex rounded-lg border border-emerald-300 p-0.5 dark:border-emerald-500/40">
            <Segment
              label="Original"
              active={!showCleaned}
              onClick={() => setShowCleaned(false)}
            />
            <Segment
              label="Cleaned"
              active={showCleaned}
              onClick={() => setShowCleaned(true)}
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
                  key={index}
                  region={region}
                  index={index}
                  page={detection}
                  text={analysis?.texts?.[index] ?? null}
                  excluded={excluded.has(index)}
                  active={selected === index}
                  onSelect={() => onSelect(selected === index ? null : index)}
                />
              ))}

            {!showCleaned && (
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

function RegionBox({
  region,
  index,
  page,
  text,
  excluded,
  active,
  onSelect,
}: {
  region: Detection['regions'][number]
  index: number
  page: { width: number; height: number }
  text: string | null
  excluded: boolean
  active: boolean
  onSelect: () => void
}) {
  const [x0, y0, x1, y1] = region.box
  const unsure = region.confidence < UNSURE

  return (
    <button
      type="button"
      onClick={onSelect}
      title={excluded ? `${text ?? ''} — left alone`.trim() : text || undefined}
      aria-label={
        excluded
          ? `Block ${index + 1}, left alone`
          : text
            ? `Block ${index + 1}: ${text}`
            : `Text block ${index + 1}`
      }
      aria-pressed={active}
      style={{
        left: `${(x0 / page.width) * 100}%`,
        top: `${(y0 / page.height) * 100}%`,
        width: `${((x1 - x0) / page.width) * 100}%`,
        height: `${((y1 - y0) / page.height) * 100}%`,
      }}
      className={`absolute border-2 transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white ${
        excluded
          ? 'border-dashed border-slate-400/80 hover:bg-slate-400/15'
          : active
            ? 'border-indigo-500 bg-indigo-500/20'
            : unsure
              ? 'border-amber-500/80 hover:bg-amber-500/15'
              : 'border-indigo-500/70 hover:bg-indigo-500/15'
      }`}
    >
      <span
        className={`absolute -top-px -left-px px-1 text-[10px] leading-4 font-semibold text-white tabular-nums ${
          excluded
            ? 'bg-slate-500/80 line-through'
            : active
              ? 'bg-indigo-500'
              : unsure
                ? 'bg-amber-500'
                : 'bg-indigo-500/80'
        }`}
      >
        {index + 1}
      </span>
    </button>
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
        <p className="mt-4 text-sm font-medium text-slate-500 dark:text-slate-400">
          Click a page in the gallery to put it on the board
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
