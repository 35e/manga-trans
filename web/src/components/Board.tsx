import { useEffect, useRef, useState } from 'react'
import type { Analysis, Detection, Stage } from '../lib/api'
import { UNSURE } from '../lib/api'
import type { GalleryImage } from '../lib/images'

type Props = {
  image: GalleryImage | null
  analysis: Analysis | null
  stage: Stage | null
  error: string | null
  selected: number | null
  onSelect: (index: number | null) => void
  onDetect: () => void
}

export function Board({
  image,
  analysis,
  stage,
  error,
  selected,
  onSelect,
  onDetect,
}: Props) {
  const surface = useRef<HTMLDivElement>(null)
  const page = useFittedPage(surface, image)
  const [showBoxes, setShowBoxes] = useState(true)
  const detection = analysis?.detection ?? null
  const busy = stage !== null

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

        {detection && (
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
          className="inline-flex shrink-0 items-center gap-2 rounded-lg bg-indigo-600 px-3.5 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-500 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-500 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {busy && (
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
          )}
          {stage === 'detecting'
            ? 'Detecting…'
            : stage === 'reading'
              ? 'Reading…'
              : detection
                ? 'Read again'
                : 'Detect text'}
        </button>
      </div>

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
              src={image.url}
              alt={image.name}
              className="block h-full w-full select-none"
              draggable={false}
            />

            {showBoxes &&
              detection?.regions.map((region, index) => (
                <RegionBox
                  key={index}
                  region={region}
                  index={index}
                  page={detection}
                  text={analysis?.texts?.[index] ?? null}
                  active={selected === index}
                  onSelect={() => onSelect(selected === index ? null : index)}
                />
              ))}
          </div>
        )}

        {!image && <BoardEmpty />}
      </div>
    </section>
  )
}

function RegionBox({
  region,
  index,
  page,
  text,
  active,
  onSelect,
}: {
  region: Detection['regions'][number]
  index: number
  page: { width: number; height: number }
  text: string | null
  active: boolean
  onSelect: () => void
}) {
  const [x0, y0, x1, y1] = region.box
  const unsure = region.confidence < UNSURE

  return (
    <button
      type="button"
      onClick={onSelect}
      title={text || undefined}
      aria-label={text ? `Block ${index + 1}: ${text}` : `Text block ${index + 1}`}
      aria-pressed={active}
      style={{
        left: `${(x0 / page.width) * 100}%`,
        top: `${(y0 / page.height) * 100}%`,
        width: `${((x1 - x0) / page.width) * 100}%`,
        height: `${((y1 - y0) / page.height) * 100}%`,
      }}
      className={`absolute border-2 transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white ${
        active
          ? 'border-indigo-500 bg-indigo-500/20'
          : unsure
            ? 'border-amber-500/80 hover:bg-amber-500/15'
            : 'border-indigo-500/70 hover:bg-indigo-500/15'
      }`}
    >
      <span
        className={`absolute -top-px -left-px px-1 text-[10px] leading-4 font-semibold text-white tabular-nums ${
          active ? 'bg-indigo-500' : unsure ? 'bg-amber-500' : 'bg-indigo-500/80'
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
 * `object-contain`, because the boxes are positioned against this exact rect.
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
