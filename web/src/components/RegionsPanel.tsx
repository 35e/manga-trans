import { useEffect, useRef, useState } from 'react'
import type { Analysis, Region } from '../lib/api'
import { UNSURE } from '../lib/api'
import { plural } from '../lib/images'

type Props = {
  analysis: Analysis
  reading: boolean
  selected: number | null
  onSelect: (index: number | null) => void
  onToggleExcluded: (index: number) => void
}

/** What the detector found and what the reader made of it, block by block. */
export function RegionsPanel({
  analysis,
  reading,
  selected,
  onSelect,
  onToggleExcluded,
}: Props) {
  const list = useRef<HTMLUListElement>(null)
  const { regions } = analysis.detection
  const { texts } = analysis
  const excluded = new Set(analysis.excluded)
  const kept = regions.length - excluded.size

  useEffect(() => {
    if (selected === null) return
    list.current
      ?.querySelector(`[data-index="${selected}"]`)
      ?.scrollIntoView({ block: 'nearest' })
  }, [selected])

  return (
    <aside className="flex w-full shrink-0 flex-col border-slate-200 bg-white max-lg:h-72 max-lg:border-t lg:w-72 lg:border-l xl:w-96 dark:border-white/10 dark:bg-slate-950">
      <div className="flex items-start justify-between gap-2 border-b border-slate-200 px-4 py-3 dark:border-white/10">
        <div>
          <h2 className="text-sm font-semibold text-slate-900 dark:text-white">
            Text
          </h2>
          <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
            {regions.length === 0
              ? 'No lettering found on this page'
              : reading
                ? `Reading ${plural(regions.length, 'block')}…`
                : excluded.size > 0
                  ? `${plural(kept, 'block')} to clean · ${excluded.size} left alone`
                  : `${plural(regions.length, 'block')}, read by manga-ocr`}
          </p>
        </div>
        {texts && texts.some((text) => text) && (
          <CopyAll
            texts={texts.filter((text, index) => text && !excluded.has(index))}
          />
        )}
      </div>

      {regions.length === 0 ? (
        <p className="px-4 py-6 text-sm text-slate-500 dark:text-slate-400">
          The detector saw no lettering here.
        </p>
      ) : (
        <ul ref={list} className="min-h-0 flex-1 space-y-1.5 overflow-y-auto p-3">
          {regions.map((region, index) => (
            <Block
              key={index}
              region={region}
              index={index}
              text={texts?.[index] ?? null}
              reading={reading}
              excluded={excluded.has(index)}
              active={selected === index}
              onSelect={() => onSelect(selected === index ? null : index)}
              onToggleExcluded={() => onToggleExcluded(index)}
            />
          ))}
        </ul>
      )}
    </aside>
  )
}

function Block({
  region,
  index,
  text,
  reading,
  excluded,
  active,
  onSelect,
  onToggleExcluded,
}: {
  region: Region
  index: number
  text: string | null
  reading: boolean
  excluded: boolean
  active: boolean
  onSelect: () => void
  onToggleExcluded: () => void
}) {
  const [x0, y0, x1, y1] = region.box
  const unsure = region.confidence < UNSURE

  return (
    <li data-index={index} className="group relative">
      <button
        type="button"
        onClick={onSelect}
        aria-pressed={active}
        className={`w-full rounded-lg border py-2 pr-9 pl-2.5 text-left transition-colors ${
          excluded
            ? 'border-dashed border-slate-300 bg-slate-50 dark:border-white/15 dark:bg-white/5'
            : active
              ? 'border-indigo-500 bg-indigo-50 dark:bg-indigo-500/10'
              : 'border-slate-200 hover:border-slate-300 hover:bg-slate-50 dark:border-white/10 dark:hover:border-white/20 dark:hover:bg-white/5'
        }`}
      >
        <div className="flex items-baseline gap-2">
          <span className="text-xs font-semibold text-slate-400 tabular-nums dark:text-slate-500">
            {index + 1}
          </span>

          {text === null ? (
            <span className="text-sm text-slate-400 italic dark:text-slate-500">
              {reading ? 'reading…' : 'not read'}
            </span>
          ) : text === '' ? (
            <span className="text-sm text-slate-400 italic dark:text-slate-500">
              nothing read here
            </span>
          ) : (
            <p
              lang="ja"
              className={`min-w-0 flex-1 text-sm leading-relaxed select-text ${
                excluded
                  ? 'text-slate-400 line-through dark:text-slate-500'
                  : 'text-slate-900 dark:text-white'
              }`}
            >
              {text}
            </p>
          )}
        </div>

        <p className="mt-1 text-[11px] text-slate-400 tabular-nums dark:text-slate-500">
          {excluded ? (
            <span className="font-medium text-slate-500 dark:text-slate-400">
              left alone
            </span>
          ) : region.manual ? (
            <span className="font-medium text-indigo-600 dark:text-indigo-400">
              added by hand
            </span>
          ) : (
            <span className={unsure ? 'text-amber-600 dark:text-amber-400' : ''}>
              {Math.round(region.confidence * 100)}%
            </span>
          )}{' '}
          · {x0}, {y0} · {x1 - x0} × {y1 - y0}
        </p>
      </button>

      <button
        type="button"
        onClick={onToggleExcluded}
        title={
          excluded
            ? 'Clean this block after all'
            : 'Leave this block alone: do not clean it'
        }
        aria-label={
          excluded
            ? `Clean block ${index + 1} after all`
            : `Leave block ${index + 1} alone`
        }
        className="absolute top-1.5 right-1.5 rounded-md p-1.5 text-slate-400 transition-colors hover:bg-slate-200 hover:text-slate-700 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-500 dark:hover:bg-white/10 dark:hover:text-white"
      >
        <svg
          aria-hidden="true"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.9"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="size-3.5"
        >
          {excluded ? (
            <path d="M4 12a8 8 0 1 0 2.3-5.6M4 4v4h4" />
          ) : (
            <path d="M6 6l12 12M18 6 6 18" />
          )}
        </svg>
      </button>
    </li>
  )
}

/** Every block still being cleaned, in reading order, on the clipboard. */
function CopyAll({ texts }: { texts: string[] }) {
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    if (!copied) return
    const timer = setTimeout(() => setCopied(false), 1600)
    return () => clearTimeout(timer)
  }, [copied])

  return (
    <button
      type="button"
      onClick={() => {
        navigator.clipboard.writeText(texts.join('\n')).then(
          () => setCopied(true),
          () => setCopied(false),
        )
      }}
      className="shrink-0 rounded-md border border-slate-200 px-2 py-1 text-xs font-medium text-slate-600 transition-colors hover:bg-slate-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-500 dark:border-white/15 dark:text-slate-300 dark:hover:bg-white/5"
    >
      {copied ? 'Copied' : 'Copy all'}
    </button>
  )
}
