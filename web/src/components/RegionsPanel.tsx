import { useEffect, useRef, useState } from 'react'
import type { Analysis, Region } from '../lib/api'
import { UNSURE } from '../lib/api'
import { plural } from '../lib/images'

type Props = {
  analysis: Analysis
  reading: boolean
  selected: number | null
  onSelect: (index: number | null) => void
}

/** What the detector found and what the reader made of it, block by block. */
export function RegionsPanel({ analysis, reading, selected, onSelect }: Props) {
  const list = useRef<HTMLUListElement>(null)
  const { regions } = analysis.detection
  const { texts } = analysis

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
                : `${plural(regions.length, 'block')}, read by manga-ocr`}
          </p>
        </div>
        {texts && texts.some((text) => text) && <CopyAll texts={texts} />}
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
              active={selected === index}
              onSelect={() => onSelect(selected === index ? null : index)}
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
  active,
  onSelect,
}: {
  region: Region
  index: number
  text: string | null
  reading: boolean
  active: boolean
  onSelect: () => void
}) {
  const [x0, y0, x1, y1] = region.box
  const unsure = region.confidence < UNSURE

  return (
    <li data-index={index}>
      <button
        type="button"
        onClick={onSelect}
        aria-pressed={active}
        className={`w-full rounded-lg border px-2.5 py-2 text-left transition-colors ${
          active
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
              className="min-w-0 flex-1 text-sm leading-relaxed text-slate-900 select-text dark:text-white"
            >
              {text}
            </p>
          )}
        </div>

        <p className="mt-1 text-[11px] text-slate-400 tabular-nums dark:text-slate-500">
          <span className={unsure ? 'text-amber-600 dark:text-amber-400' : ''}>
            {Math.round(region.confidence * 100)}%
          </span>{' '}
          · {x0}, {y0} · {x1 - x0} × {y1 - y0}
        </p>
      </button>
    </li>
  )
}

/** Every block, in reading order, on the clipboard. */
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
        const all = texts.filter((text) => text).join('\n')
        navigator.clipboard.writeText(all).then(
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
