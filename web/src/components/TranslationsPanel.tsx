import { useEffect, useRef } from 'react'
import type { Lettering } from '../lib/api'
import { plural } from '../lib/images'

type Props = {
  originals: (string | null)[]
  lettering: (Lettering | null)[]
  selected: number | null
  onSelect: (index: number | null) => void
  onChange: (index: number, patch: Partial<Lettering>) => void
  onFit: (index: number) => void
}

/** The translated lines: the words, and how big they are set. */
export function TranslationsPanel({
  originals,
  lettering,
  selected,
  onSelect,
  onChange,
  onFit,
}: Props) {
  const list = useRef<HTMLUListElement>(null)
  const set = lettering.filter(Boolean).length

  useEffect(() => {
    if (selected === null) return
    list.current
      ?.querySelector(`[data-index="${selected}"]`)
      ?.scrollIntoView({ block: 'nearest' })
  }, [selected])

  return (
    <aside className="flex w-full shrink-0 flex-col border-slate-200 bg-white max-lg:h-72 max-lg:border-t lg:w-72 lg:border-l xl:w-96 dark:border-white/10 dark:bg-slate-950">
      <div className="border-b border-slate-200 px-4 py-3 dark:border-white/10">
        <h2 className="text-sm font-semibold text-slate-900 dark:text-white">
          Translation
        </h2>
        <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
          {set === 0
            ? 'Nothing translated yet'
            : `${plural(set, 'line')} set on the page`}
        </p>
      </div>

      {set === 0 ? (
        <p className="px-4 py-6 text-sm text-slate-500 dark:text-slate-400">
          Read the page first, then translate it. Each line lands where its
          original was, and can be resized from there.
        </p>
      ) : (
        <ul ref={list} className="min-h-0 flex-1 space-y-2 overflow-y-auto p-3">
          {lettering.map((line, index) =>
            line === null ? null : (
              <li
                key={index}
                data-index={index}
                className={`rounded-lg border p-2.5 transition-colors ${
                  selected === index
                    ? 'border-indigo-500 bg-indigo-50 dark:bg-indigo-500/10'
                    : 'border-slate-200 dark:border-white/10'
                }`}
                onFocus={() => onSelect(index)}
              >
                <p
                  lang="ja"
                  className="truncate text-[11px] text-slate-400 dark:text-slate-500"
                  title={originals[index] ?? ''}
                >
                  {index + 1}. {originals[index] || '—'}
                </p>

                <textarea
                  value={line.text}
                  rows={2}
                  onChange={(event) =>
                    onChange(index, { text: event.target.value })
                  }
                  className="mt-1 w-full resize-y rounded-md border border-slate-200 bg-white px-2 py-1 text-sm text-slate-900 focus-visible:outline-2 focus-visible:outline-indigo-500 dark:border-white/15 dark:bg-slate-900 dark:text-white"
                />

                <div className="mt-1.5 flex items-center gap-2">
                  <p className="flex-1 text-[11px] text-slate-400 tabular-nums dark:text-slate-500">
                    {Math.round(line.size)}px in a {line.box[2] - line.box[0]} ×{' '}
                    {line.box[3] - line.box[1]} box
                    {selected === index && (
                      <span className="text-slate-500 dark:text-slate-400">
                        {' '}
                        · ↑↓ to resize
                      </span>
                    )}
                  </p>

                  <button
                    type="button"
                    onClick={() => onFit(index)}
                    title="The largest size that lands in this box"
                    className="rounded-md border border-slate-200 px-2 py-1 text-[11px] font-medium text-slate-600 transition-colors hover:bg-slate-50 dark:border-white/15 dark:text-slate-300 dark:hover:bg-white/5"
                  >
                    Fit
                  </button>
                </div>
              </li>
            ),
          )}
        </ul>
      )}
    </aside>
  )
}
