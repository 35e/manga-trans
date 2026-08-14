import { useEffect, useRef, useState } from 'react'
import type { Lettering } from '../lib/api'
import { plural } from '../lib/images'
import { Button, FOCUS } from './ui'

type Props = {
  originals: (string | null)[]
  lettering: (Lettering | null)[]
  selected: number | null
  onSelect: (index: number | null) => void
  onChange: (index: number, patch: Partial<Lettering>) => void
  onFit: (index: number) => void
  /** Cut this line, and the block it is set in, in two at that point. */
  onSplit: (index: number, at: number) => void
}

/** The translated lines: the words, and how big they are set. */
export function TranslationsPanel({
  originals,
  lettering,
  selected,
  onSelect,
  onChange,
  onFit,
  onSplit,
}: Props) {
  const list = useRef<HTMLUListElement>(null)
  const set = lettering.filter(Boolean).length

  // Kept after the box loses focus on purpose: pressing Split takes the focus
  // away, and a cursor forgotten then is a button that can never be pressed.
  const [caret, setCaret] = useState<{ line: number; at: number } | null>(null)

  useEffect(() => {
    if (selected === null) return
    list.current
      ?.querySelector(`[data-index="${selected}"]`)
      ?.scrollIntoView({ block: 'nearest' })
  }, [selected])

  return (
    <aside className="flex w-full shrink-0 flex-col border-line bg-surface max-lg:h-72 max-lg:border-t lg:w-72 lg:border-l xl:w-96">
      <div className="shrink-0 border-b border-line px-4 py-3">
        <h2 className="text-sm font-semibold text-ink">Translation</h2>
        <p className="mt-0.5 text-xs text-faint">
          {set === 0 ? 'Nothing translated yet' : `${plural(set, 'line')} set on the page`}
        </p>
      </div>

      {set === 0 ? (
        <p className="px-4 py-6 text-sm text-faint">
          Read the page first, then translate it. Each line lands where its original
          was, and can be resized from there.
        </p>
      ) : (
        <ul ref={list} className="min-h-0 flex-1 space-y-2 overflow-y-auto p-3">
          {lettering.map((line, index) => {
            if (line === null) return null

            // A cut needs words on both sides of it, so a cursor at either end
            // of the line is not one.
            const at = caret?.line === index ? caret.at : null
            const splittable =
              at !== null &&
              line.text.slice(0, at).trim() !== '' &&
              line.text.slice(at).trim() !== ''

            return (
              <li
                key={index}
                data-index={index}
                className={`rounded-lg border p-2.5 transition-colors ${
                  selected === index ? 'border-accent bg-accent/10' : 'border-line'
                }`}
                onFocus={() => onSelect(index)}
              >
                <p
                  lang="ja"
                  className="truncate text-[11px] text-faint"
                  title={originals[index] ?? ''}
                >
                  {index + 1}. {originals[index] || '—'}
                </p>

                <textarea
                  value={line.text}
                  rows={2}
                  onChange={(event) => {
                    onChange(index, { text: event.target.value })
                    setCaret({ line: index, at: event.target.selectionStart })
                  }}
                  // Fires whenever the cursor moves, by click or by key.
                  onSelect={(event) =>
                    setCaret({ line: index, at: event.currentTarget.selectionStart })
                  }
                  className={`mt-1 w-full resize-y rounded-md border border-line bg-raised px-2 py-1 text-sm text-ink ${FOCUS}`}
                />

                <div className="mt-1.5 flex items-center gap-2">
                  <p className="flex-1 text-[11px] text-faint tabular-nums">
                    {Math.round(line.size)}px in a {line.box[2] - line.box[0]} ×{' '}
                    {line.box[3] - line.box[1]} box
                    {line.angle > 0 && ` · ${Math.round(line.angle)}°`}
                    {selected === index && (
                      <span className="text-muted"> · ↑↓ to resize, ←→ to turn</span>
                    )}
                  </p>

                  <Button
                    onClick={() => {
                      if (at === null) return
                      onSplit(index, at)
                      setCaret(null)
                    }}
                    disabled={!splittable}
                    title={
                      splittable
                        ? 'Two bubbles in one block: cut the line here, and the block with it'
                        : 'Put the cursor where the second bubble starts, then split'
                    }
                    className="px-2 py-1 text-[11px]"
                  >
                    Split
                  </Button>

                  <Button
                    onClick={() => onFit(index)}
                    title="The largest size that lands in this box, held to the size this page is lettered at — the arrow keys go past that"
                    className="px-2 py-1 text-[11px]"
                  >
                    Fit
                  </Button>
                </div>
              </li>
            )
          })}
        </ul>
      )}
    </aside>
  )
}
