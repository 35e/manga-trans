import type { Brush } from '../lib/mask'

type Props = {
  brush: Brush
  onBrush: (brush: Brush) => void
  onFill: () => void
  onFillLetters: () => void
  canFill: boolean
  tracing: boolean
  onClear: () => void
  canClear: boolean
  spread: number
  onSpread: (spread: number) => void
  note: string | null
}

const SPREADS = [0, 2, 4, 6, 8, 12, 16]

const SIZES = { min: 4, max: 160 }

/** The brush, and the two shortcuts worth having beside it. */
export function MaskTools({
  brush,
  onBrush,
  onFill,
  onFillLetters,
  canFill,
  tracing,
  onClear,
  canClear,
  spread,
  onSpread,
  note,
}: Props) {
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-2 border-b border-slate-200 bg-slate-50 px-4 py-2 dark:border-white/10 dark:bg-white/5">
      <div className="flex rounded-lg border border-slate-300 p-0.5 dark:border-white/15">
        <Tool
          label="Draw"
          active={!brush.erase}
          onClick={() => onBrush({ ...brush, erase: false })}
        />
        <Tool
          label="Erase"
          active={brush.erase}
          onClick={() => onBrush({ ...brush, erase: true })}
        />
      </div>

      <label className="flex items-center gap-2 text-xs font-medium text-slate-600 dark:text-slate-300">
        Size
        <input
          type="range"
          min={SIZES.min}
          max={SIZES.max}
          value={brush.radius}
          onChange={(event) =>
            onBrush({ ...brush, radius: Number(event.target.value) })
          }
          className="w-28 accent-indigo-600"
          aria-label="Brush size"
        />
        <span className="w-10 text-right tabular-nums">{brush.radius * 2}px</span>
      </label>

      {note && (
        <span className="text-xs text-amber-700 dark:text-amber-400">{note}</span>
      )}

      <div className="ml-auto flex items-center gap-2">
        <span className="text-xs font-medium text-slate-500 dark:text-slate-400">
          {tracing ? 'Tracing the lettering…' : 'Fill from'}
        </span>
        <button
          type="button"
          onClick={onFillLetters}
          disabled={!canFill || tracing}
          title="Mark the lettering itself, leaving the art it sits on"
          className="rounded-lg border border-slate-300 px-2.5 py-1.5 text-xs font-medium text-slate-700 transition-colors hover:bg-white disabled:opacity-40 dark:border-white/15 dark:text-slate-200 dark:hover:bg-white/10"
        >
          Letters
        </button>
        <label
          className="flex items-center gap-1 text-xs font-medium text-slate-500 dark:text-slate-400"
          title="How far past the ink the mask reaches, in page pixels. Raise it if edges are left behind."
        >
          +
          <select
            value={spread}
            onChange={(event) => onSpread(Number(event.target.value))}
            aria-label="How far past the ink to mark"
            className="rounded-lg border border-slate-300 bg-white px-1.5 py-1 text-xs text-slate-900 dark:border-white/15 dark:bg-slate-900 dark:text-white"
          >
            {SPREADS.map((size) => (
              <option key={size} value={size}>
                {size}px
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          onClick={onFill}
          disabled={!canFill}
          title="Mark the whole box around every block the detector found"
          className="rounded-lg border border-slate-300 px-2.5 py-1.5 text-xs font-medium text-slate-700 transition-colors hover:bg-white disabled:opacity-40 dark:border-white/15 dark:text-slate-200 dark:hover:bg-white/10"
        >
          Blocks
        </button>
        <button
          type="button"
          onClick={onClear}
          disabled={!canClear}
          className="rounded-lg border border-slate-300 px-2.5 py-1.5 text-xs font-medium text-slate-700 transition-colors hover:bg-white disabled:opacity-40 dark:border-white/15 dark:text-slate-200 dark:hover:bg-white/10"
        >
          Clear mask
        </button>
      </div>
    </div>
  )
}

function Tool({
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
