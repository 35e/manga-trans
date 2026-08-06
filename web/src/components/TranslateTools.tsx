type Props = {
  models: string[]
  model: string
  onModel: (model: string) => void
  target: string
  onTarget: (target: string) => void
  onFitAll: () => void
  canFit: boolean
  note: string | null
}

/** Which model, into what, and the button that sets it going. */
export function TranslateTools({
  models,
  model,
  onModel,
  target,
  onTarget,
  onFitAll,
  canFit,
  note,
}: Props) {
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-2 border-b border-slate-200 bg-slate-50 px-4 py-2 dark:border-white/10 dark:bg-white/5">
      <label className="flex items-center gap-2 text-xs font-medium text-slate-600 dark:text-slate-300">
        Model
        <select
          value={model}
          onChange={(event) => onModel(event.target.value)}
          disabled={models.length === 0}
          className="rounded-lg border border-slate-300 bg-white px-2 py-1 text-xs text-slate-900 disabled:opacity-40 dark:border-white/15 dark:bg-slate-900 dark:text-white"
        >
          {models.length === 0 && <option value="">none found</option>}
          {models.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
      </label>

      <label className="flex items-center gap-2 text-xs font-medium text-slate-600 dark:text-slate-300">
        Into
        <input
          value={target}
          onChange={(event) => onTarget(event.target.value)}
          spellCheck={false}
          className="w-28 rounded-lg border border-slate-300 bg-white px-2 py-1 text-xs text-slate-900 dark:border-white/15 dark:bg-slate-900 dark:text-white"
        />
      </label>

      {note && (
        <span className="text-xs text-amber-700 dark:text-amber-400">{note}</span>
      )}

      <div className="ml-auto flex items-center gap-2">
        <button
          type="button"
          onClick={onFitAll}
          disabled={!canFit}
          title="Set every line at the largest size that lands in its box"
          className="rounded-lg border border-slate-300 px-2.5 py-1.5 text-xs font-medium text-slate-700 transition-colors hover:bg-white disabled:opacity-40 dark:border-white/15 dark:text-slate-200 dark:hover:bg-white/10"
        >
          Fit all
        </button>
      </div>
    </div>
  )
}
