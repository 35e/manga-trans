import type { BoardMode } from '../lib/api'

export type Step = {
  id: BoardMode
  label: string
  /** Finished, so the next one is worth going to. */
  done: boolean
  /** There is something to do here yet: greyed out when there is not. */
  open: boolean
}

type Props = {
  steps: Step[]
  current: BoardMode
  onPick: (step: BoardMode) => void
}

/**
 * The way through a page — find the words, hide them, letter it — as three
 * steps that say which are done and which one is being worked on.
 *
 * They are not a wizard: any of them can be gone to at any time. They are here
 * because the order is the thing that is not obvious, and because a page half
 * done should look half done.
 */
export function Steps({ steps, current, onPick }: Props) {
  return (
    <ol className="flex shrink-0 items-center">
      {steps.map((step, at) => (
        <li key={step.id} className="flex items-center">
          {at > 0 && (
            <span
              aria-hidden="true"
              className={`mx-1 h-px w-4 ${
                steps[at - 1].done
                  ? 'bg-indigo-400'
                  : 'bg-slate-300 dark:bg-white/15'
              }`}
            />
          )}

          <button
            type="button"
            onClick={() => onPick(step.id)}
            aria-current={current === step.id ? 'step' : undefined}
            className={`flex items-center gap-1.5 rounded-lg py-1 pr-2.5 pl-1 text-xs font-medium transition-colors ${
              current === step.id
                ? 'bg-indigo-50 text-indigo-700 dark:bg-indigo-500/15 dark:text-indigo-300'
                : step.open
                  ? 'text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-white/10'
                  : 'text-slate-400 hover:bg-slate-100 dark:text-slate-500 dark:hover:bg-white/10'
            }`}
          >
            <span
              className={`flex size-5 items-center justify-center rounded-full text-[10px] font-semibold tabular-nums ${
                step.done
                  ? 'bg-indigo-600 text-white'
                  : current === step.id
                    ? 'bg-indigo-600/15 text-indigo-700 ring-1 ring-indigo-500 dark:text-indigo-300'
                    : 'bg-slate-200 text-slate-500 dark:bg-white/10 dark:text-slate-400'
              }`}
            >
              {step.done ? (
                <svg
                  aria-hidden="true"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="3.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="size-3"
                >
                  <path d="m5 13 5 5L20 7" />
                </svg>
              ) : (
                at + 1
              )}
              <span className="sr-only">{step.done ? 'done' : `step ${at + 1}`}</span>
            </span>
            {step.label}
          </button>
        </li>
      ))}
    </ol>
  )
}
