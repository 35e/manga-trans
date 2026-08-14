import type { BoardMode } from '../lib/api'
import { CheckIcon } from './icons'
import { FOCUS } from './ui'

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
 * The way through a page as three steps saying which are done. Not a wizard: any
 * of them can be gone to at any time.
 */
export function Steps({ steps, current, onPick }: Props) {
  return (
    <ol className="flex shrink-0 items-center gap-0.5 rounded-lg border border-line bg-canvas p-0.5">
      {steps.map((step, at) => (
        <li key={step.id} className="flex items-center">
          <button
            type="button"
            onClick={() => onPick(step.id)}
            aria-current={current === step.id ? 'step' : undefined}
            className={`flex items-center gap-1.5 rounded-md py-2 pr-2.5 pl-1.5 text-xs font-medium whitespace-nowrap transition-colors ${FOCUS} ${
              current === step.id
                ? 'bg-raised text-ink'
                : step.open
                  ? 'text-muted hover:bg-raised/60 hover:text-ink'
                  : 'text-faint hover:bg-raised/60'
            }`}
          >
            <span
              className={`grid size-4 place-items-center rounded-full text-[10px] font-semibold tabular-nums ${
                step.done
                  ? 'bg-accent text-white'
                  : current === step.id
                    ? 'text-accent-lit ring-1 ring-accent'
                    : 'text-faint ring-1 ring-line'
              }`}
            >
              {step.done ? <CheckIcon className="size-2.5" /> : at + 1}
              <span className="sr-only">{step.done ? 'done' : `step ${at + 1}`}</span>
            </span>
            {step.label}
          </button>
        </li>
      ))}
    </ol>
  )
}
