import { useEffect, useState } from 'react'
import type { BatchRun } from '../hooks/useBatch'
import type { Stage } from '../lib/api'
import { plural } from '../lib/images'
import { CloseIcon, DownloadIcon } from './icons'
import { Button, IconButton, Spinner } from './ui'

/** What the page in hand is having done to it, said the way the board says it. */
const LABELS: Record<Stage, string> = {
  detecting: 'Detecting',
  reading: 'Reading',
  tracing: 'Tracing',
  cleaning: 'Cleaning',
  translating: 'Translating',
  surveying: 'Reading the chapter',
}

type Props = {
  run: BatchRun
  /** What the API is doing for the page in hand, if it is doing anything. */
  stage: Stage | null
  onOpen: (id: string) => void
  onStop: () => void
  onDismiss: () => void
  /** The folder back out as an archive, once there is one to hand back. */
  onDownload?: () => void
  packing: { done: number; total: number } | null
}

/** How far a folder run has got, shown wherever the rail is. */
export function BatchProgress({
  run,
  stage,
  onOpen,
  onStop,
  onDismiss,
  onDownload,
  packing,
}: Props) {
  const share = run.total === 0 ? 0 : run.done / run.total
  const failed = run.failed.length

  // Held between two steps, or the word blinks out in the gap and reads as
  // something having gone wrong. Cleared before it is filled in, in that order.
  const [held, setHeld] = useState<Stage | null>(stage)
  useEffect(() => setHeld(null), [run.page?.id])
  useEffect(() => {
    if (stage) setHeld(stage)
  }, [stage])
  const doing = stage ?? held

  return (
    <section className="mx-3 mb-3 shrink-0 rounded-lg border border-line bg-raised px-2.5 py-2">
      <div className="flex items-center justify-between gap-2">
        <p className="min-w-0 truncate text-[11px] font-medium text-ink">
          {run.label}
          {/* Which of the two runs this is: a folder is read and then lettered,
              and the bar has to say which of them it is showing. */}
          <span className="ml-1 font-normal text-faint">· {run.phase}</span>
        </p>
        {run.finished ? (
          <IconButton label="Dismiss" onClick={onDismiss} className="-my-1 size-5">
            <CloseIcon className="size-3" />
          </IconButton>
        ) : (
          <button
            type="button"
            onClick={onStop}
            disabled={run.stopping}
            className="shrink-0 rounded-md px-1.5 py-0.5 text-[11px] font-medium text-faint transition-colors hover:bg-danger/15 hover:text-danger disabled:opacity-50 disabled:hover:bg-transparent disabled:hover:text-faint"
            title="Stop once the page in hand is finished"
          >
            {run.stopping ? 'Stopping…' : 'Stop'}
          </button>
        )}
      </div>

      <div
        role="progressbar"
        aria-label={`${run.label} progress`}
        aria-valuemin={0}
        aria-valuemax={run.total}
        aria-valuenow={run.done}
        aria-valuetext={`${run.done} of ${plural(run.total, 'page')}`}
        className="mt-1.5 h-1 overflow-hidden rounded-full bg-canvas"
      >
        <div
          className={`h-full rounded-full transition-[width] duration-300 ${
            failed > 0 ? 'bg-warn' : 'bg-accent'
          }`}
          style={{ width: `${Math.round(share * 100)}%` }}
        />
      </div>

      <div className="mt-1.5 flex items-baseline justify-between gap-2 text-[11px]">
        <p className="min-w-0 truncate text-faint">
          {run.page ? (
            <>
              {!run.finished && <Spinner className="mr-1 inline size-2.5 align-[-1px]" />}
              {doing ? `${LABELS[doing]} ` : ''}
              <button
                type="button"
                onClick={() => onOpen(run.page?.id ?? '')}
                className="underline decoration-dotted underline-offset-2 hover:text-ink"
                title="Put this page on the board"
              >
                {run.page.name}
              </button>
            </>
          ) : run.note ? (
            // Work on the whole chapter, so there is no page to name.
            <>
              <Spinner className="mr-1 inline size-2.5 align-[-1px]" />
              {run.note}
            </>
          ) : run.finished ? (
            <>
              {run.stopping ? 'Stopped' : 'Done'}
              {failed > 0 && ` · ${plural(failed, 'page')} fell over`}
            </>
          ) : (
            'Starting…'
          )}
        </p>
        <p className="shrink-0 text-faint tabular-nums">
          {run.done} / {run.total}
        </p>
      </div>

      {run.finished && failed > 0 && (
        <ul className="mt-1.5 space-y-0.5 border-t border-line pt-1.5 text-[11px] leading-snug text-warn">
          {run.failed.map((page, at) => (
            <li
              key={`${page.name}:${at}`}
              className="truncate"
              title={`${page.name}: ${page.why}`}
            >
              <span className="font-medium">{page.name}</span> — {page.why}
            </li>
          ))}
        </ul>
      )}

      {/* The moment the chapter is actually ready, which is the moment a person
          wants it. A run stopped part way still has pages worth saving, so this
          is offered then too — the archive holds every page either way. */}
      {run.finished && onDownload && (
        <Button
          variant="outline"
          onClick={onDownload}
          disabled={packing !== null}
          className="mt-2 w-full"
          title={`Save ${run.label} as one archive`}
        >
          {packing ? (
            <>
              <Spinner className="mr-1.5 inline size-3 align-[-2px]" />
              Packing {packing.done} / {packing.total}…
            </>
          ) : (
            <>
              <DownloadIcon className="mr-1.5 inline size-3.5 align-[-3px]" />
              Download chapter
            </>
          )}
        </Button>
      )}
    </section>
  )
}
