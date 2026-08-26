import { useEffect, useState } from 'react'
import type { BatchRun } from '../hooks/useBatch'
import type { Stage } from '../lib/api'
import { plural } from '../lib/images'
import { CloseIcon, DownloadIcon } from './icons'
import { Button, IconButton, Spinner } from './ui'

const LABELS: Record<Stage, string> = {
  detecting: 'Detecting',
  reading: 'Reading',
  tracing: 'Tracing',
  cleaning: 'Cleaning',
  translating: 'Translating',
}

type Props = {
  run: BatchRun
  stage: Stage | null
  onOpen: (id: string) => void
  onStop: () => void
  onDismiss: () => void
  onDownload?: () => void
  onReview?: () => void
  packing: { done: number; total: number } | null
}

export function BatchProgress({
  run,
  stage,
  onOpen,
  onStop,
  onDismiss,
  onDownload,
  onReview,
  packing,
}: Props) {
  const share = run.total === 0 ? 0 : run.done / run.total
  const failed = run.failed.length

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
          <span className="ml-1 font-normal text-faint">
            · {run.phase}
            {run.phases > 1 && ` · ${run.phaseAt + 1} of ${run.phases}`}
          </span>
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
          ) : run.finished ? (
            <>
              {run.stopping ? 'Stopped' : 'Done'}
              {failed > 0 && ` · ${plural(failed, 'page')} fell over`}
            </>
          ) : (
            'Starting…'
          )}
        </p>
        {run.total > 0 && (
          <p className="shrink-0 text-faint tabular-nums">
            {run.done} / {run.total}
          </p>
        )}
      </div>

      {run.finished && failed > 0 && (
        <ul className="mt-1.5 space-y-0.5 border-t border-line pt-1.5 text-[11px] leading-snug text-warn">
          {run.failed.map((page, at) => (
            <li
              key={`${page.id}:${at}`}
              className="truncate"
              title={`${page.name}: ${page.why}`}
            >
              <span className="font-medium">{page.name}</span> — {page.why}
            </li>
          ))}
        </ul>
      )}

      {run.finished && !run.stopping && onReview && (
        <Button variant="outline" onClick={onReview} className="mt-2 w-full">
          Review chapter
        </Button>
      )}

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
