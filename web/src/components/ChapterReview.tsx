import type { Analysis } from '../lib/api'
import type { GalleryFolder, GalleryImage } from '../lib/images'
import { plural } from '../lib/images'
import type { Lines } from '../lib/lettering'
import type { PageReview } from '../lib/review'
import { SAYS, reviewed } from '../lib/review'
import { CloseIcon, DownloadIcon } from './icons'
import { Button, FOCUS, IconButton, Spinner } from './ui'

type Props = {
  folder: GalleryFolder
  pages: GalleryImage[]
  analyses: Record<string, Analysis>
  lettering: Record<string, Lines>
  failed: Record<string, string>
  onEdit: (id: string) => void
  onClose: () => void
  onDownload: () => void
  packing: { done: number; total: number } | null
}

export function ChapterReview({
  folder,
  pages,
  analyses,
  lettering,
  failed,
  onEdit,
  onClose,
  onDownload,
  packing,
}: Props) {
  return (
    <section className="flex min-h-0 min-w-0 flex-1 flex-col bg-canvas">
      <header className="flex shrink-0 items-center gap-4 border-b border-line bg-surface px-4 py-2.5">
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium text-ink">{folder.name}</p>
          <p className="text-xs text-faint">
            {plural(pages.length, 'page')} · read it through before you save it
          </p>
        </div>

        <Button
          variant="primary"
          size="md"
          onClick={onDownload}
          disabled={packing !== null}
          title={`Save ${folder.name} as one archive`}
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

        <IconButton label="Close the review" onClick={onClose}>
          <CloseIcon className="size-4" />
        </IconButton>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto">
        <ol className="mx-auto max-w-3xl space-y-2 p-4">
          {pages.map((page, at) => (
            <Row
              key={page.id}
              page={page}
              at={at}
              review={reviewed(analyses[page.id], lettering[page.id], failed[page.id])}
              onEdit={() => onEdit(page.id)}
            />
          ))}
        </ol>
      </div>
    </section>
  )
}

function Row({
  page,
  at,
  review,
  onEdit,
}: {
  page: GalleryImage
  at: number
  review: PageReview
  onEdit: () => void
}) {
  return (
    <li className="flex items-start gap-3 rounded-lg border border-line bg-surface p-2.5">
      <img
        src={page.url}
        alt=""
        className="h-20 w-14 shrink-0 rounded-md border border-line object-cover"
        draggable={false}
      />

      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
          <span className="text-xs font-medium text-ink tabular-nums">p{at + 1}</span>
          <span className="min-w-0 truncate text-xs text-faint">{page.name}</span>
          <span className="text-xs text-faint tabular-nums">
            {plural(review.lines.length, 'line')}
          </span>
          {review.troubles.map((trouble) => (
            <span
              key={trouble}
              className="rounded-md bg-warn/15 px-1.5 py-0.5 text-[11px] font-medium text-warn"
            >
              {SAYS[trouble]}
            </span>
          ))}
        </div>

        {review.lines.length === 0 ? (
          <p className="mt-1 text-xs text-faint italic">nothing lettered</p>
        ) : (
          <ul className="mt-1 space-y-0.5">
            {review.lines.slice(0, 2).map((line, index) => (
              <li key={index} className="truncate text-xs text-muted">
                “{line}”
              </li>
            ))}
          </ul>
        )}
      </div>

      <button
        type="button"
        onClick={onEdit}
        className={`shrink-0 self-center rounded-lg border border-line px-2.5 py-1.5 text-xs font-medium text-muted transition-colors hover:bg-raised hover:text-ink ${FOCUS}`}
        title="Put this page on the board"
      >
        Edit ▸
      </button>
    </li>
  )
}
