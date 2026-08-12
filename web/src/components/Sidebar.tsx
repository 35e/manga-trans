import { useEffect, useRef, useState } from 'react'
import type { BatchRun } from '../hooks/useBatch'
import type { LibraryNotice } from '../hooks/useImageLibrary'
import type { Stage } from '../lib/api'
import type { GalleryFolder, GalleryImage } from '../lib/images'
import { formatBytes, plural } from '../lib/images'
import { BatchProgress } from './BatchProgress'
import { Dropzone } from './Dropzone'
import { Gallery } from './Gallery'
import { BackIcon, DownloadIcon } from './icons'
import { Button, FOCUS, Note, Spinner } from './ui'

type Props = {
  images: GalleryImage[]
  folders: GalleryFolder[]
  activeId: string | null
  onOpen: (id: string) => void
  onRemove: (id: string) => void
  onRemoveFolder: (id: string) => void
  onFiles: (files: FileList | File[] | null) => void
  dragging: boolean
  busy: boolean
  notice: LibraryNotice | null
  onDismissNotice: () => void
  onClearAll: () => void
  /** Running a whole folder: what is happening, and how to start and stop it. */
  batch: BatchRun | null
  batchStage: Stage | null
  onRunFolder: (folder: GalleryFolder) => void
  onStopBatch: () => void
  onDismissBatch: () => void
  /** Whether a model has been picked, without which a run only cleans. */
  canTranslate: boolean
  /** The pages that have been lettered, which a run would do over. */
  lettered: string[]
  /** A folder back out as the archive it came in as. */
  onDownloadFolder: (folder: GalleryFolder) => void
  /** The pages worth putting in one: anything cleaned or lettered. */
  workedOn: string[]
  /** How far through packing one, or null when nothing is being packed. */
  packing: { done: number; total: number } | null
}

/** The rail down the side: what to drop pages into, and every page dropped in. */
export function Sidebar({
  images,
  folders,
  activeId,
  onOpen,
  onRemove,
  onRemoveFolder,
  onFiles,
  dragging,
  busy,
  notice,
  onDismissNotice,
  onClearAll,
  batch,
  batchStage,
  onRunFolder,
  onStopBatch,
  onDismissBatch,
  canTranslate,
  lettered,
  onDownloadFolder,
  workedOn,
  packing,
}: Props) {
  const [open, setOpen] = useState<string | null>(null)
  const folder = folders.find((held) => held.id === open) ?? null

  // A folder that was deleted while it was open leaves nothing to be inside of.
  useEffect(() => {
    if (open !== null && !folders.some((held) => held.id === open)) setOpen(null)
  }, [open, folders])

  // Anything dropped in while a folder is open lands outside it — loose, or in a
  // folder of its own — so the rail comes back out to where it can be seen.
  const held = useRef(images.length)
  useEffect(() => {
    const grew = images.length > held.current
    held.current = images.length
    if (grew) setOpen(null)
  }, [images.length])

  // Counted for whatever is being looked at: inside a folder, that folder.
  const counted = folder
    ? images.filter((image) => image.folder === folder.id)
    : images
  const total = counted.reduce((sum, image) => sum + image.size, 0)

  return (
    <aside className="flex shrink-0 flex-col border-line bg-surface max-lg:h-64 max-lg:border-b lg:w-60 lg:border-r xl:w-72">
      <div className="shrink-0 p-3">
        <Dropzone onFiles={onFiles} dragging={dragging} busy={busy} />
      </div>

      {notice && (
        <div className="mx-3 mb-3 flex shrink-0 items-start justify-between gap-2 rounded-lg border border-warn/30 bg-warn/10 px-2.5 py-2 text-[11px] leading-snug text-warn">
          <span>{notice.text}</span>
          <button
            type="button"
            onClick={onDismissNotice}
            aria-label="Dismiss"
            className="shrink-0 font-semibold hover:underline"
          >
            ✕
          </button>
        </div>
      )}

      {batch && (
        <BatchProgress
          run={batch}
          stage={batchStage}
          onOpen={onOpen}
          onStop={onStopBatch}
          onDismiss={onDismissBatch}
          onDownload={
            // The folder it ran through, which is not always the one open.
            folders.some((held) => held.id === batch.folder)
              ? () => {
                  const ran = folders.find((held) => held.id === batch.folder)
                  if (ran) onDownloadFolder(ran)
                }
              : undefined
          }
          packing={packing}
        />
      )}

      {folder && (
        <FolderBar
          folder={folder}
          pages={counted}
          lettered={counted.filter((image) => lettered.includes(image.id)).length}
          done={counted.filter((image) => workedOn.includes(image.id)).length}
          onBack={() => setOpen(null)}
          onRun={() => onRunFolder(folder)}
          onDownload={() => onDownloadFolder(folder)}
          running={batch !== null && !batch.finished}
          packing={packing}
          canTranslate={canTranslate}
        />
      )}

      <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-3">
        <Gallery
          images={images}
          folders={folders}
          open={open}
          onOpenFolder={setOpen}
          activeId={activeId}
          onOpen={onOpen}
          onRemove={onRemove}
          onRemoveFolder={onRemoveFolder}
        />
      </div>

      {counted.length > 0 && (
        <div className="flex shrink-0 items-center justify-between gap-2 border-t border-line px-3 py-2 text-[11px] text-faint">
          <span className="tabular-nums">
            {plural(counted.length, 'page')} · {formatBytes(total)}
          </span>
          <ClearAll onClear={onClearAll} />
        </div>
      )}
    </aside>
  )
}

/**
 * The head of an opened folder: the way back out, and the one button that runs
 * the whole chapter through. Armed first when there is lettering in the folder
 * already, since a run does every page over and hand work is not recoverable.
 */
function FolderBar({
  folder,
  pages,
  lettered,
  done,
  onBack,
  onRun,
  onDownload,
  running,
  packing,
  canTranslate,
}: {
  folder: GalleryFolder
  pages: GalleryImage[]
  lettered: number
  /** Pages that have been cleaned or lettered, so there is something to save. */
  done: number
  onBack: () => void
  onRun: () => void
  onDownload: () => void
  running: boolean
  packing: { done: number; total: number } | null
  canTranslate: boolean
}) {
  const [armed, setArmed] = useState(false)

  useEffect(() => {
    if (!armed) return
    const timer = setTimeout(() => setArmed(false), 4000)
    return () => clearTimeout(timer)
  }, [armed])

  // Nothing to lose, so nothing to ask about.
  useEffect(() => {
    if (lettered === 0) setArmed(false)
  }, [lettered])

  return (
    <div className="mx-3 mb-3 shrink-0 rounded-lg border border-line bg-raised px-2 py-2">
      <div className="flex items-center gap-1">
        <button
          type="button"
          onClick={onBack}
          aria-label="Back to everything"
          title="Back to everything"
          className={`-ml-1 grid size-6 shrink-0 place-items-center rounded-md text-faint transition-colors hover:bg-surface hover:text-ink ${FOCUS}`}
        >
          <BackIcon className="size-4" />
        </button>
        <p className="min-w-0 flex-1 truncate text-[11px] font-medium text-ink">
          {folder.name}
        </p>
        <span className="shrink-0 text-[11px] text-faint tabular-nums">
          {plural(pages.length, 'page')}
        </span>
      </div>

      <Button
        variant={armed ? 'primary' : 'outline'}
        onClick={() => {
          if (!armed && lettered > 0) setArmed(true)
          else {
            setArmed(false)
            onRun()
          }
        }}
        onBlur={() => setArmed(false)}
        disabled={running || pages.length === 0}
        className="mt-2 w-full"
        title={
          running
            ? 'A folder is already being run'
            : `Detect, clean and letter all ${plural(pages.length, 'page')}`
        }
      >
        {armed
          ? `Do ${plural(lettered, 'lettered page')} again?`
          : canTranslate
            ? 'Clean & translate all'
            : 'Clean all'}
      </Button>

      {/* Every page goes in, at the best state it reached — so this is offered as
          soon as any one of them has been worked on, not only once the whole
          chapter is through. Nothing worked on is nothing but the originals back,
          which is not worth handing over. */}
      <Button
        variant="outline"
        onClick={onDownload}
        disabled={running || packing !== null || done === 0}
        className="mt-1.5 w-full"
        title={
          done === 0
            ? 'Nothing has been cleaned or lettered in this folder yet'
            : `Save all ${plural(pages.length, 'page')} as one archive`
        }
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

      {!canTranslate && (
        <p className="mt-1.5 leading-snug">
          <Note>no model picked — pages will be cleaned but not translated</Note>
        </p>
      )}
    </div>
  )
}

/** Two taps to empty the gallery, without a browser dialog. */
function ClearAll({ onClear }: { onClear: () => void }) {
  const [armed, setArmed] = useState(false)

  useEffect(() => {
    if (!armed) return
    const timer = setTimeout(() => setArmed(false), 4000)
    return () => clearTimeout(timer)
  }, [armed])

  return (
    <button
      type="button"
      onClick={() => {
        if (armed) onClear()
        setArmed(!armed)
      }}
      onBlur={() => setArmed(false)}
      className={`shrink-0 rounded-md px-2 py-1 font-medium transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-danger ${
        armed ? 'bg-danger text-white' : 'text-faint hover:bg-danger/15 hover:text-danger'
      }`}
    >
      {armed ? 'Sure?' : 'Clear all'}
    </button>
  )
}
