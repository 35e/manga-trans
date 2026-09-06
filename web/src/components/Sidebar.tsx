import { useEffect, useId, useRef, useState } from 'react'
import type { BatchRun } from '../hooks/useBatch'
import type { LibraryNotice } from '../hooks/useImageLibrary'
import type { Stage } from '../lib/api'
import type { GalleryFolder, GalleryImage } from '../lib/images'
import { formatBytes, plural } from '../lib/images'
import { BatchProgress } from './BatchProgress'
import { Dropzone } from './Dropzone'
import { Gallery } from './Gallery'
import { BackIcon, DownloadIcon, PlusIcon } from './icons'
import { Button, FOCUS, Note, Spinner, TextInput } from './ui'

type Props = {
  images: GalleryImage[]
  folders: GalleryFolder[]
  open: string | null
  onOpenFolder: (id: string | null) => void
  onNewFolder: (name: string) => boolean
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
  batch: BatchRun | null
  batchStage: Stage | null
  onTranslateFolder: (folder: GalleryFolder) => void
  onReprocessFolder: (folder: GalleryFolder) => void
  onStopBatch: () => void
  onDismissBatch: () => void
  onReviewBatch: () => void
  canTranslate: boolean
  lettered: string[]
  onDownloadFolder: (folder: GalleryFolder) => void
  workedOn: string[]
  packing: { done: number; total: number } | null
}

export function Sidebar({
  images,
  folders,
  open,
  onOpenFolder,
  onNewFolder,
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
  onTranslateFolder,
  onReprocessFolder,
  onStopBatch,
  onDismissBatch,
  onReviewBatch,
  canTranslate,
  lettered,
  onDownloadFolder,
  workedOn,
  packing,
}: Props) {
  const [pagesExpanded, setPagesExpanded] = useState(false)
  const pagesId = useId()
  const active = images.find((image) => image.id === activeId)
  const siblings = active ? images.filter((image) => image.folder === active.folder) : []
  const position = siblings.findIndex((image) => image.id === activeId)
  const activeFolder = folders.find((held) => held.id === active?.folder)
  const openPage = (id: string) => {
    const image = images.find((held) => held.id === id)
    if (!image) return
    onOpenFolder(image.folder ?? null)
    onOpen(id)
  }
  const folder = folders.find((held) => held.id === open) ?? null

  const counted = folder
    ? images.filter((image) => image.folder === folder.id)
    : images
  const total = counted.reduce((sum, image) => sum + image.size, 0)

  const inside = folder ? counted.length : 0
  const held = useRef({ total: images.length, inside })
  useEffect(() => {
    const was = held.current
    held.current = { total: images.length, inside }
    if (images.length > was.total && inside === was.inside) onOpenFolder(null)
  }, [images.length, inside, onOpenFolder])

  return (
    <aside aria-label="Pages" className="flex shrink-0 flex-col overflow-hidden border-line bg-surface max-lg:border-b lg:w-60 lg:border-r xl:w-72">
      <div className="flex shrink-0 items-center justify-between gap-2 px-3 pt-3">
        <h2 className="text-xs font-semibold text-ink">Pages</h2>
        <Button
          variant="ghost"
          aria-expanded={pagesExpanded}
          aria-controls={pagesId}
          onClick={() => setPagesExpanded(!pagesExpanded)}
          className="lg:hidden"
        >
          {pagesExpanded ? 'Hide pages' : 'Show pages'}
        </Button>
      </div>
      <div className="shrink-0 space-y-2 p-3">
        <Dropzone onFiles={onFiles} dragging={dragging} busy={busy} />
      </div>

      {active && (
        <nav aria-label="Active page navigation" className="shrink-0 space-y-1.5 border-b border-line px-3 pb-3">
          <p className="truncate text-[11px] text-faint" title={activeFolder?.name ?? 'Loose pages'}>
            {activeFolder?.name ?? (active.folder ? 'Folder pages' : 'Loose pages')}
          </p>
          <div className="flex items-center justify-between gap-1">
            <Button
              aria-label="Previous page"
              disabled={position <= 0}
              onClick={() => openPage(siblings[position - 1].id)}
            >
              Previous
            </Button>
            <span aria-live="polite" aria-atomic="true" className="text-center text-[11px] text-muted tabular-nums">
              Page {position + 1} of {siblings.length}
            </span>
            <Button
              aria-label="Next page"
              disabled={position < 0 || position >= siblings.length - 1}
              onClick={() => openPage(siblings[position + 1].id)}
            >
              Next
            </Button>
          </div>
        </nav>
      )}


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
          onOpen={openPage}
          onStop={onStopBatch}
          onDismiss={onDismissBatch}
          onReview={onReviewBatch}
          onDownload={
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

      <div
        id={pagesId}
        className={`${pagesExpanded ? 'flex' : 'hidden'} min-h-0 flex-1 flex-col max-lg:max-h-64 lg:flex`}
      >
        {!folder && (
          <div className="shrink-0 space-y-2 p-3">
            <NewFolder onCreate={onNewFolder} />
            <p className="text-[11px] font-medium text-faint">Folders &amp; loose pages</p>
          </div>
        )}
      <div className="min-h-0 flex-1 overflow-y-auto pt-3">
        {folder && (
          <FolderBar
            folder={folder}
            pages={counted}
            lettered={counted.filter((image) => lettered.includes(image.id)).length}
            done={counted.filter((image) => workedOn.includes(image.id)).length}
            onBack={() => onOpenFolder(null)}
            onTranslate={() => onTranslateFolder(folder)}
            onReprocess={() => onReprocessFolder(folder)}
            onDownload={() => onDownloadFolder(folder)}
            running={batch !== null && !batch.finished}
            packing={packing}
            canTranslate={canTranslate}
          />
        )}

        <div className="px-3 pb-3">
          <Gallery
            images={images}
            folders={folders}
            open={open}
            onOpenFolder={onOpenFolder}
            activeId={activeId}
            onOpen={openPage}
            onRemove={onRemove}
            onRemoveFolder={onRemoveFolder}
          />
        </div>
      </div>

      {counted.length > 0 && (
        <div className="flex shrink-0 items-center justify-between gap-2 border-t border-line px-3 py-2 text-[11px] text-faint">
          <span className="tabular-nums">
            {plural(counted.length, 'page')} · {formatBytes(total)}
          </span>
          <ClearAll onClear={onClearAll} />
        </div>
      )}
      </div>
    </aside>
  )
}

function NewFolder({ onCreate }: { onCreate: (name: string) => boolean }) {
  const [naming, setNaming] = useState(false)
  const [name, setName] = useState('')
  const field = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (naming) field.current?.focus()
  }, [naming])

  if (!naming) {
    return (
      <Button
        variant="outline"
        onClick={() => setNaming(true)}
        className="w-full"
        title="Start an empty folder. Pages dropped in while it is open go into it"
      >
        <PlusIcon className="mr-1.5 inline size-3.5 align-[-3px]" />
        New folder
      </Button>
    )
  }

  const settle = () => {
    const called = name.trim()
    if (!called) {
      setNaming(false)
      return
    }
    if (onCreate(called)) {
      setName('')
      setNaming(false)
    }
  }

  return (
    <div className="flex items-center gap-1.5">
      <TextInput
        ref={field}
        value={name}
        onChange={(event) => setName(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === 'Enter') settle()
          if (event.key === 'Escape') {
            setName('')
            setNaming(false)
          }
        }}
        onBlur={settle}
        placeholder="Chapter name"
        aria-label="Name for the new folder"
        spellCheck={false}
        className="min-w-0 flex-1"
      />
      <Button variant="primary" onClick={settle} title="Make the folder">
        Add
      </Button>
    </div>
  )
}

function FolderBar({
  folder,
  pages,
  lettered,
  done,
  onBack,
  onTranslate,
  onReprocess,
  onDownload,
  running,
  packing,
  canTranslate,
}: {
  folder: GalleryFolder
  pages: GalleryImage[]
  lettered: number
  done: number
  onBack: () => void
  onTranslate: () => void
  onReprocess: () => void
  onDownload: () => void
  running: boolean
  packing: { done: number; total: number } | null
  canTranslate: boolean
}) {

  return (
    <div className="mx-3 mb-3 rounded-lg border border-line bg-raised px-2 py-2">
      <div className="flex items-center gap-1">
        <button
          type="button"
          onClick={onBack}
          aria-label="Back to all pages and folders"
          title="Back to all pages and folders"
          className={`-ml-1 grid size-6 shrink-0 place-items-center rounded-md text-faint transition-colors hover:bg-surface hover:text-ink ${FOCUS}`}
        >
          <BackIcon className="size-4" />
        </button>
        <span className="shrink-0 text-[11px] text-faint">All pages /</span>
        <p title={folder.name} className="min-w-0 flex-1 truncate text-[11px] font-medium text-ink">
          {folder.name}
        </p>
        <span className="shrink-0 text-[11px] text-faint tabular-nums">
          {plural(pages.length, 'page')}
        </span>
      </div>

      <Button
        variant="outline"
        onClick={onTranslate}
        disabled={running || pages.length === 0}
        className="mt-2 w-full"
        title={
          running
            ? 'A folder is already being run'
            : 'Resume unfinished stages and keep saved regions, translations, and cleanup'
        }
      >
        {done > 0 || lettered > 0
          ? 'Resume folder'
          : canTranslate
            ? 'Translate folder'
            : 'Read & clean folder'}
      </Button>

      <Button
        variant="ghost"
        onClick={onReprocess}
        disabled={running || pages.length === 0}
        className="mt-1 w-full"
        title="Discard this folder’s derived work and start again from the original pages"
      >
        Reprocess all pages
      </Button>

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
          <Note>
            no model picked — pages will be read and cleaned but not translated
          </Note>
        </p>
      )}

    </div>
  )
}

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
