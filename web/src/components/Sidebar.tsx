import { useEffect, useRef, useState } from 'react'
import type { BatchRun } from '../hooks/useBatch'
import type { LibraryNotice } from '../hooks/useImageLibrary'
import type { CastMember, Fact, Stage, Story, Term } from '../lib/api'
import type { GalleryFolder, GalleryImage } from '../lib/images'
import { formatBytes, plural } from '../lib/images'
import { isEmpty } from '../lib/story'
import { BatchProgress } from './BatchProgress'
import { Dropzone } from './Dropzone'
import { Gallery } from './Gallery'
import { BackIcon, DownloadIcon, PlusIcon } from './icons'
import { Button, FOCUS, Note, Select, Spinner, TextInput } from './ui'

type Props = {
  images: GalleryImage[]
  folders: GalleryFolder[]
  /** Which folder is being looked into, held above so a drop can land in it. */
  open: string | null
  onOpenFolder: (id: string | null) => void
  /** Start one of your own. False when the name is already taken. */
  onNewFolder: (name: string) => boolean
  /** What the open folder has settled on translating its names as. */
  terms: Term[]
  /** Where the open folder's chapter has got to, as its last page left it. */
  story: Story | null
  /** A fact about one of the cast, put right by hand. */
  onCorrect: (name: string, fact: Fact, value: string) => void
  onForgetTerms: () => void
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
  open,
  onOpenFolder,
  onNewFolder,
  terms,
  story,
  onCorrect,
  onForgetTerms,
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
  const folder = folders.find((held) => held.id === open) ?? null

  // Counted for whatever is being looked at: inside a folder, that folder.
  const counted = folder
    ? images.filter((image) => image.folder === folder.id)
    : images
  const total = counted.reduce((sum, image) => sum + image.size, 0)

  // A page dropped while a folder is open lands in it; an archive makes one of
  // its own. Only the second sends the rail back out — going out to look at a
  // folder you did not drop into, and staying put when you dropped into this one.
  const inside = folder ? counted.length : 0
  const held = useRef({ total: images.length, inside })
  useEffect(() => {
    const was = held.current
    held.current = { total: images.length, inside }
    if (images.length > was.total && inside === was.inside) onOpenFolder(null)
  }, [images.length, inside, onOpenFolder])

  return (
    <aside className="flex shrink-0 flex-col border-line bg-surface max-lg:h-64 max-lg:border-b lg:w-60 lg:border-r xl:w-72">
      <div className="shrink-0 space-y-2 p-3">
        <Dropzone onFiles={onFiles} dragging={dragging} busy={busy} />
        {!folder && <NewFolder onCreate={onNewFolder} />}
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
          onBack={() => onOpenFolder(null)}
          onRun={() => onRunFolder(folder)}
          onDownload={() => onDownloadFolder(folder)}
          running={batch !== null && !batch.finished}
          packing={packing}
          canTranslate={canTranslate}
          terms={terms}
          story={story}
          onCorrect={onCorrect}
          onForgetTerms={onForgetTerms}
        />
      )}

      <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-3">
        <Gallery
          images={images}
          folders={folders}
          open={open}
          onOpenFolder={onOpenFolder}
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
 * A folder started by hand, for pages that did not arrive as a chapter.
 *
 * Named as it is made, there being nothing anywhere that renames one — and the
 * field is left open on a name already taken, which is the only way it is
 * refused, so the next one typed lands somewhere.
 */
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
    // Nothing typed is a change of mind, not a folder called nothing.
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
  terms,
  story,
  onCorrect,
  onForgetTerms,
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
  terms: Term[]
  story: Story | null
  onCorrect: (name: string, fact: Fact, value: string) => void
  onForgetTerms: () => void
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

      {(terms.length > 0 || !isEmpty(story)) && (
        <Chapter
          terms={terms}
          story={story}
          onCorrect={onCorrect}
          onForget={onForgetTerms}
        />
      )}
    </div>
  )
}

/**
 * What this chapter has settled on calling things and where its story has got to.
 * Both go over with every page in the folder, so a name keeps one spelling and a
 * page that opens mid-argument is translated as that rather than as strangers
 * meeting — across pages no model ever sees together.
 *
 * Shown rather than only used, because the first page to name someone decides it
 * for the rest of the chapter and there would otherwise be nothing saying why a
 * later page came out as it did. The story is the other way round — the last page
 * to speak decides it — which is worth being able to see for the same reason.
 *
 * The cast is the one part that can be corrected: a chapter says who someone is
 * when it is ready to, which may be pages after the first translation needed to
 * know, and someone reading it already knows. Anything set here is settled — the
 * model is told so on every page after it, and stops being asked to work it out.
 */
function Chapter({
  terms,
  story,
  onCorrect,
  onForget,
}: {
  terms: Term[]
  story: Story | null
  onCorrect: (name: string, fact: Fact, value: string) => void
  onForget: () => void
}) {
  return (
    <div className="mt-2 border-t border-line pt-2">
      <div className="flex items-center justify-between gap-2">
        <p
          className="text-[11px] font-medium text-faint"
          title="What this chapter has been translated with so far: where the story has got to, who is in it, and the names and coinages already settled. Every page in the folder is translated against them"
        >
          Chapter so far
        </p>
        <button
          type="button"
          onClick={onForget}
          title="Start over — the pages already lettered are left as they are"
          className={`shrink-0 rounded-md px-1.5 py-0.5 text-[11px] text-faint transition-colors hover:bg-surface hover:text-ink ${FOCUS}`}
        >
          Forget
        </button>
      </div>
      {story && story.scene !== '' && (
        <p
          className="mt-1 max-h-16 overflow-y-auto text-[11px] leading-snug text-muted italic"
          title={story.scene}
        >
          {story.scene}
        </p>
      )}
      {story && story.cast.length > 0 && (
        <ul className="mt-1.5 max-h-32 space-y-1 overflow-y-auto">
          {story.cast.map((person) => (
            <Person key={person.name} person={person} onCorrect={onCorrect} />
          ))}
        </ul>
      )}
      <ul className="mt-1 max-h-24 overflow-y-auto">
        {terms.map((term) => (
          <li
            key={term.source}
            // What it is, where the page that named it said: too long for the row
            // and the row is one line on purpose.
            title={
              term.note ? `${term.source} → ${term.target} — ${term.note}` : undefined
            }
            className="flex items-baseline gap-1 text-[11px] leading-snug"
          >
            <span className="min-w-0 shrink truncate text-muted">{term.source}</span>
            <span className="shrink-0 text-faint" aria-hidden>
              →
            </span>
            <span className="min-w-0 flex-1 truncate text-ink">{term.target}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

/**
 * One of the cast, and the two things about them that can be put right.
 *
 * `unknown` is not a gap to be filled in for the sake of it: it is what the
 * chapter has honestly shown so far, and setting it back to unknown hands the
 * question to the model again rather than asserting anything.
 */
function Person({
  person,
  onCorrect,
}: {
  person: CastMember
  onCorrect: (name: string, fact: Fact, value: string) => void
}) {
  const [noting, setNoting] = useState(false)
  const [note, setNote] = useState(person.note ?? '')
  const settled = person.settled ?? []

  const settle = () => {
    setNoting(false)
    if (note !== (person.note ?? '')) onCorrect(person.name, 'note', note)
  }

  return (
    <li>
      <div className="flex items-center gap-1">
        <span className="min-w-0 flex-1 truncate text-[11px] text-ink">
          {person.name}
          {settled.length > 0 && (
            <span
              className="ml-1 text-faint"
              title="Set by hand. The model is told these are not its to change"
            >
              ●
            </span>
          )}
        </span>
        <Select
          value={person.gender}
          onChange={(event) => onCorrect(person.name, 'gender', event.target.value)}
          title={
            settled.includes('gender')
              ? 'Set by hand. Every page from here is translated knowing this'
              : 'What the chapter has shown so far. Set it and it is settled'
          }
          className={`shrink-0 px-1 py-0 text-[11px] ${
            settled.includes('gender') ? 'text-ink' : 'text-faint'
          }`}
        >
          <option value="unknown">not shown</option>
          <option value="female">she</option>
          <option value="male">he</option>
        </Select>
      </div>

      {noting ? (
        <TextInput
          autoFocus
          value={note}
          onChange={(event) => setNote(event.target.value)}
          onBlur={settle}
          onKeyDown={(event) => {
            if (event.key === 'Enter') settle()
            if (event.key === 'Escape') {
              setNote(person.note ?? '')
              setNoting(false)
            }
          }}
          placeholder="who they are"
          className="mt-0.5 w-full px-1 py-0 text-[11px]"
        />
      ) : (
        <button
          type="button"
          // Taken from the person rather than from what was typed last time: a
          // page since may have said something, and this row is not the record.
          onClick={() => {
            setNote(person.note ?? '')
            setNoting(true)
          }}
          title={
            settled.includes('note')
              ? 'Set by hand. The model is told it is not to change it'
              : 'What the chapter has made of them. Click to say it yourself'
          }
          className={`block w-full truncate px-1 text-left text-[11px] leading-snug ${
            settled.includes('note') ? 'text-muted' : 'text-faint'
          } hover:text-ink`}
        >
          {person.note || 'who they are…'}
        </button>
      )}
    </li>
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
