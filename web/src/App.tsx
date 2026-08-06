import { useCallback, useEffect, useState } from 'react'
import { Dropzone } from './components/Dropzone'
import { Gallery } from './components/Gallery'
import { Lightbox } from './components/Lightbox'
import { useFileDrop } from './hooks/useFileDrop'
import { useImageLibrary } from './hooks/useImageLibrary'
import { formatBytes, plural } from './lib/images'

function App() {
  const { images, add, remove, clear, busy, notice, dismissNotice } =
    useImageLibrary()
  const dragging = useFileDrop(add)

  const [openId, setOpenId] = useState<string | null>(null)
  const openIndex = images.findIndex((image) => image.id === openId)
  const open = openIndex === -1 ? null : images[openIndex]

  // An image removed from under the viewer (or a cleared library) closes it.
  useEffect(() => {
    if (openId !== null && openIndex === -1) setOpenId(null)
  }, [openId, openIndex])

  const step = useCallback(
    (delta: number) => {
      setOpenId((current) => {
        const at = images.findIndex((image) => image.id === current)
        if (at === -1) return current
        const next = (at + delta + images.length) % images.length
        return images[next].id
      })
    },
    [images],
  )

  const closeViewer = useCallback(() => setOpenId(null), [])

  const removeOpen = useCallback(() => {
    if (!open) return
    // Move to whatever takes its place before it goes.
    const next = images[openIndex + 1] ?? images[openIndex - 1] ?? null
    setOpenId(next?.id ?? null)
    remove(open.id)
  }, [images, open, openIndex, remove])

  const total = images.reduce((sum, image) => sum + image.size, 0)

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
      <header className="sticky top-0 z-30 border-b border-slate-200 bg-slate-50/80 backdrop-blur dark:border-white/10 dark:bg-slate-950/80">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-4">
          <div>
            <h1 className="text-lg font-semibold tracking-tight">
              manga-trans
            </h1>
            <p className="text-sm text-slate-500 dark:text-slate-400">
              {images.length === 0
                ? 'Drop the pages you want to work on'
                : `${plural(images.length, 'page')} · ${formatBytes(total)}`}
            </p>
          </div>
          {images.length > 0 && <ClearAll onClear={clear} />}
        </div>
      </header>

      <main className="mx-auto max-w-6xl space-y-6 px-4 py-8">
        <Dropzone onFiles={add} dragging={dragging} busy={busy} />

        {notice && (
          <div
            role="status"
            className="flex items-center justify-between gap-4 rounded-xl border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-200"
          >
            <span>{notice.text}</span>
            <button
              type="button"
              onClick={dismissNotice}
              className="shrink-0 rounded-md px-2 py-1 font-medium transition-colors hover:bg-amber-200/60 dark:hover:bg-amber-500/20"
            >
              Dismiss
            </button>
          </div>
        )}

        <Gallery images={images} onRemove={remove} onOpen={setOpenId} />
      </main>

      {open && (
        <Lightbox
          image={open}
          position={{ index: openIndex, total: images.length }}
          onClose={closeViewer}
          onStep={step}
          onRemove={removeOpen}
        />
      )}

      {dragging && (
        <div className="pointer-events-none fixed inset-0 z-40 flex items-center justify-center bg-indigo-500/10 p-8 backdrop-blur-[2px]">
          <div className="rounded-2xl border-2 border-dashed border-indigo-400 bg-white/90 px-8 py-6 text-center shadow-xl dark:bg-slate-900/90">
            <p className="text-base font-semibold">Drop anywhere</p>
            <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
              Images are added to the gallery
            </p>
          </div>
        </div>
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
        if (armed) {
          onClear()
          setArmed(false)
        } else {
          setArmed(true)
        }
      }}
      onBlur={() => setArmed(false)}
      className={`shrink-0 rounded-lg border px-3 py-1.5 text-sm font-medium transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-red-500 ${
        armed
          ? 'border-red-600 bg-red-600 text-white hover:bg-red-700'
          : 'border-slate-300 text-slate-600 hover:border-red-400 hover:text-red-600 dark:border-white/15 dark:text-slate-300 dark:hover:border-red-500/60 dark:hover:text-red-400'
      }`}
    >
      {armed ? 'Delete them all?' : 'Clear all'}
    </button>
  )
}

export default App
