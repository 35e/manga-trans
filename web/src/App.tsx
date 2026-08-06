import { useCallback, useEffect, useState } from 'react'
import { Board } from './components/Board'
import { Dropzone } from './components/Dropzone'
import { Gallery } from './components/Gallery'
import { RegionsPanel } from './components/RegionsPanel'
import { useFileDrop } from './hooks/useFileDrop'
import { useImageLibrary } from './hooks/useImageLibrary'
import type { Analysis, Stage } from './lib/api'
import { API_BASE, detect, read } from './lib/api'
import { formatBytes, plural } from './lib/images'

function App() {
  const { images, add, remove, clear, busy, notice, dismissNotice } =
    useImageLibrary()
  const dragging = useFileDrop(add)

  const [activeId, setActiveId] = useState<string | null>(null)
  const active = images.find((image) => image.id === activeId) ?? null

  // Kept per page, so going back to one already done shows its blocks again
  // without asking the API twice.
  const [analyses, setAnalyses] = useState<Record<string, Analysis>>({})
  const [working, setWorking] = useState<{ id: string; stage: Stage } | null>(
    null,
  )
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<number | null>(null)

  const analysis = active ? (analyses[active.id] ?? null) : null
  const stage = working?.id === active?.id ? (working?.stage ?? null) : null

  // The first page dropped in goes straight to the board, and so does whatever
  // is left when the one on it is deleted.
  useEffect(() => {
    if (active === null && images.length > 0) setActiveId(images[0].id)
  }, [active, images])

  useEffect(() => {
    setSelected(null)
    setError(null)
  }, [activeId])

  const removeImage = useCallback(
    (id: string) => {
      remove(id)
      setAnalyses((current) =>
        Object.fromEntries(
          Object.entries(current).filter(([key]) => key !== id),
        ),
      )
    },
    [remove],
  )

  const clearAll = useCallback(() => {
    clear()
    setAnalyses({})
    setActiveId(null)
  }, [clear])

  /** Find the lettering, then read it. The boxes show as soon as they land. */
  const runDetect = useCallback(async () => {
    if (!active) return
    const { id, file } = active

    setError(null)
    setSelected(null)
    setWorking({ id, stage: 'detecting' })
    try {
      const detection = await detect(file)
      setAnalyses((current) => ({ ...current, [id]: { detection, texts: null } }))
      if (detection.regions.length === 0) return

      setWorking({ id, stage: 'reading' })
      const texts = await read(
        file,
        detection.regions.map((region) => region.box),
      )
      // Only if the page is still in the library: it may have been deleted
      // while the reader was working.
      setAnalyses((current) =>
        id in current ? { ...current, [id]: { detection, texts } } : current,
      )
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setWorking(null)
    }
  }, [active])

  const total = images.reduce((sum, image) => sum + image.size, 0)

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-slate-50 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
      <header className="flex shrink-0 items-center justify-between gap-4 border-b border-slate-200 px-4 py-2.5 dark:border-white/10">
        <h1 className="text-sm font-semibold tracking-tight">manga-trans</h1>
        <p className="truncate text-xs text-slate-400 dark:text-slate-500">
          {API_BASE}
        </p>
      </header>

      <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
        <aside className="flex shrink-0 flex-col border-slate-200 bg-white max-lg:h-64 max-lg:border-b lg:w-60 lg:border-r xl:w-72 dark:border-white/10 dark:bg-slate-950">
          <div className="shrink-0 p-3">
            <Dropzone onFiles={add} dragging={dragging} busy={busy} />
          </div>

          {notice && (
            <div className="mx-3 mb-3 flex shrink-0 items-start justify-between gap-2 rounded-lg bg-amber-50 px-2.5 py-2 text-[11px] leading-snug text-amber-900 dark:bg-amber-500/10 dark:text-amber-200">
              <span>{notice.text}</span>
              <button
                type="button"
                onClick={dismissNotice}
                aria-label="Dismiss"
                className="shrink-0 font-semibold hover:underline"
              >
                ✕
              </button>
            </div>
          )}

          <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-3">
            <Gallery
              images={images}
              activeId={active?.id ?? null}
              onOpen={setActiveId}
              onRemove={removeImage}
            />
          </div>

          {images.length > 0 && (
            <div className="flex shrink-0 items-center justify-between gap-2 border-t border-slate-200 px-3 py-2 text-[11px] text-slate-500 dark:border-white/10 dark:text-slate-400">
              <span className="tabular-nums">
                {plural(images.length, 'page')} · {formatBytes(total)}
              </span>
              <ClearAll onClear={clearAll} />
            </div>
          )}
        </aside>

        <Board
          image={active}
          analysis={analysis}
          stage={stage}
          error={error}
          selected={selected}
          onSelect={setSelected}
          onDetect={runDetect}
        />

        {active && analysis && (
          <RegionsPanel
            analysis={analysis}
            reading={stage === 'reading'}
            selected={selected}
            onSelect={setSelected}
          />
        )}
      </div>

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
      className={`shrink-0 rounded-md px-2 py-1 font-medium transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-red-500 ${
        armed
          ? 'bg-red-600 text-white hover:bg-red-700'
          : 'text-slate-500 hover:bg-red-50 hover:text-red-600 dark:text-slate-400 dark:hover:bg-red-500/10 dark:hover:text-red-400'
      }`}
    >
      {armed ? 'Sure?' : 'Clear all'}
    </button>
  )
}

export default App
