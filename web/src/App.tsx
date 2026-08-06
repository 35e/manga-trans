import { useCallback, useEffect, useRef, useState } from 'react'
import { Board } from './components/Board'
import { Dropzone } from './components/Dropzone'
import { Gallery } from './components/Gallery'
import { RegionsPanel } from './components/RegionsPanel'
import { Settings } from './components/Settings'
import { TranslationsPanel } from './components/TranslationsPanel'
import { useFileDrop } from './hooks/useFileDrop'
import { useImageLibrary } from './hooks/useImageLibrary'
import { useMasks } from './hooks/useMasks'
import { useObjectUrls } from './hooks/useObjectUrls'
import { useSettings } from './hooks/useSettings'
import type { Analysis, BoardMode, Box, Lettering, Stage } from './lib/api'
import {
  API_BASE,
  clean,
  defaultPrompt,
  detect,
  letterMask,
  models as listModels,
  read,
  translate,
} from './lib/api'
import { compose, save } from './lib/compose'
import { SIZE_MAX, SIZE_MIN, fitSize, ready } from './lib/fit'
import type { GalleryImage } from './lib/images'
import { formatBytes, plural } from './lib/images'

const said = (cause: unknown) =>
  cause instanceof Error ? cause.message : String(cause)

/** The same record without one key. */
function without<T>(record: Record<string, T>, key: string): Record<string, T> {
  return Object.fromEntries(
    Object.entries(record).filter(([held]) => held !== key),
  )
}

function App() {
  const { images, add, remove, clear, busy, notice, dismissNotice } =
    useImageLibrary()
  const dragging = useFileDrop(add)
  const { forPage, drop: dropMask, clear: clearMasks } = useMasks()
  const {
    urls: cleanedPages,
    set: setCleaned,
    drop: dropCleaned,
    clear: clearCleaned,
  } = useObjectUrls()

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
  const [mode, setMode] = useState<BoardMode>('inspect')

  // The translated lines, one set per page, aligned with that page's blocks.
  const [lettering, setLettering] = useState<Record<string, (Lettering | null)[]>>(
    {},
  )
  const [ollamaModels, setOllamaModels] = useState<string[]>([])
  const [model, setModel] = useState('')
  const [target, setTarget] = useState('English')
  const [noModels, setNoModels] = useState<string | null>(null)
  const [applying, setApplying] = useState(false)

  const { prompt, setPrompt } = useSettings()
  const [builtInPrompt, setBuiltInPrompt] = useState<string | null>(null)
  const [settingsOpen, setSettingsOpen] = useState(false)

  // The API's own prompt, to show as the starting point and to go back to.
  useEffect(() => {
    let dropped = false
    defaultPrompt().then(
      (found) => {
        if (!dropped) setBuiltInPrompt(found)
      },
      () => undefined,
    )
    return () => {
      dropped = true
    }
  }, [])
  const [showCleaned, setShowCleaned] = useState(false)

  // Fetched early: a font is only asked for when something needs it, and the
  // first thing that needs this one is measuring where the lettering breaks.
  useEffect(() => {
    void ready()
  }, [])

  // What Ollama has to translate with, asked for once.
  useEffect(() => {
    let dropped = false
    listModels().then(
      (found) => {
        if (dropped) return
        setOllamaModels(found)
        setModel((chosen) => chosen || found[0] || '')
        setNoModels(found.length === 0 ? 'Ollama has no models pulled' : null)
      },
      (cause: unknown) => {
        if (!dropped) {
          setNoModels(cause instanceof Error ? cause.message : String(cause))
        }
      },
    )
    return () => {
      dropped = true
    }
  }, [])

  // The traced lettering, one bitmap per page. Held here rather than fetched
  // twice: tracing is another pass of the detector, so it is worth keeping.
  const [letters, setLetters] = useState<Record<string, ImageBitmap>>({})
  const lettersHeld = useRef(letters)
  lettersHeld.current = letters

  useEffect(() => {
    return () => {
      for (const bitmap of Object.values(lettersHeld.current)) bitmap.close()
    }
  }, [])

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

  const cleanedPage = active ? (cleanedPages[active.id] ?? null) : null

  // A page opens on the first thing left to do with it, so picking one up again
  // lands where it was left rather than back at the beginning.
  const analysesNow = useRef(analyses)
  analysesNow.current = analyses
  const cleanedNow = useRef(cleanedPages)
  cleanedNow.current = cleanedPages

  useEffect(() => {
    if (!activeId) return
    const found = analysesNow.current[activeId]
    setMode(
      !found?.texts ? 'inspect' : !cleanedNow.current[activeId] ? 'mask' : 'translate',
    )
  }, [activeId])

  // A page arrives showing what came in; once it has been cleaned, and whenever
  // lettering is being set over it, the board shows the cleaned one. Declared in
  // this order so the later word wins.
  useEffect(() => setShowCleaned(false), [activeId])
  useEffect(() => {
    if (cleanedPage) setShowCleaned(true)
  }, [cleanedPage])
  useEffect(() => {
    if (mode === 'translate' && cleanedPage) setShowCleaned(true)
  }, [mode, cleanedPage])

  const removeImage = useCallback(
    (id: string) => {
      remove(id)
      dropMask(id)
      dropCleaned(id)
      lettersHeld.current[id]?.close()
      setLetters((current) => without(current, id))
      setLettering((current) => without(current, id))
      setAnalyses((current) => without(current, id))
    },
    [remove, dropMask, dropCleaned],
  )

  const clearAll = useCallback(() => {
    clear()
    clearMasks()
    clearCleaned()
    for (const bitmap of Object.values(lettersHeld.current)) bitmap.close()
    setLetters({})
    setLettering({})
    setAnalyses({})
    setActiveId(null)
  }, [clear, clearMasks, clearCleaned])

  /**
   * Find the lettering and read it, and hand back what was found.
   *
   * Every step below takes the page it works on and returns its answer rather
   * than only leaving it in state, so that one of them can be run straight into
   * the next — which is what the whole page in one go does.
   */
  const detectAndRead = useCallback(
    async (page: GalleryImage): Promise<Analysis | null> => {
      const { id, file } = page

      setError(null)
      setSelected(null)
      setWorking({ id, stage: 'detecting' })
      try {
        const detection = await detect(file)
        let found: Analysis = {
          detection,
          texts: detection.regions.length === 0 ? [] : null,
          excluded: [],
        }
        setAnalyses((current) => ({ ...current, [id]: found }))
        // Lettering is held per block; these are new blocks, so what was set
        // against the old ones no longer means anything.
        setLettering((current) => without(current, id))

        if (detection.regions.length > 0) {
          setWorking({ id, stage: 'reading' })
          const texts = await read(
            file,
            detection.regions.map((region) => region.box),
          )
          found = { ...found, texts }
          // Only if the page is still in the library: it may have been deleted
          // while the reader was working.
          setAnalyses((current) =>
            id in current ? { ...current, [id]: found } : current,
          )
        }
        return found
      } catch (cause) {
        setError(said(cause))
        return null
      } finally {
        setWorking(null)
      }
    },
    [],
  )

  /**
   * Take one block out of what will be cleaned, or put it back. The mask is
   * kept in step: a block dropped is erased from it, a block restored is
   * stamped back in, so what is marked always matches what the list says.
   */
  const toggleExcluded = useCallback(
    (index: number) => {
      if (!active) return
      const held = analyses[active.id]
      const box = held?.detection.regions[index]?.box
      if (!held || !box) return

      const excluded = new Set(held.excluded)
      const mask = forPage(active)
      if (excluded.has(index)) {
        excluded.delete(index)
        if (mask && !mask.empty) mask.boxes([box])
      } else {
        excluded.add(index)
        mask?.boxes([box], true)
      }

      setAnalyses((current) => ({
        ...current,
        [active.id]: { ...held, excluded: [...excluded] },
      }))
    },
    [active, analyses, forPage],
  )

  /**
   * The lettering itself, traced pixel by pixel, so a clean can hide the words
   * and leave the art they were drawn over. Another pass of the detector, so it
   * is asked for once per page and then kept.
   */
  const tracePage = useCallback(
    async (page: GalleryImage): Promise<ImageBitmap | null> => {
      const held = lettersHeld.current[page.id]
      if (held) return held

      setError(null)
      setWorking({ id: page.id, stage: 'tracing' })
      try {
        const traced = await createImageBitmap(await letterMask(page.file))
        setLetters((current) => ({ ...current, [page.id]: traced }))
        return traced
      } catch (cause) {
        setError(said(cause))
        return null
      } finally {
        setWorking(null)
      }
    },
    [],
  )

  const traceLetters = useCallback(
    () => (active ? tracePage(active) : Promise.resolve(null)),
    [active, tracePage],
  )

  /**
   * What to hide: whatever has been marked by hand, or — if nothing has been —
   * the lettering itself, inside the blocks that were not left alone.
   */
  const marksFor = useCallback(
    async (page: GalleryImage, found: Analysis): Promise<Blob | null> => {
      const mask = forPage(page)
      if (!mask) return null

      if (mask.empty) {
        const skip = new Set(found.excluded)
        const boxes = found.detection.regions
          .filter((_, index) => !skip.has(index))
          .map((region) => region.box)
        if (boxes.length === 0) return null

        const traced = await tracePage(page)
        if (traced) mask.letters(traced, boxes)
        else mask.boxes(boxes)
      }
      return mask.toBlob()
    },
    [forPage, tracePage],
  )

  /** Hide everything the mask marks, and keep the page that comes back. */
  const cleanPage = useCallback(
    async (page: GalleryImage, marks: Blob): Promise<boolean> => {
      setError(null)
      setWorking({ id: page.id, stage: 'cleaning' })
      try {
        setCleaned(page.id, await clean(page.file, marks))
        return true
      } catch (cause) {
        setError(said(cause))
        return false
      } finally {
        setWorking(null)
      }
    },
    [setCleaned],
  )

  const pageLettering = active ? (lettering[active.id] ?? []) : []

  /**
   * Translate the page: every block that was read and not left alone, sent
   * together. Each line lands in the box its original came out of, set at
   * whatever size fits that box.
   */
  const translatePage = useCallback(
    async (page: GalleryImage, found: Analysis): Promise<boolean> => {
      if (!model || !found.texts) return false

      const skip = new Set(found.excluded)
      const wanted = found.texts
        .map((text, index) => ({ text, index }))
        .filter(({ text, index }) => text.trim() && !skip.has(index))
      if (wanted.length === 0) return false

      setError(null)
      setWorking({ id: page.id, stage: 'translating' })
      try {
        const [got] = await Promise.all([
          translate(
            wanted.map((line) => line.text),
            model,
            target,
            prompt,
          ),
          // Sizes are about to be worked out by measuring: the face has to be in.
          ready(),
        ])
        const set: (Lettering | null)[] = found.detection.regions.map(() => null)
        wanted.forEach((line, at) => {
          const text = (got[at] ?? '').trim()
          if (!text) return
          const box = found.detection.regions[line.index].box
          set[line.index] = {
            text,
            box,
            size: fitSize(text, box[2] - box[0], box[3] - box[1]),
          }
        })
        setLettering((current) => ({ ...current, [page.id]: set }))
        return true
      } catch (cause) {
        setError(said(cause))
        return false
      } finally {
        setWorking(null)
      }
    },
    [model, target, prompt],
  )

  // The three steps as the buttons on the board call them, each moving on to
  // the next once it has something to move on with.
  const runDetect = useCallback(async () => {
    if (!active) return
    if (await detectAndRead(active)) setMode('mask')
  }, [active, detectAndRead])

  const runClean = useCallback(
    async (marks: Blob) => {
      if (!active) return
      if (await cleanPage(active, marks)) setMode('translate')
    },
    [active, cleanPage],
  )

  const runTranslate = useCallback(async () => {
    const found = active ? analyses[active.id] : null
    if (active && found) await translatePage(active, found)
  }, [active, analyses, translatePage])

  /**
   * The usual way through, in one go: find the words, hide them, letter the
   * page. Each step feeds the next, and any of them can still be run on its own.
   */
  const runAll = useCallback(async () => {
    if (!active) return
    const page = active

    const found = await detectAndRead(page)
    if (!found) return

    setMode('mask')
    const marks = await marksFor(page, found)
    if (marks && !(await cleanPage(page, marks))) return

    setMode('translate')
    if (model && found.texts?.some((text) => text.trim())) {
      await translatePage(page, found)
    }
  }, [active, detectAndRead, marksFor, cleanPage, translatePage, model])

  const changeLettering = useCallback(
    (index: number, patch: Partial<Lettering>) => {
      if (!active) return
      const { id } = active
      setLettering((current) => {
        const page = current[id]
        const line = page?.[index]
        if (!line) return current
        const next = [...page]
        next[index] = { ...line, ...patch }
        return { ...current, [id]: next }
      })
    },
    [active],
  )

  const setLetteringBox = useCallback(
    (index: number, box: Box) => changeLettering(index, { box }),
    [changeLettering],
  )

  const fitOne = useCallback(
    (index: number) => {
      const line = active ? lettering[active.id]?.[index] : null
      if (!line) return
      const [x0, y0, x1, y1] = line.box
      changeLettering(index, { size: fitSize(line.text, x1 - x0, y1 - y0) })
    },
    [active, lettering, changeLettering],
  )

  /** Arrow keys on the board: one point at a time, five with shift. */
  const nudgeSize = useCallback(
    (index: number, by: number) => {
      const line = active ? lettering[active.id]?.[index] : null
      if (!line) return
      changeLettering(index, {
        size: Math.min(SIZE_MAX, Math.max(SIZE_MIN, Math.round(line.size) + by)),
      })
    },
    [active, lettering, changeLettering],
  )

  /** Set the lettering into the page and hand it over as a PNG. */
  const applyToImage = useCallback(async () => {
    if (!active) return
    const set = lettering[active.id]
    if (!set?.some(Boolean)) return

    setApplying(true)
    setError(null)
    try {
      // Exactly what the board is showing under the lettering, so what comes
      // out is what was arranged.
      const base = showCleaned && cleanedPage ? cleanedPage : active.url
      const page = await compose(base, active.width, active.height, set)
      save(page, `${active.name.replace(/\.[^.]+$/, '')}-lettered.png`)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setApplying(false)
    }
  }, [active, lettering, cleanedPage, showCleaned])

  const fitAll = useCallback(() => {
    if (!active) return
    const { id } = active
    setLettering((current) => {
      const page = current[id]
      if (!page) return current
      return {
        ...current,
        [id]: page.map((line) =>
          line === null
            ? null
            : {
                ...line,
                size: fitSize(
                  line.text,
                  line.box[2] - line.box[0],
                  line.box[3] - line.box[1],
                ),
              },
        ),
      }
    })
  }, [active])

  const total = images.reduce((sum, image) => sum + image.size, 0)

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-slate-50 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
      <header className="flex shrink-0 items-center justify-between gap-4 border-b border-slate-200 px-4 py-2.5 dark:border-white/10">
        <h1 className="text-sm font-semibold tracking-tight">manga-trans</h1>
        <div className="flex min-w-0 items-center gap-3">
          <p className="truncate text-xs text-slate-400 dark:text-slate-500">
            {API_BASE}
          </p>
          <button
            type="button"
            onClick={() => setSettingsOpen(true)}
            aria-label="Settings"
            title="Settings"
            className="shrink-0 rounded-lg p-1.5 text-slate-500 transition-colors hover:bg-slate-100 hover:text-slate-900 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-500 dark:text-slate-400 dark:hover:bg-white/10 dark:hover:text-white"
          >
            <svg
              aria-hidden="true"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.7"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="size-4"
            >
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.6 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.6a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
            </svg>
          </button>
        </div>
      </header>

      {settingsOpen && (
        <Settings
          onClose={() => setSettingsOpen(false)}
          prompt={prompt}
          fallback={builtInPrompt}
          onSave={setPrompt}
          apiBase={API_BASE}
          models={ollamaModels}
        />
      )}

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
          mask={forPage(active)}
          cleaned={cleanedPage}
          stage={stage}
          error={error}
          selected={selected}
          onSelect={setSelected}
          onDetect={runDetect}
          onClean={runClean}
          onRunAll={runAll}
          onToggleExcluded={toggleExcluded}
          letters={active ? (letters[active.id] ?? null) : null}
          onTrace={traceLetters}
          mode={mode}
          onMode={setMode}
          showCleaned={showCleaned}
          onShowCleaned={setShowCleaned}
          translating={{
            models: ollamaModels,
            model,
            onModel: setModel,
            target,
            onTarget: setTarget,
            onTranslate: runTranslate,
            onFitAll: fitAll,
            lettering: pageLettering,
            onBox: setLetteringBox,
            onSize: nudgeSize,
            onApply: applyToImage,
            applying,
            note:
              noModels ??
              (!analysis?.texts
                ? 'find the text first: there is nothing to translate yet'
                : pageLettering.some(Boolean) && !cleanedPage
                  ? 'this page has not been cleaned yet'
                  : null),
          }}
        />

        {active && analysis && mode !== 'translate' && (
          <RegionsPanel
            analysis={analysis}
            reading={stage === 'reading'}
            selected={selected}
            onSelect={setSelected}
            onToggleExcluded={toggleExcluded}
          />
        )}

        {active && analysis && mode === 'translate' && (
          <TranslationsPanel
            originals={analysis.texts ?? analysis.detection.regions.map(() => null)}
            lettering={pageLettering}
            selected={selected}
            onSelect={setSelected}
            onChange={changeLettering}
            onFit={fitOne}
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
