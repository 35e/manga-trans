import { useCallback, useEffect, useRef, useState } from 'react'
import { Board } from './components/Board'
import { RegionsPanel } from './components/RegionsPanel'
import { Settings } from './components/Settings'
import { Sidebar } from './components/Sidebar'
import { TranslationsPanel } from './components/TranslationsPanel'
import { GearIcon } from './components/icons'
import { IconButton } from './components/ui'
import { useFileDrop } from './hooks/useFileDrop'
import { useImageLibrary } from './hooks/useImageLibrary'
import { useLetterMasks } from './hooks/useLetterMasks'
import { useMasks } from './hooks/useMasks'
import { useObjectUrls } from './hooks/useObjectUrls'
import { useOllama } from './hooks/useOllama'
import { usePrompt } from './hooks/usePrompt'
import type { Analysis, BoardMode, Box, Fill, Lettering, Region, Stage } from './lib/api'
import {
  API_BASE,
  UNSURE,
  bubbles,
  clean,
  detect,
  letterMask,
  read,
  translate,
} from './lib/api'
import { compose, save } from './lib/compose'
import { SIZE_MAX, SIZE_MIN, ready } from './lib/fit'
import type { GalleryImage } from './lib/images'
import { stem } from './lib/images'
import type { Lines } from './lib/lettering'
import * as lines from './lib/lettering'
import { mark } from './lib/mask'
import { halves, insertionFor, movedIndex } from './lib/order'
import * as blocks from './lib/regions'

const said = (cause: unknown) =>
  cause instanceof Error ? cause.message : String(cause)

/** The same record without one key. */
function without<T>(record: Record<string, T>, key: string): Record<string, T> {
  return Object.fromEntries(Object.entries(record).filter(([held]) => held !== key))
}

const newRegion = (box: Box, from?: Region): Region => ({
  id: crypto.randomUUID(),
  box,
  confidence: from?.confidence ?? 1,
  manual: from ? from.manual : true,
})

function App() {
  const { images, add, remove, clear, busy, notice, dismissNotice } = useImageLibrary()
  const dragging = useFileDrop(add)
  const { forPage, drop: dropMask, clear: clearMasks } = useMasks()
  const traced = useLetterMasks()
  const {
    urls: cleanedPages,
    set: setCleaned,
    drop: dropCleaned,
    clear: clearCleaned,
  } = useObjectUrls()
  const ollama = useOllama()
  const { prompt, setPrompt, builtIn: builtInPrompt } = usePrompt()

  const [activeId, setActiveId] = useState<string | null>(null)
  const active = images.find((image) => image.id === activeId) ?? null

  // Both kept per page, so going back to one already done shows its work again
  // without asking the API twice.
  const [analyses, setAnalyses] = useState<Record<string, Analysis>>({})
  const [lettering, setLettering] = useState<Record<string, Lines>>({})

  const [working, setWorking] = useState<{ id: string; stage: Stage } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<number | null>(null)
  const [mode, setMode] = useState<BoardMode>('inspect')
  const [applying, setApplying] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [showCleaned, setShowCleaned] = useState(false)

  // How far past the ink a tracing reaches. What is enough depends on the scan,
  // so it is the reader's to raise when edges are being left behind.
  const [spread, setSpread] = useState(4)
  const [fill, setFill] = useState<Fill>('art')

  const analysis = active ? (analyses[active.id] ?? null) : null
  const pageLettering = active ? (lettering[active.id] ?? []) : []
  const cleanedPage = active ? (cleanedPages[active.id] ?? null) : null
  const stage = working?.id === active?.id ? (working?.stage ?? null) : null

  // Fetched early: a font is only asked for when something needs it, and the
  // first thing that needs this one is measuring where the lettering breaks.
  useEffect(() => {
    void ready()
  }, [])

  // The first page dropped in goes straight to the board, and so does whatever
  // is left when the one on it is deleted.
  useEffect(() => {
    if (active === null && images.length > 0) setActiveId(images[0].id)
  }, [active, images])

  useEffect(() => {
    setSelected(null)
    setError(null)
  }, [activeId])

  // A page opens on the first thing left to do with it, so picking one up again
  // lands where it was left rather than back at the beginning. Read through refs
  // so this runs when the page changes and not when its work does.
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

  // A page arrives showing what came in; once cleaned, and whenever lettering is
  // being set over it, the board shows the cleaned one. In this order so the
  // later word wins.
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
      traced.drop(id)
      setLettering((current) => without(current, id))
      setAnalyses((current) => without(current, id))
    },
    [remove, dropMask, dropCleaned, traced],
  )

  const clearAll = useCallback(() => {
    clear()
    clearMasks()
    clearCleaned()
    traced.clear()
    setLettering({})
    setAnalyses({})
    setActiveId(null)
  }, [clear, clearMasks, clearCleaned, traced])

  /** Run one step against one page, keeping the banner and the error in step. */
  const during = useCallback(
    async <T,>(id: string, stage: Stage, step: () => Promise<T>): Promise<T | null> => {
      setError(null)
      setWorking({ id, stage })
      try {
        return await step()
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
   * What the blocks that just changed say, and what room every block on the
   * page is now in.
   *
   * Reading is asked only about the blocks that changed, since it is the slow
   * half. The balloons are asked about all of them, because where several blocks
   * share one balloon it is shared out between them — so adding or moving one
   * changes the answer for its neighbours too, and asking about it alone would
   * hand it the whole balloon and letter it over the top of them.
   */
  const reread = useCallback(
    async (page: GalleryImage, boxes: Box[], ids: string[]) => {
      const every = analysesNow.current[page.id]?.detection.regions ?? []
      const everyId = every.map((region) => region.id)

      const answered = await during(page.id, 'reading', () =>
        Promise.all([
          read(page.file, boxes),
          bubbles(
            page.file,
            every.map((region) => region.box),
          ),
        ]),
      )
      if (!answered) return
      const [texts, balloons] = answered

      setAnalyses((current) => {
        const now = current[page.id]
        if (!now) return current
        const said = ids.reduce(
          (held, id, at) => blocks.withReading(held, id, texts[at]),
          now,
        )
        const next = blocks.withRooms(said, everyId, balloons)
        return next === now ? current : { ...current, [page.id]: next }
      })
    },
    [during],
  )

  /**
   * Find the lettering and read it, and hand back what was found.
   *
   * Every step below returns its answer as well as leaving it in state, so one
   * can be run straight into the next — which is what "do all three" does.
   */
  const detectAndRead = useCallback(
    async (page: GalleryImage): Promise<Analysis | null> => {
      const { id, file } = page
      setSelected(null)

      return during(id, 'detecting', async () => {
        const detection = await detect(file)
        let found: Analysis = {
          detection,
          texts: detection.regions.length === 0 ? [] : null,
          // Whatever the detector is not sure of starts left alone. It is still
          // read and still listed, so what it says can be seen before deciding.
          excluded: detection.regions.flatMap((region, index) =>
            region.confidence < UNSURE ? [index] : [],
          ),
        }
        setAnalyses((current) => ({ ...current, [id]: found }))
        // These are new blocks, so what was lettered against the old ones no
        // longer means anything.
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
      })
    },
    [during],
  )

  /**
   * The lettering itself, traced pixel by pixel, so a clean can hide the words
   * and leave the art they were drawn over.
   */
  const tracePage = useCallback(
    async (page: GalleryImage): Promise<ImageBitmap | null> => {
      const held = traced.at(page.id, spread)
      if (held) return held

      return during(page.id, 'tracing', async () => {
        const bitmap = await createImageBitmap(await letterMask(page.file, spread))
        traced.keep(page.id, spread, bitmap)
        return bitmap
      })
    },
    [during, traced, spread],
  )

  const traceLetters = useCallback(
    () => (active ? tracePage(active) : Promise.resolve(null)),
    [active, tracePage],
  )

  /**
   * Take one block out of what will be cleaned, or put it back. The mask is kept
   * in step, so what is marked always matches what the list says.
   */
  const toggleExcluded = useCallback(
    (index: number) => {
      if (!active) return
      const held = analyses[active.id]
      const box = held?.detection.regions[index]?.box
      if (!held || !box) return

      const mask = forPage(active)
      if (held.excluded.includes(index)) {
        if (mask && !mask.empty) mask.boxes([box])
      } else {
        mask?.boxes([box], true)
      }

      setAnalyses((current) => ({
        ...current,
        [active.id]: blocks.toggledExcluded(held, index),
      }))
    },
    [active, analyses, forPage],
  )

  /**
   * A block the detector missed, drawn by hand. It goes in where reading order
   * puts it, not on the end, so the page still translates as one conversation.
   */
  const addRegion = useCallback(
    async (box: Box) => {
      if (!active) return
      const held = analyses[active.id]
      if (!held) return

      const at = insertionFor(
        held.detection.regions.map((region) => region.box),
        box,
      )
      const added = newRegion(box)
      setAnalyses((current) => ({
        ...current,
        [active.id]: blocks.inserted(held, at, added),
      }))
      setLettering((current) =>
        current[active.id]
          ? { ...current, [active.id]: lines.inserted(current[active.id], at) }
          : current,
      )
      setSelected(at)

      // Marked for hiding along with the rest, if the rest already are.
      const mask = forPage(active)
      if (mask && !mask.empty) mark(mask, [box], traced.at(active.id, spread))

      await reread(active, [box], [added.id])
    },
    [active, analyses, forPage, traced, spread, reread],
  )

  /**
   * A block's box while it is being dragged. Only the box moves here: what is
   * inside it is read again once the drag is over rather than every frame.
   */
  const setRegionBox = useCallback(
    (index: number, box: Box) => {
      if (!active) return
      setAnalyses((current) => {
        const held = current[active.id]
        if (!held) return current
        return { ...current, [active.id]: blocks.withBox(held, index, box) }
      })
    },
    [active],
  )

  /**
   * That drag, once it is over: read what the block says now, and keep the mask
   * in step — the rectangle it used to cover comes out, where it is now goes in.
   *
   * This is what splitting two bubbles the detector ran together comes down to:
   * pull this one off the second, and what it says is no longer both of them.
   */
  const rereadRegion = useCallback(
    async (index: number, was: Box) => {
      if (!active) return
      const held = analyses[active.id]
      const region = held?.detection.regions[index]
      if (!held || !region || region.box.join() === was.join()) return

      const mask = forPage(active)
      if (mask && !mask.empty) {
        mask.boxes([was], true)
        if (!held.excluded.includes(index)) {
          mark(mask, [region.box], traced.at(active.id, spread))
        }
      }

      // Nothing has been read on this page yet, so there is nothing to bring up
      // to date: detecting will read the lot.
      if (!held.texts) return
      await reread(active, [region.box], [region.id])
    },
    [active, analyses, forPage, traced, spread, reread],
  )

  /**
   * A block dragged to a different place in the list. The order is a guess the
   * detector makes, and an inset panel or a caption is enough to throw it — and
   * it is also the order the page is translated in.
   */
  const moveRegion = useCallback(
    (from: number, to: number) => {
      if (!active || from === to) return
      const { id } = active

      setAnalyses((current) =>
        current[id] ? { ...current, [id]: blocks.moved(current[id], from, to) } : current,
      )
      setLettering((current) =>
        current[id] ? { ...current, [id]: lines.moved(current[id], from, to) } : current,
      )
      setSelected((now) => (now === null ? now : movedIndex(now, from, to)))
    },
    [active],
  )

  /**
   * One block that turned out to be two bubbles, cut in two at `at` — a place in
   * the translated line, which is where the join shows. The block is what is
   * really being cut: two bubbles were always two blocks.
   */
  const splitRegion = useCallback(
    async (index: number, at: number) => {
      if (!active) return
      const held = analyses[active.id]
      const region = held?.detection.regions[index]
      const line = lettering[active.id]?.[index]
      if (!held || !region || !line) return

      const before = line.text.slice(0, at).trim()
      const rest = line.text.slice(at).trim()
      // Neither half may be empty: that is a cursor at one end, not a cut.
      if (!before || !rest) return

      const [firstBox, secondBox] = halves(region.box, at / line.text.length)
      const added = newRegion(secondBox, region)

      setAnalyses((current) => {
        const now = current[active.id]
        if (now?.detection.regions[index]?.id !== region.id) return current
        return { ...current, [active.id]: blocks.split(now, index, firstBox, added) }
      })
      setLettering((current) => {
        const page = current[active.id]
        if (!page) return current
        return {
          ...current,
          [active.id]: lines.split(
            page,
            index,
            [
              { text: before, box: firstBox },
              { text: rest, box: secondBox },
            ],
            held.texts?.[index] ?? '',
            region.box,
          ),
        }
      })
      setSelected(index)

      await reread(active, [firstBox, secondBox], [region.id, added.id])
    },
    [active, analyses, lettering, reread],
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
        const boxes = blocks.toClean(found)
        if (boxes.length === 0) return null
        mark(mask, boxes, await tracePage(page))
      }
      return mask.toBlob()
    },
    [forPage, tracePage],
  )

  /** Hide everything the mask marks, and keep the page that comes back. */
  const cleanPage = useCallback(
    async (page: GalleryImage, marks: Blob): Promise<boolean> => {
      const cleaned = await during(page.id, 'cleaning', () =>
        clean(page.file, marks, fill),
      )
      if (!cleaned) return false
      setCleaned(page.id, cleaned)
      return true
    },
    [during, setCleaned, fill],
  )

  /**
   * Translate the page: every block that was read and not left alone, sent
   * together. Each line lands in the balloon its original was written in, not in
   * the box the original came out of, which for vertical Japanese is a column
   * too narrow to set a word of English across.
   */
  const translatePage = useCallback(
    async (page: GalleryImage, found: Analysis): Promise<boolean> => {
      if (!ollama.model || !found.texts) return false

      const skip = new Set(found.excluded)
      const wanted = found.texts
        .map((text, index) => ({ text, index }))
        .filter(({ text, index }) => text.trim() && !skip.has(index))
      if (wanted.length === 0) return false

      const got = await during(page.id, 'translating', async () => {
        const [answers] = await Promise.all([
          translate(
            wanted.map((line) => line.text),
            ollama.model,
            ollama.target,
            prompt,
          ),
          // Sizes are about to be worked out by measuring: the face has to be in.
          ready(),
        ])
        return answers
      })
      if (!got) return false

      const set: Lines = found.detection.regions.map(() => null)
      wanted.forEach((line, at) => {
        const text = (got[at] ?? '').trim()
        if (text) set[line.index] = lines.laidOut(found, line.index, text)
      })
      setLettering((current) => ({ ...current, [page.id]: set }))
      return true
    },
    [during, ollama.model, ollama.target, prompt],
  )

  // The three steps as the buttons on the board call them, each moving on to the
  // next once it has something to move on with.
  const runDetect = useCallback(async () => {
    if (active && (await detectAndRead(active))) setMode('mask')
  }, [active, detectAndRead])

  const runClean = useCallback(
    async (marks: Blob) => {
      if (active && (await cleanPage(active, marks))) setMode('translate')
    },
    [active, cleanPage],
  )

  const runTranslate = useCallback(async () => {
    const found = active ? analyses[active.id] : null
    if (active && found) await translatePage(active, found)
  }, [active, analyses, translatePage])

  /** The usual way through, in one go. Each step feeds the next. */
  const runAll = useCallback(async () => {
    if (!active) return
    const page = active

    const found = await detectAndRead(page)
    if (!found) return

    setMode('mask')
    const marks = await marksFor(page, found)
    if (marks && !(await cleanPage(page, marks))) return

    setMode('translate')
    if (ollama.model && found.texts?.some((text) => text.trim())) {
      await translatePage(page, found)
    }
  }, [active, detectAndRead, marksFor, cleanPage, translatePage, ollama.model])

  const changeLettering = useCallback(
    (index: number, patch: Partial<Lettering>) => {
      if (!active) return
      const { id } = active
      setLettering((current) =>
        current[id]
          ? { ...current, [id]: lines.withLine(current[id], index, patch) }
          : current,
      )
    },
    [active],
  )

  const setLetteringBox = useCallback(
    (index: number, box: Box) => changeLettering(index, { box }),
    [changeLettering],
  )

  /** How far the line is turned, kept to one full turn so it reads plainly. */
  const setLetteringAngle = useCallback(
    (index: number, angle: number) =>
      changeLettering(index, { angle: ((angle % 360) + 360) % 360 }),
    [changeLettering],
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

  const fitOne = useCallback(
    (index: number) => {
      const held = active ? analyses[active.id] : null
      const line = active ? lettering[active.id]?.[index] : null
      const block = held?.detection.regions[index]
      if (!held || !line || !block) return
      changeLettering(index, {
        size: lines.sizeFor(line.text, line.box, held.texts?.[index] ?? '', block.box),
      })
    },
    [active, analyses, lettering, changeLettering],
  )

  const fitAll = useCallback(() => {
    if (!active) return
    const { id } = active
    const held = analyses[id]
    if (!held) return
    setLettering((current) =>
      current[id] ? { ...current, [id]: lines.refitted(current[id], held) } : current,
    )
  }, [active, analyses])

  /**
   * Put every line back in the balloon its block was written in, and resize it
   * to suit.
   *
   * Detecting already answers with those balloons, so this is for the blocks it
   * never saw: one drawn by hand, one cut in two, one pulled off its neighbour.
   * It is also the way back after a box has been dragged about by hand.
   */
  const fitBoxes = useCallback(async () => {
    if (!active) return
    const { id, file } = active
    const found = analyses[id]
    if (!found) return

    const balloons = await during(id, 'fitting', async () => {
      const [answers] = await Promise.all([
        bubbles(
          file,
          found.detection.regions.map((region) => region.box),
        ),
        ready(),
      ])
      return answers
    })
    if (!balloons) return

    setAnalyses((current) => {
      const now = current[id]
      const next = now && blocks.withBubbles(now, balloons)
      return next ? { ...current, [id]: next } : current
    })
    setLettering((current) => {
      const page = current[id]
      const next = page && lines.intoBubbles(page, found, balloons)
      return next ? { ...current, [id]: next } : current
    })
  }, [active, analyses, during])

  /** Set the lettering into the page and hand it over as a PNG. */
  const applyToImage = useCallback(async () => {
    if (!active) return
    const set = lettering[active.id]
    if (!set?.some(Boolean)) return

    setApplying(true)
    setError(null)
    try {
      // Exactly what the board is showing under the lettering, so what comes out
      // is what was arranged.
      const base = showCleaned && cleanedPage ? cleanedPage : active.url
      const page = await compose(base, active.width, active.height, set)
      save(page, `${stem(active.name)}-lettered.png`)
    } catch (cause) {
      setError(said(cause))
    } finally {
      setApplying(false)
    }
  }, [active, lettering, cleanedPage, showCleaned])

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-canvas text-ink">
      <header className="flex shrink-0 items-center justify-between gap-4 border-b border-line bg-surface px-4 py-2.5">
        <h1 className="text-sm font-semibold tracking-tight text-ink">manga-trans</h1>
        <div className="flex min-w-0 items-center gap-2">
          <p className="truncate font-mono text-[11px] text-faint">{API_BASE}</p>
          <IconButton label="Settings" onClick={() => setSettingsOpen(true)}>
            <GearIcon />
          </IconButton>
        </div>
      </header>

      {settingsOpen && (
        <Settings
          onClose={() => setSettingsOpen(false)}
          prompt={prompt}
          fallback={builtInPrompt}
          onSave={setPrompt}
          apiBase={API_BASE}
          models={ollama.models}
        />
      )}

      <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
        <Sidebar
          images={images}
          activeId={active?.id ?? null}
          onOpen={setActiveId}
          onRemove={removeImage}
          onFiles={add}
          dragging={dragging}
          busy={busy}
          notice={notice}
          onDismissNotice={dismissNotice}
          onClearAll={clearAll}
        />

        <Board
          image={active}
          analysis={analysis}
          mask={forPage(active)}
          cleaned={cleanedPage}
          stage={stage}
          error={error}
          selected={selected}
          onSelect={setSelected}
          mode={mode}
          onMode={setMode}
          showCleaned={showCleaned}
          onShowCleaned={setShowCleaned}
          onRunAll={runAll}
          onDetect={runDetect}
          inspecting={{
            onAddRegion: addRegion,
            onRegionBox: setRegionBox,
            onRegionSettled: rereadRegion,
            onToggleExcluded: toggleExcluded,
          }}
          masking={{
            onClean: runClean,
            letters: traced.at(active?.id, spread),
            onTrace: traceLetters,
            spread,
            onSpread: setSpread,
            fill,
            onFill: setFill,
          }}
          translating={{
            models: ollama.models,
            model: ollama.model,
            onModel: ollama.setModel,
            target: ollama.target,
            onTarget: ollama.setTarget,
            onTranslate: runTranslate,
            onFitAll: fitAll,
            onFitBoxes: fitBoxes,
            lettering: pageLettering,
            onBox: setLetteringBox,
            onTurn: setLetteringAngle,
            onSize: nudgeSize,
            onApply: applyToImage,
            applying,
            note:
              ollama.problem ??
              (!analysis?.texts
                ? 'find the text first: there is nothing to translate yet'
                : pageLettering.some(Boolean) && !cleanedPage
                  ? 'this page has not been cleaned yet'
                  : null),
          }}
        />

        {active &&
          analysis &&
          (mode === 'translate' ? (
            <TranslationsPanel
              originals={analysis.texts ?? analysis.detection.regions.map(() => null)}
              lettering={pageLettering}
              selected={selected}
              onSelect={setSelected}
              onChange={changeLettering}
              onFit={fitOne}
              onSplit={splitRegion}
            />
          ) : (
            <RegionsPanel
              analysis={analysis}
              reading={stage === 'reading'}
              selected={selected}
              onSelect={setSelected}
              onToggleExcluded={toggleExcluded}
              onMove={moveRegion}
            />
          ))}
      </div>

      {dragging && (
        <div className="pointer-events-none fixed inset-0 z-40 flex items-center justify-center bg-accent/10 p-8 backdrop-blur-[2px]">
          <div className="rounded-2xl border-2 border-dashed border-accent bg-surface/90 px-8 py-6 text-center shadow-xl">
            <p className="text-base font-semibold text-ink">Drop anywhere</p>
            <p className="mt-1 text-sm text-faint">Images are added to the gallery</p>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
