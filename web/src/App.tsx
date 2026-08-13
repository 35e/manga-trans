import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Board } from './components/Board'
import { RegionsPanel } from './components/RegionsPanel'
import { Settings } from './components/Settings'
import { Sidebar } from './components/Sidebar'
import { TranslationsPanel } from './components/TranslationsPanel'
import { GearIcon } from './components/icons'
import { IconButton } from './components/ui'
import { useBatch } from './hooks/useBatch'
import { useFileDrop } from './hooks/useFileDrop'
import { useImageLibrary } from './hooks/useImageLibrary'
import { useLanguage } from './hooks/useLanguage'
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
  said,
  translate,
} from './lib/api'
import { archiveName, finished } from './lib/chapter'
import { compose, save } from './lib/compose'
import { SIZE_MAX, SIZE_MIN, ready } from './lib/fit'
import type { GalleryFolder, GalleryImage } from './lib/images'
import { stem } from './lib/images'
import type { Lines } from './lib/lettering'
import * as lines from './lib/lettering'
import { mark } from './lib/mask'
import { halves, insertionFor, movedIndex } from './lib/order'
import * as blocks from './lib/regions'
import type { Packed } from './lib/zip'
import { pack } from './lib/zip'

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
  const { images, folders, add, remove, dropFolder, clear, busy, notice, dismissNotice } =
    useImageLibrary()
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
  const source = useLanguage()
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
  /** How far through packing a folder into an archive, or null when not. */
  const [packing, setPacking] = useState<{ done: number; total: number } | null>(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [showCleaned, setShowCleaned] = useState(false)

  // How far past the ink a tracing reaches. What is enough depends on the scan,
  // so it is the reader's to raise when edges are being left behind.
  const [spread, setSpread] = useState(4)
  const [fill, setFill] = useState<Fill>('art')

  const analysis = active ? (analyses[active.id] ?? null) : null
  const pageLettering = active ? (lettering[active.id] ?? []) : []
  // Which pages have been lettered, so a folder run can ask before doing them
  // over: a line moved or rewritten by hand cannot be got back.
  const lettered = Object.entries(lettering)
    .filter(([, set]) => set.some(Boolean))
    .map(([id]) => id)
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

  // Which page is on the board, for the steps that run against a page that may
  // not be it: a folder run must not clear the selection or move the step tabs
  // out from under whoever is looking at another page while it works.
  const activeNow = useRef(activeId)
  activeNow.current = activeId
  const onBoard = useCallback((id: string) => id === activeNow.current, [])

  // And whether a page is still here at all: a run works through the list it
  // picked up, and a page can be deleted out from under it.
  const imagesNow = useRef(images)
  imagesNow.current = images
  const held = useCallback((id: string) => imagesNow.current.some((it) => it.id === id), [])

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

  /** Everything held about one page that is not the page itself. */
  const forget = useCallback(
    (id: string) => {
      dropMask(id)
      dropCleaned(id)
      traced.drop(id)
      setLettering((current) => without(current, id))
      setAnalyses((current) => without(current, id))
    },
    [dropMask, dropCleaned, traced],
  )

  const removeImage = useCallback(
    (id: string) => {
      remove(id)
      forget(id)
    },
    [remove, forget],
  )

  /**
   * Why a step gave up, for a caller that has to say so somewhere other than the
   * banner — a folder run names the page it happened to. Set by {@link during},
   * and read straight after the step it belongs to.
   */
  const lastFailure = useRef<string | null>(null)

  /** Run one step against one page, keeping the banner and the error in step. */
  const during = useCallback(
    async <T,>(id: string, stage: Stage, step: () => Promise<T>): Promise<T | null> => {
      setError(null)
      setWorking({ id, stage })
      try {
        return await step()
      } catch (cause) {
        const why = said(cause)
        lastFailure.current = why
        setError(why)
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
          read(page.file, boxes, source.code),
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
    [during, source.code],
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
      // These blocks are about to be replaced, so a place in the old list means
      // nothing. Only for the page being looked at: a folder run works its way
      // through the others without touching what is picked out on this one.
      if (onBoard(id)) setSelected(null)

      return during(id, 'detecting', async () => {
        const detection = await detect(file, source.code)
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
            source.code,
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
    [during, onBoard, source.code],
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
   * Mark these boxes into the page's mask by the lettering inside them, tracing
   * the page first if that has not been asked for yet.
   *
   * Tracing rather than taking whatever is already held is the whole of it:
   * without it there is nothing to mark but the box, and the clean takes out the
   * whole rectangle instead of the words in it. Which is what a block put back
   * by hand, or one drawn where the detector missed one, used to come out as.
   *
   * Only where something is marked already — an untouched mask is seeded in one
   * go when the clean step is opened, and clearing it means it.
   */
  const markLetters = useCallback(
    async (page: GalleryImage, boxes: Box[]) => {
      const mask = forPage(page)
      if (!mask || mask.empty) return
      const letters = await tracePage(page)
      // It may have been cleared while the tracing was in the air.
      if (!mask.empty) mark(mask, boxes, letters)
    },
    [forPage, tracePage],
  )

  /**
   * Take one block out of what will be cleaned, or put it back. The mask is kept
   * in step, so what is marked always matches what the list says.
   */
  const toggleExcluded = useCallback(
    async (index: number) => {
      if (!active) return
      const held = analyses[active.id]
      const box = held?.detection.regions[index]?.box
      if (!held || !box) return

      const putBack = held.excluded.includes(index)
      setAnalyses((current) => ({
        ...current,
        [active.id]: blocks.toggledExcluded(held, index),
      }))

      // Put back, it is marked by its lettering as the rest of the page was;
      // taken out, the whole box is erased — right either way round, since all
      // that was ever marked inside it is that lettering.
      if (putBack) await markLetters(active, [box])
      else forPage(active)?.boxes([box], true)
    },
    [active, analyses, forPage, markLetters],
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
        source.rtl,
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
      await markLetters(active, [box])
      await reread(active, [box], [added.id])
    },
    [active, analyses, markLetters, reread, source.rtl],
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

      forPage(active)?.boxes([was], true)
      if (!held.excluded.includes(index)) await markLetters(active, [region.box])

      // Nothing has been read on this page yet, so there is nothing to bring up
      // to date: detecting will read the lot.
      if (!held.texts) return
      await reread(active, [region.box], [region.id])
    },
    [active, analyses, forPage, markLetters, reread],
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

      const [firstBox, secondBox] = halves(
        region.box,
        at / line.text.length,
        source.rtl,
      )
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
    [active, analyses, lettering, reread, source.rtl],
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
            source.language?.name,
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
    [during, ollama.model, ollama.target, prompt, source.language?.name],
  )

  // The three steps as the buttons on the board call them. Only cleaning moves on
  // by itself: detection is imperfect and its boxes are there to be looked at and
  // corrected, so it stays on the step that shows them.
  const runDetect = useCallback(async () => {
    if (active) await detectAndRead(active)
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

  /**
   * The usual way through one page, in one go: find the words, hide them, letter
   * it. Each step feeds the next.
   *
   * It takes the page rather than reading the active one because a folder run puts
   * every page in a folder through this same way, and hands back why it gave up
   * rather than only leaving it in the banner, because a run has to say which page
   * that was.
   */
  const pipeline = useCallback(
    async (page: GalleryImage): Promise<string | null> => {
      // Deleted since the run picked up its list. Nothing to do and nothing wrong.
      if (!held(page.id)) return null

      const found = await detectAndRead(page)
      if (!found) return lastFailure.current ?? 'the text could not be found'

      const marks = await marksFor(page, found)
      if (marks) {
        lastFailure.current = null
        if (!(await cleanPage(page, marks))) {
          return lastFailure.current ?? 'the page could not be cleaned'
        }
      }

      // The tracing is worked out from the page and can be had again; a folder of
      // fifty would otherwise hold fifty page-sized bitmaps for the sake of one.
      // The page on the board keeps its own, since that is the one being brushed.
      if (!onBoard(page.id)) traced.drop(page.id)
      else setMode('translate')

      if (ollama.model && found.texts?.some((text) => text.trim())) {
        lastFailure.current = null
        await translatePage(page, found)
        // Not every empty answer is a refusal: a page whose every block was left
        // alone has nothing to translate and nothing went wrong with it.
        if (lastFailure.current) return lastFailure.current
      }
      return null
    },
    [
      detectAndRead,
      marksFor,
      cleanPage,
      translatePage,
      traced,
      onBoard,
      held,
      ollama.model,
    ],
  )

  /** "Do all three": that, for the page on the board. */
  const runAll = useCallback(() => {
    if (active) void pipeline(active)
  }, [active, pipeline])

  const {
    run: batch,
    start: startBatch,
    stop: stopBatch,
    dismiss: dismissBatch,
  } = useBatch(pipeline)

  /** Every page in a folder, in the order the archive put them. */
  const runFolder = useCallback(
    (folder: GalleryFolder) => {
      void startBatch(
        folder,
        images.filter((image) => image.folder === folder.id),
      )
    },
    [startBatch, images],
  )

  const removeFolder = useCallback(
    (id: string) => {
      if (batch?.folder === id) stopBatch()
      for (const image of images) if (image.folder === id) forget(image.id)
      dropFolder(id)
    },
    [batch?.folder, stopBatch, images, forget, dropFolder],
  )

  const clearAll = useCallback(() => {
    stopBatch()
    clear()
    clearMasks()
    clearCleaned()
    traced.clear()
    setLettering({})
    setAnalyses({})
    setActiveId(null)
  }, [stopBatch, clear, clearMasks, clearCleaned, traced])

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

  /**
   * A whole folder back out as the archive it came in as.
   *
   * One page at a time rather than all at once: each one is drawn at the page's
   * own resolution, and forty of those in the air together is forty page-sized
   * canvases held for no gain — the work is the same either way, and this way
   * there is something true to count.
   *
   * A page that will not compose is put in as it arrived rather than left out.
   * A chapter with a gap in it is not a chapter: the page that fell over is
   * still part of the story, and dropping it renumbers everything after it.
   */
  const downloadFolder = useCallback(
    async (folder: GalleryFolder) => {
      const pages = imagesNow.current.filter((image) => image.folder === folder.id)
      if (pages.length === 0 || packing) return

      setPacking({ done: 0, total: pages.length })
      setError(null)
      try {
        const held: Packed[] = []
        let anyLettered = false

        for (const [at, page] of pages.entries()) {
          try {
            const made = await finished(
              page,
              lettering[page.id],
              cleanedNow.current[page.id] ?? null,
            )
            anyLettered ||= made.reached === 'lettered'
            held.push(made)
          } catch {
            held.push({
              name: page.name,
              bytes: new Uint8Array(await page.file.arrayBuffer()),
            })
          }
          setPacking({ done: at + 1, total: pages.length })
        }

        save(await pack(held), archiveName(folder, ollama.target, anyLettered))
      } catch (cause) {
        setError(said(cause))
      } finally {
        setPacking(null)
      }
    },
    [lettering, ollama.target, packing],
  )

  /** Pages worth putting in an archive: anything that was cleaned or lettered. */
  const workedOn = useMemo(
    () =>
      Object.keys(cleanedPages).concat(
        Object.entries(lettering)
          .filter(([, set]) => set.some((line) => line !== null && line.text.trim()))
          .map(([id]) => id),
      ),
    [cleanedPages, lettering],
  )

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
          folders={folders}
          activeId={active?.id ?? null}
          onOpen={setActiveId}
          onRemove={removeImage}
          onRemoveFolder={removeFolder}
          onFiles={add}
          dragging={dragging}
          busy={busy}
          notice={notice}
          onDismissNotice={dismissNotice}
          onClearAll={clearAll}
          batch={batch}
          // The bar names the page it is working on, so it wants that page's
          // stage rather than the board's — which is only ever the active page's.
          batchStage={working && working.id === batch?.page?.id ? working.stage : null}
          onRunFolder={runFolder}
          onStopBatch={stopBatch}
          onDismissBatch={dismissBatch}
          canTranslate={Boolean(ollama.model)}
          lettered={lettered}
          onDownloadFolder={(folder) => void downloadFolder(folder)}
          workedOn={workedOn}
          packing={packing}
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
          runningFolder={batch !== null && !batch.finished}
          showCleaned={showCleaned}
          onShowCleaned={setShowCleaned}
          onRunAll={runAll}
          onDetect={runDetect}
          inspecting={{
            languages: source.offered,
            language: source.code,
            onLanguage: source.setCode,
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
