import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Board } from './components/Board'
import { RegionsPanel } from './components/RegionsPanel'
import { Settings } from './components/Settings'
import { Sidebar } from './components/Sidebar'
import { TranslationsPanel } from './components/TranslationsPanel'
import { GearIcon } from './components/icons'
import { IconButton } from './components/ui'
import { useBatch } from './hooks/useBatch'
import { useChapter } from './hooks/useChapter'
import { useFileDrop } from './hooks/useFileDrop'
import { useGlossary } from './hooks/useGlossary'
import { useImageLibrary } from './hooks/useImageLibrary'
import { useLanguage } from './hooks/useLanguage'
import { useLetterMasks } from './hooks/useLetterMasks'
import { useMasks } from './hooks/useMasks'
import { useObjectUrls } from './hooks/useObjectUrls'
import { useOllama } from './hooks/useOllama'
import { usePrompt } from './hooks/usePrompt'
import { useStory } from './hooks/useStory'
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
  survey,
  translate,
} from './lib/api'
import { SURVEY_PAGES, asStory, fits } from './lib/bible'
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
  // A block drawn from nothing is of no kind: the detector never saw it.
  kind: from?.kind,
  manual: from ? from.manual : true,
})

function App() {
  const {
    images,
    folders,
    makeFolder,
    add,
    remove,
    dropFolder,
    clear,
    busy,
    notice,
    dismissNotice,
  } = useImageLibrary()

  // Which folder the rail is looking into. Held here rather than in the Sidebar
  // because a drop lands in it, and the drop is caught at the window.
  const [openFolder, setOpenFolder] = useState<string | null>(null)
  const openNow = useRef(openFolder)
  openNow.current = openFolder

  const addTo = useCallback(
    (files: FileList | File[] | null) => void add(files, openNow.current ?? undefined),
    [add],
  )
  const dragging = useFileDrop(addTo)

  /** A folder of one's own, opened as it is made so the next drop lands in it. */
  const newFolder = useCallback(
    (name: string) => {
      const made = makeFolder(name)
      if (made) setOpenFolder(made)
      return made !== null
    },
    [makeFolder],
  )

  // A folder deleted while it was open leaves nothing to be inside of.
  useEffect(() => {
    if (openFolder !== null && !folders.some((held) => held.id === openFolder)) {
      setOpenFolder(null)
    }
  }, [openFolder, folders])
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
  const {
    terms: chapterTerms,
    now: termsNow,
    learn: learnTerms,
    correct: correctTerm,
    forget: forgetTerms,
    clear: clearTerms,
  } = useGlossary()
  const {
    stories: chapterStories,
    now: storyNow,
    learn: learnStory,
    correct: correctStory,
    forget: forgetStory,
    clear: clearStories,
  } = useStory()
  const {
    bibles: chapterBibles,
    now: bibleNow,
    learn: learnBible,
    correct: correctBible,
    forget: forgetBible,
    clear: clearBibles,
  } = useChapter()
  const { prompt, setPrompt, builtIn: builtInPrompt } = usePrompt()

  const [activeId, setActiveId] = useState<string | null>(null)
  const active = images.find((image) => image.id === activeId) ?? null

  // Both kept per page, so going back to one already done shows its work again.
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

  // How far past the ink a tracing reaches, in the detector's pixels.
  const [spread, setSpread] = useState(4)
  const [fill, setFill] = useState<Fill>('art')

  const analysis = active ? (analyses[active.id] ?? null) : null
  const pageLettering = active ? (lettering[active.id] ?? []) : []
  // So a folder run can ask before doing them over: a line rewritten by hand
  // cannot be got back.
  const lettered = Object.entries(lettering)
    .filter(([, set]) => set.some(Boolean))
    .map(([id]) => id)
  const cleanedPage = active ? (cleanedPages[active.id] ?? null) : null
  const stage = working?.id === active?.id ? (working?.stage ?? null) : null

  // Fetched early: the first thing that needs the face is measuring with it.
  useEffect(() => {
    void ready()
  }, [])

  useEffect(() => {
    if (active === null && images.length > 0) setActiveId(images[0].id)
  }, [active, images])

  useEffect(() => {
    setSelected(null)
    setError(null)
  }, [activeId])

  // Read through refs, so a page opening runs when the page changes and not
  // when its work does.
  const analysesNow = useRef(analyses)
  analysesNow.current = analyses
  const cleanedNow = useRef(cleanedPages)
  cleanedNow.current = cleanedPages

  // Which page is on the board: a folder run must not move the view about under
  // someone reading another page.
  const activeNow = useRef(activeId)
  activeNow.current = activeId
  const onBoard = useCallback((id: string) => id === activeNow.current, [])

  // And whether a page is still here: a run can have one deleted under it.
  const imagesNow = useRef(images)
  imagesNow.current = images
  const held = useCallback((id: string) => imagesNow.current.some((it) => it.id === id), [])

  /**
   * Which page of its chapter a page is. The bible's beats are indexed by this,
   * so it must stay the same filter in the same order a folder run uses.
   */
  const pagesOf = useCallback(
    (folder: string) => imagesNow.current.filter((it) => it.folder === folder),
    [],
  )
  const placeOf = useCallback(
    (page: GalleryImage): number | null => {
      if (!page.folder) return null
      const at = pagesOf(page.folder).findIndex((it) => it.id === page.id)
      return at === -1 ? null : at
    },
    [pagesOf],
  )
  const pagesIn = useCallback(
    (folder: string) => pagesOf(folder).length,
    [pagesOf],
  )

  useEffect(() => {
    if (!activeId) return
    const found = analysesNow.current[activeId]
    setMode(
      !found?.texts ? 'inspect' : !cleanedNow.current[activeId] ? 'mask' : 'translate',
    )
  }, [activeId])

  // In this order, so the later word wins.
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
   * banner. Set by {@link during}, read straight after the step it belongs to.
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
   * What the changed blocks say, and what room **every** block is now in.
   *
   * Reading is asked only about what changed, since it is the slow half. The
   * balloons must be asked about all of them: a box asked about alone is handed
   * the whole balloon and lettered over the top of its neighbours.
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
   * Find the lettering and read it. Every step below returns its answer as well
   * as leaving it in state, so one can be run straight into the next.
   */
  const detectAndRead = useCallback(
    async (page: GalleryImage): Promise<Analysis | null> => {
      const { id, file } = page
      // Only for the page being looked at: a run must not clear the selection
      // on a page someone else is reading.
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

  /** The lettering itself, traced pixel by pixel. */
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
   * Mark these boxes into the page's mask **by the lettering inside them**,
   * tracing the page first if that has not been asked for yet. Without the
   * tracing there is nothing to mark but the box, and the clean takes out the
   * whole rectangle. Only where something is marked already.
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

  /** Take one block out of what will be cleaned, or put it back, mask and all. */
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
   * A block the detector missed, drawn by hand. It goes in **where reading order
   * puts it**, not on the end: that order is also the order the page translates in.
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

  /** A block's box while it is being dragged; it is read again once that ends. */
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
   * in step — the old rectangle out, where it is now in.
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

  /** A block dragged to a different place in the list, which is reading order. */
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
   * together and translated against everything the chapter has settled on.
   */
  const translatePage = useCallback(
    async (page: GalleryImage, found: Analysis): Promise<boolean> => {
      if (!ollama.model || !found.texts) return false

      const skip = new Set(found.excluded)
      const wanted = found.texts
        .map((text, index) => ({ text, index }))
        .filter(({ text, index }) => text.trim() && !skip.has(index))
      if (wanted.length === 0) return false

      const chapter = page.folder
      const got = await during(page.id, 'translating', async () => {
        // The face has to be in before anything is measured, and the budgets
        // below are measured before the request rather than after it.
        await ready()
        const sending = wanted.map(({ text, index }) => {
          const region = found.detection.regions[index]
          return {
            text,
            kind: region?.kind,
            budget: region ? lines.budgetFor(region, text) : undefined,
          }
        })
        // The bible only where it still describes this folder — see `bible.fits`.
        const read = chapter ? bibleNow.current[chapter] : null
        const of = chapter ? placeOf(page) : null
        const surveyed = chapter && of !== null && fits(read, pagesIn(chapter))
        return translate(sending, ollama.model, ollama.target, {
          system: prompt,
          source: source.language?.name,
          glossary: chapter ? termsNow.current[chapter] : null,
          previously: chapter ? storyNow.current[chapter] : null,
          chapter: surveyed ? read : null,
          page: surveyed ? of : null,
        })
      })
      if (!got) return false

      if (chapter) {
        learnTerms(chapter, got.terms)
        learnStory(chapter, got.story)
      }

      const set: Lines = found.detection.regions.map(() => null)
      wanted.forEach((line, at) => {
        const text = (got.texts[at] ?? '').trim()
        if (text) set[line.index] = lines.laidOut(found, line.index, text)
      })
      setLettering((current) => ({ ...current, [page.id]: set }))
      return true
    },
    [
      during,
      ollama.model,
      ollama.target,
      prompt,
      source.language?.name,
      termsNow,
      learnTerms,
      storyNow,
      learnStory,
      bibleNow,
      placeOf,
      pagesIn,
    ],
  )

  // Only cleaning moves the tabs on by itself: detection is imperfect and its
  // boxes are there to be looked at.
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
   * Find the words on a page and read them. Takes the page rather than reading
   * the active one, and hands back why it gave up, because a run has to say
   * which page that was.
   */
  const readPage = useCallback(
    async (
      page: GalleryImage,
    ): Promise<{ found: Analysis | null; why: string | null }> => {
      // Deleted since the run picked up its list: nothing to do, nothing wrong.
      if (!held(page.id)) return { found: null, why: null }

      const found = await detectAndRead(page)
      if (!found) {
        return { found: null, why: lastFailure.current ?? 'the text could not be found' }
      }

      // The next thing to do to a page that has been read is hide it, which is
      // where the page-change rule would put it anyway. Here rather than in
      // `detectAndRead`, so the board's own Detect still leaves the tabs alone.
      if (onBoard(page.id)) setMode('mask')
      return { found, why: null }
    },
    [detectAndRead, onBoard, held],
  )

  /** And hide them, which is the slow half and so the one paid for later. */
  const hidePage = useCallback(
    async (page: GalleryImage, found: Analysis): Promise<string | null> => {
      const marks = await marksFor(page, found)
      if (marks) {
        lastFailure.current = null
        if (!(await cleanPage(page, marks))) {
          return lastFailure.current ?? 'the page could not be cleaned'
        }
      }

      // Dropped as soon as the clean lands, unless this is the page being
      // brushed: fifty pages would otherwise hold fifty page-sized bitmaps.
      if (!onBoard(page.id)) traced.drop(page.id)
      else setMode('translate')
      return null
    },
    [marksFor, cleanPage, traced, onBoard],
  )

  /** That, as a folder run wants it: why it gave up, or null when it came out. */
  const examine = useCallback(
    async (page: GalleryImage) => (await readPage(page)).why,
    [readPage],
  )

  /**
   * Hide the words and letter it, reading the page first if nobody has. The
   * analysis is taken **from the step that found it** rather than read back out
   * of state, which the step has only asked React to hold; a page read by the
   * *previous* run comes through `analysesNow`, that run being one this closure
   * never saw.
   */
  const render = useCallback(
    async (page: GalleryImage): Promise<string | null> => {
      if (!held(page.id)) return null

      let found: Analysis | null = analysesNow.current[page.id] ?? null
      if (!found?.texts) {
        const done = await readPage(page)
        if (done.why) return done.why
        found = done.found
      }
      if (!found?.texts) return lastFailure.current ?? 'the text could not be found'

      // Read by the first run and not hidden, which is the ordinary way round
      // now. Skipped where it has been: the clean is the slow half and is not
      // paid for twice.
      if (!cleanedNow.current[page.id]) {
        const why = await hidePage(page, found)
        if (why) return why
      }

      if (ollama.model && found.texts.some((text) => text.trim())) {
        lastFailure.current = null
        await translatePage(page, found)
        // Not every empty answer is a refusal, so the reason decides rather
        // than the boolean.
        if (lastFailure.current) return lastFailure.current
      }
      return null
    },
    [readPage, hidePage, translatePage, held, ollama.model],
  )

  /** "Do all three": that, for the page on the board. */
  const runAll = useCallback(() => {
    if (active) void render(active)
  }, [active, render])

  const {
    run: batch,
    start: startBatch,
    stop: stopBatch,
    dismiss: dismissBatch,
  } = useBatch()

  /**
   * The whole chapter read before a word of it is translated, a windowful at a
   * time with the running answer handed back in. A page with nothing on it is
   * still sent, as an empty list: the beats are positional.
   */
  const surveyChapter = useCallback(
    async (
      folder: GalleryFolder,
      pages: GalleryImage[],
      say: (note: string | null) => void,
    ): Promise<string | null> => {
      if (!ollama.model) return null

      const written = pages.map((page) => {
        const found = analysesNow.current[page.id]
        if (!found?.texts) return []
        const skip = new Set(found.excluded)
        return found.texts.filter((text, at) => text.trim() && !skip.has(at))
      })
      if (!written.some((page) => page.length > 0)) return null

      for (let first = 0; first < written.length; first += SURVEY_PAGES) {
        const window = written.slice(first, first + SURVEY_PAGES)
        const last = Math.min(first + window.length, written.length)
        say(`reading pages ${first + 1}–${last} of ${written.length}`)
        lastFailure.current = null
        const said = await during(pages[first].id, 'surveying', () =>
          survey(window, ollama.model as string, ollama.target, {
            source: source.language?.name,
            chapter: bibleNow.current[folder.id] ?? null,
            first,
          }),
        )
        if (!said) return lastFailure.current ?? 'the chapter could not be read'
        learnBible(folder.id, said, first)
      }

      // Seeded once and at the end, **never window by window**: the glossary
      // keeps the first rendering of a term, so folding each window in as it
      // arrived would freeze the guesses of the ones that had not read the end.
      const read = bibleNow.current[folder.id]
      if (read) {
        learnTerms(folder.id, read.terms)
        learnStory(folder.id, asStory(read))
      }
      return null
    },
    [
      during,
      ollama.model,
      ollama.target,
      source.language?.name,
      bibleNow,
      learnBible,
      learnTerms,
      learnStory,
    ],
  )

  /**
   * Read the folder: every page found and read, and then the chapter itself read
   * whole out of what they said. One run rather than two, so there is one card
   * and one Stop — reading the chapter is part of reading it. Nothing is hidden
   * here: the clean is the slow half, and what a chapter turns out to be is worth
   * having before it is paid for.
   */
  const readFolder = useCallback(
    (folder: GalleryFolder) => {
      const pages = images.filter((image) => image.folder === folder.id)
      void startBatch(folder, pages, 'Reading', examine, (say) =>
        surveyChapter(folder, pages, say),
      )
    },
    [startBatch, images, examine, surveyChapter],
  )

  /** And then hide the words and letter it, every page against the whole chapter. */
  const translateFolder = useCallback(
    (folder: GalleryFolder) => {
      void startBatch(
        folder,
        images.filter((image) => image.folder === folder.id),
        'Cleaning & translating',
        render,
      )
    },
    [startBatch, images, render],
  )

  const removeFolder = useCallback(
    (id: string) => {
      if (batch?.folder === id) stopBatch()
      for (const image of images) if (image.folder === id) forget(image.id)
      forgetTerms(id)
      forgetStory(id)
      forgetBible(id)
      dropFolder(id)
    },
    [
      batch?.folder,
      stopBatch,
      images,
      forget,
      forgetTerms,
      forgetStory,
      forgetBible,
      dropFolder,
    ],
  )

  const clearAll = useCallback(() => {
    stopBatch()
    clear()
    clearMasks()
    clearCleaned()
    traced.clear()
    clearTerms()
    clearStories()
    clearBibles()
    setLettering({})
    setAnalyses({})
    setActiveId(null)
  }, [
    stopBatch,
    clear,
    clearMasks,
    clearCleaned,
    traced,
    clearTerms,
    clearStories,
    clearBibles,
  ])

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
      // Exactly what the board is showing, so what comes out is what was arranged.
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
   * A whole folder back out as the archive it came in as, drawn afresh on the
   * click. **One page at a time**: forty page-sized canvases in the air together
   * buy nothing. A page that will not compose goes in as it arrived rather than
   * being left out — dropping one renumbers everything after it.
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
          open={openFolder}
          onOpenFolder={setOpenFolder}
          onNewFolder={newFolder}
          terms={openFolder ? (chapterTerms[openFolder] ?? []) : []}
          story={openFolder ? (chapterStories[openFolder] ?? null) : null}
          bible={openFolder ? (chapterBibles[openFolder] ?? null) : null}
          onCorrect={(name, fact, value) =>
            openFolder && correctStory(openFolder, name, fact, value)
          }
          onCorrectChapter={(field, value) =>
            openFolder && correctBible(openFolder, field, value)
          }
          onCorrectTerm={(source, target) =>
            openFolder && correctTerm(openFolder, source, target)
          }
          onForgetTerms={() => {
            if (!openFolder) return
            forgetTerms(openFolder)
            forgetStory(openFolder)
            forgetBible(openFolder)
          }}
          activeId={active?.id ?? null}
          onOpen={setActiveId}
          onRemove={removeImage}
          onRemoveFolder={removeFolder}
          onFiles={addTo}
          dragging={dragging}
          busy={busy}
          notice={notice}
          onDismissNotice={dismissNotice}
          onClearAll={clearAll}
          batch={batch}
          // The bar names the page it is working on, so it wants that page's
          // stage rather than the board's.
          batchStage={working && working.id === batch?.page?.id ? working.stage : null}
          onReadFolder={readFolder}
          onTranslateFolder={translateFolder}
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
