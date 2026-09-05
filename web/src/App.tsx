import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Board } from './components/Board'
import { RegionsPanel } from './components/RegionsPanel'
import { Settings } from './components/Settings'
import { Sidebar } from './components/Sidebar'
import { TranslationsPanel } from './components/TranslationsPanel'
import { ChapterReview } from './components/ChapterReview'
import { GearIcon } from './components/icons'
import { IconButton } from './components/ui'
import type { Phase } from './hooks/useBatch'
import { useBatch } from './hooks/useBatch'
import { useFileDrop } from './hooks/useFileDrop'
import { useImageLibrary } from './hooks/useImageLibrary'
import { useLanguage } from './hooks/useLanguage'
import { useLetterMasks } from './hooks/useLetterMasks'
import { useMasks } from './hooks/useMasks'
import { useObjectUrls } from './hooks/useObjectUrls'
import { useLlamaCpp } from './hooks/useLlamaCpp'
import { usePrompt } from './hooks/usePrompt'
import type { Analysis, Box, Fill, Lettering, Region, Stage, Tool } from './lib/api'
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
import { halves, insertAt, insertionFor, moveAt, movedIndex } from './lib/order'
import * as blocks from './lib/regions'
import type { Packed } from './lib/zip'
import { pack } from './lib/zip'

function without<T>(record: Record<string, T>, key: string): Record<string, T> {
  return Object.fromEntries(Object.entries(record).filter(([held]) => held !== key))
}

const newRegion = (box: Box, from?: Region): Region => ({
  id: crypto.randomUUID(),
  box,
  confidence: from?.confidence ?? 1,
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

  const [openFolder, setOpenFolder] = useState<string | null>(null)
  const openNow = useRef(openFolder)
  openNow.current = openFolder

  const addTo = useCallback(
    (files: FileList | File[] | null) => void add(files, openNow.current ?? undefined),
    [add],
  )
  const dragging = useFileDrop(addTo)

  const newFolder = useCallback(
    (name: string) => {
      const made = makeFolder(name)
      if (made) setOpenFolder(made)
      return made !== null
    },
    [makeFolder],
  )

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
  const llamaCpp = useLlamaCpp()
  const source = useLanguage()
  const { prompt, setPrompt, builtIn: builtInPrompt } = usePrompt()

  const [activeId, setActiveId] = useState<string | null>(null)
  const active = images.find((image) => image.id === activeId) ?? null

  const [analyses, setAnalyses] = useState<Record<string, Analysis>>({})
  const [lettering, setLettering] = useState<Record<string, Lines>>({})

  const [working, setWorking] = useState<{ id: string; stage: Stage } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<number | null>(null)
  const [tool, setTool] = useState<Tool>('boxes')
  const [applying, setApplying] = useState(false)
  const [packing, setPacking] = useState<{ done: number; total: number } | null>(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [reviewing, setReviewing] = useState<string | null>(null)
  const [showCleaned, setShowCleaned] = useState(false)

  const [spread, setSpread] = useState(4)
  const [fill, setFill] = useState<Fill>('white')

  const analysis = active ? (analyses[active.id] ?? null) : null
  const pageLettering = active ? (lettering[active.id] ?? []) : []
  const lettered = Object.entries(lettering)
    .filter(([, set]) => set.some(Boolean))
    .map(([id]) => id)
  const cleanedPage = active ? (cleanedPages[active.id] ?? null) : null
  const stage = working?.id === active?.id ? (working?.stage ?? null) : null

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

  const analysesNow = useRef(analyses)
  analysesNow.current = analyses
  const cleanedNow = useRef(cleanedPages)
  cleanedNow.current = cleanedPages
  const letteringNow = useRef(lettering)
  letteringNow.current = lettering

  const activeNow = useRef(activeId)
  activeNow.current = activeId
  const onBoard = useCallback((id: string) => id === activeNow.current, [])

  const imagesNow = useRef(images)
  imagesNow.current = images
  const held = useCallback((id: string) => imagesNow.current.some((it) => it.id === id), [])


  useEffect(() => {
    if (!activeId) return
    const set = letteringNow.current[activeId]
    setTool(set?.some(Boolean) ? 'text' : 'boxes')
  }, [activeId])

  useEffect(() => setShowCleaned(false), [activeId])
  useEffect(() => {
    if (cleanedPage) setShowCleaned(true)
  }, [cleanedPage])

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

  const lastFailure = useRef<string | null>(null)

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

  const detectAndRead = useCallback(
    async (page: GalleryImage): Promise<Analysis | null> => {
      const { id, file } = page
      if (onBoard(id)) setSelected(null)

      return during(id, 'detecting', async () => {
        const detection = await detect(file, source.code)
        let found: Analysis = {
          detection,
          texts: detection.regions.length === 0 ? [] : null,
          excluded: detection.regions.flatMap((region, index) =>
            region.confidence < UNSURE ? [index] : [],
          ),
        }
        setAnalyses((current) => ({ ...current, [id]: found }))
        setLettering((current) => without(current, id))

        if (detection.regions.length > 0) {
          setWorking({ id, stage: 'reading' })
          const texts = await read(
            file,
            detection.regions.map((region) => region.box),
            source.code,
          )
          found = { ...found, texts }
          setAnalyses((current) =>
            id in current ? { ...current, [id]: found } : current,
          )
        }
        return found
      })
    },
    [during, onBoard, source.code],
  )

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

  const markLetters = useCallback(
    async (page: GalleryImage, boxes: Box[]) => {
      const mask = forPage(page)
      if (!mask || mask.empty) return
      const letters = await tracePage(page)
      if (!mask.empty) mark(mask, boxes, letters)
    },
    [forPage, tracePage],
  )

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

      if (putBack) await markLetters(active, [box])
      else forPage(active)?.boxes([box], true)
    },
    [active, analyses, forPage, markLetters],
  )

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
          ? { ...current, [active.id]: insertAt(current[active.id], at, null) }
          : current,
      )
      setSelected(at)

      await markLetters(active, [box])
      await reread(active, [box], [added.id])
    },
    [active, analyses, markLetters, reread, source.rtl],
  )

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

  const rereadRegion = useCallback(
    async (index: number, was: Box) => {
      if (!active) return
      const held = analyses[active.id]
      const region = held?.detection.regions[index]
      if (!held || !region || region.box.join() === was.join()) return

      forPage(active)?.boxes([was], true)
      if (!held.excluded.includes(index)) await markLetters(active, [region.box])

      if (!held.texts) return
      await reread(active, [region.box], [region.id])
    },
    [active, analyses, forPage, markLetters, reread],
  )

  const moveRegion = useCallback(
    (from: number, to: number) => {
      if (!active || from === to) return
      const { id } = active

      setAnalyses((current) =>
        current[id] ? { ...current, [id]: blocks.moved(current[id], from, to) } : current,
      )
      setLettering((current) =>
        current[id] ? { ...current, [id]: moveAt(current[id], from, to) } : current,
      )
      setSelected((now) => (now === null ? now : movedIndex(now, from, to)))
    },
    [active],
  )

  const splitRegion = useCallback(
    async (index: number, at: number) => {
      if (!active) return
      const held = analyses[active.id]
      const region = held?.detection.regions[index]
      const line = lettering[active.id]?.[index]
      if (!held || !region || !line) return

      const before = line.text.slice(0, at).trim()
      const rest = line.text.slice(at).trim()
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

  const translatePage = useCallback(
    async (page: GalleryImage, found: Analysis): Promise<boolean> => {
      if (!llamaCpp.model || !found.texts) return false

      const skip = new Set(found.excluded)
      const wanted = found.texts
        .map((text, index) => ({ text, index }))
        .filter(({ text, index }) => text.trim() && !skip.has(index))
      if (wanted.length === 0) return false

      const got = await during(page.id, 'translating', async () => {
        await ready()
        const sending = wanted.map(({ text, index }) => {
          const region = found.detection.regions[index]
          return {
            text,
            kind: region?.kind,
            budget: region ? lines.budgetFor(region, text) : undefined,
          }
        })
        return translate(sending, llamaCpp.model, llamaCpp.target, {
          system: prompt,
          source: source.language?.name,
        })
      })
      if (!got) return false

      const set: Lines = found.detection.regions.map(() => null)
      wanted.forEach((line, at) => {
        const text = (got[at] ?? '').trim()
        if (text) set[line.index] = lines.laidOut(found, line.index, text)
      })
      setLettering((current) => ({ ...current, [page.id]: set }))
      if (onBoard(page.id)) setTool('text')
      return true
    },
    [
      during,
      onBoard,
      llamaCpp.model,
      llamaCpp.target,
      prompt,
      source.language?.name,
    ],
  )

  const runDetect = useCallback(async () => {
    if (active) await detectAndRead(active)
  }, [active, detectAndRead])

  const runClean = useCallback(
    async (marks: Blob) => {
      if (active) await cleanPage(active, marks)
    },
    [active, cleanPage],
  )

  const runTranslate = useCallback(async () => {
    const found = active ? analyses[active.id] : null
    if (active && found) await translatePage(active, found)
  }, [active, analyses, translatePage])

  const readPage = useCallback(
    async (
      page: GalleryImage,
    ): Promise<{ found: Analysis | null; why: string | null }> => {
      if (!held(page.id)) return { found: null, why: null }

      const found = await detectAndRead(page)
      if (!found) {
        return { found: null, why: lastFailure.current ?? 'the text could not be found' }
      }

      return { found, why: null }
    },
    [detectAndRead, held],
  )

  const hidePage = useCallback(
    async (page: GalleryImage, found: Analysis): Promise<string | null> => {
      const marks = await marksFor(page, found)
      if (marks) {
        lastFailure.current = null
        if (!(await cleanPage(page, marks))) {
          return lastFailure.current ?? 'the page could not be cleaned'
        }
      }

      if (!onBoard(page.id)) traced.drop(page.id)
      return null
    },
    [marksFor, cleanPage, traced, onBoard],
  )

  const examine = useCallback(
    async (page: GalleryImage) => (await readPage(page)).why,
    [readPage],
  )

  const translateOne = useCallback(
    async (page: GalleryImage): Promise<string | null> => {
      if (!held(page.id)) return null
      const found = analysesNow.current[page.id]
      if (!found?.texts || !found.texts.some((text) => text.trim())) return null

      lastFailure.current = null
      await translatePage(page, found)
      return lastFailure.current
    },
    [held, translatePage],
  )

  const cleanOne = useCallback(
    async (page: GalleryImage): Promise<string | null> => {
      if (!held(page.id)) return null
      const found = analysesNow.current[page.id]
      if (!found || cleanedNow.current[page.id]) return null
      return hidePage(page, found)
    },
    [held, hidePage],
  )

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

      if (llamaCpp.model && found.texts.some((text) => text.trim())) {
        lastFailure.current = null
        await translatePage(page, found)
        if (lastFailure.current) return lastFailure.current
      }

      if (!cleanedNow.current[page.id]) {
        const why = await hidePage(page, found)
        if (why) return why
      }
      return null
    },
    [readPage, hidePage, translatePage, held, llamaCpp.model],
  )

  const runAll = useCallback(() => {
    if (active) void render(active)
  }, [active, render])

  const {
    run: batch,
    start: startBatch,
    stop: stopBatch,
    dismiss: dismissBatch,
  } = useBatch()

  const wasRunning = useRef(false)
  useEffect(() => {
    const running = batch !== null && !batch.finished
    if (wasRunning.current && batch?.finished && !batch.stopping) {
      setReviewing(batch.folder)
    }
    wasRunning.current = running
  }, [batch])

  const translateFolder = useCallback(
    (folder: GalleryFolder) => {
      const pages = images.filter((image) => image.folder === folder.id)

      const phases: Phase[] = [{ name: 'Reading', each: examine, blocking: true }]
      if (llamaCpp.model) phases.push({ name: 'Translating', each: translateOne })
      phases.push({ name: 'Cleaning', each: cleanOne })

      void startBatch(folder, pages, phases)
    },
    [
      startBatch,
      images,
      examine,
      translateOne,
      cleanOne,
      llamaCpp.model,
    ],
  )

  const removeFolder = useCallback(
    (id: string) => {
      if (batch?.folder === id) stopBatch()
      if (reviewing === id) setReviewing(null)
      for (const image of images) if (image.folder === id) forget(image.id)
      dropFolder(id)
    },
    [
      batch?.folder,
      stopBatch,
      reviewing,
      images,
      forget,
      dropFolder,
    ],
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
    setReviewing(null)
  }, [
    stopBatch,
    clear,
    clearMasks,
    clearCleaned,
    traced,
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

  const setLetteringAngle = useCallback(
    (index: number, angle: number) =>
      changeLettering(index, { angle: ((angle % 360) + 360) % 360 }),
    [changeLettering],
  )

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

  const applyToImage = useCallback(async () => {
    if (!active) return
    const set = lettering[active.id]
    if (!set?.some(Boolean)) return

    setApplying(true)
    setError(null)
    try {
      const base = showCleaned && cleanedPage ? cleanedPage : active.url
      const page = await compose(base, active.width, active.height, set)
      save(page, `${stem(active.name)}-lettered.png`)
    } catch (cause) {
      setError(said(cause))
    } finally {
      setApplying(false)
    }
  }, [active, lettering, cleanedPage, showCleaned])

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

        save(await pack(held), archiveName(folder, llamaCpp.target, anyLettered))
      } catch (cause) {
        setError(said(cause))
      } finally {
        setPacking(null)
      }
    },
    [lettering, llamaCpp.target, packing],
  )

  const reviewFolder = folders.find((held) => held.id === reviewing) ?? null
  const reviewFailed = useMemo(
    () =>
      Object.fromEntries((batch?.failed ?? []).map((gone) => [gone.id, gone.why])),
    [batch?.failed],
  )

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
          models={llamaCpp.models}
        />
      )}

      <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
        <Sidebar
          images={images}
          folders={folders}
          open={openFolder}
          onOpenFolder={setOpenFolder}
          onNewFolder={newFolder}
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
          batchStage={working && working.id === batch?.page?.id ? working.stage : null}
          onTranslateFolder={translateFolder}
          onStopBatch={stopBatch}
          onDismissBatch={dismissBatch}
          onReviewBatch={() => batch && setReviewing(batch.folder)}
          canTranslate={Boolean(llamaCpp.model)}
          lettered={lettered}
          onDownloadFolder={(folder) => void downloadFolder(folder)}
          workedOn={workedOn}
          packing={packing}
        />

        {reviewFolder ? (
          <ChapterReview
            folder={reviewFolder}
            pages={images.filter((image) => image.folder === reviewFolder.id)}
            analyses={analyses}
            lettering={lettering}
            failed={reviewFailed}
            onEdit={(id) => {
              setActiveId(id)
              setReviewing(null)
            }}
            onClose={() => setReviewing(null)}
            onDownload={() => void downloadFolder(reviewFolder)}
            packing={packing}
          />
        ) : (
          <Board
            image={active}
            analysis={analysis}
            mask={forPage(active)}
            cleaned={cleanedPage}
            stage={stage}
            error={error}
            selected={selected}
            onSelect={setSelected}
            tool={tool}
            onTool={setTool}
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
              models: llamaCpp.models,
              model: llamaCpp.model,
              onModel: llamaCpp.setModel,
              target: llamaCpp.target,
              onTarget: llamaCpp.setTarget,
              onTranslate: runTranslate,
              lettering: pageLettering,
              onBox: setLetteringBox,
              onTurn: setLetteringAngle,
              onSize: nudgeSize,
              onApply: applyToImage,
              applying,
              note:
                llamaCpp.problem ??
                (!analysis?.texts
                  ? 'find the text first: there is nothing to translate yet'
                  : pageLettering.some(Boolean) && !cleanedPage
                    ? 'this page has not been cleaned yet'
                    : null),
            }}
          />
        )}

        {!reviewFolder &&
          active &&
          analysis &&
          (tool === 'text' ? (
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
