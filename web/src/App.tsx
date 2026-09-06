import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { SetStateAction } from 'react'
import { Board } from './components/Board'
import { RegionsPanel } from './components/RegionsPanel'
import { Settings } from './components/Settings'
import { Sidebar } from './components/Sidebar'
import { TranslationsPanel } from './components/TranslationsPanel'
import { ChapterReview } from './components/ChapterReview'
import { GearIcon } from './components/icons'
import { Button, IconButton } from './components/ui'
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
import { usePageOperation } from './hooks/usePageOperation'
import type { PageOperation } from './hooks/usePageOperation'
import { useProject } from './hooks/useProject'
import type { ProjectData } from './lib/project'
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
import type { ChapterContext } from './lib/chapter'
import { archiveName, chapterReference, finished } from './lib/chapter'
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
    restore: restoreLibrary,
  } = useImageLibrary()

  const [openFolder, setOpenFolder] = useState<string | null>(null)
  const openNow = useRef(openFolder)
  openNow.current = openFolder

  const addTo = useCallback(
    (files: FileList | File[] | null) => {
      if (restored.current) void add(files, openNow.current ?? undefined)
    },
    [add],
  )
  const dragging = useFileDrop(addTo)

  const newFolder = useCallback(
    (name: string) => {
      if (!restored.current) return false
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
  const projectChanged = useRef<() => void>(() => {})
  const markProjectChanged = useCallback(() => projectChanged.current(), [])
  const {
    get: pageMask, load: loadMask, release: releaseMask, drop: dropMask, clear: clearMasks,
    snapshot: snapshotMasks, restore: restoreMasks,
  } = useMasks(markProjectChanged)
  const {
    get: touchupMask, load: loadTouchups, release: releaseTouchups, drop: dropTouchups, clear: clearTouchups,
    snapshot: snapshotTouchups, restore: restoreTouchups,
  } = useMasks(markProjectChanged)
  const { at: letterMaskAt, keep: keepLetterMask, drop: dropLetterMask, clear: clearLetterMasks } = useLetterMasks()
  const {
    urls: cleanedPages,
    blobs: cleanedBlobs,
    set: setCleaned,
    drop: dropCleaned,
    clear: clearCleaned,
    restore: restoreCleaned,
  } = useObjectUrls()
  const llamaCpp = useLlamaCpp()
  const source = useLanguage()
  const { setCode } = source
  const { setModel, setTarget } = llamaCpp
  const { prompt, setPrompt, builtIn: builtInPrompt } = usePrompt()

  const [activeId, setActiveId] = useState<string | null>(null)
  const active = images.find((image) => image.id === activeId) ?? null

  const analysesNow = useRef<Record<string, Analysis>>({})
  const [analyses, rememberAnalyses] = useState(analysesNow.current)
  const setAnalyses = useCallback((update: SetStateAction<Record<string, Analysis>>) => {
    analysesNow.current = typeof update === 'function' ? update(analysesNow.current) : update
    rememberAnalyses(analysesNow.current)
  }, [])
  const letteringNow = useRef<Record<string, Lines>>({})
  const [lettering, rememberLettering] = useState(letteringNow.current)
  const setLettering = useCallback((update: SetStateAction<Record<string, Lines>>) => {
    letteringNow.current = typeof update === 'function' ? update(letteringNow.current) : update
    rememberLettering(letteringNow.current)
  }, [])

  const { begin, cancel: abortPage } = usePageOperation()
  const operating = useRef<PageOperation | null>(null)
  const [working, setWorking] = useState<{
    id: string; stage: Stage; operation: PageOperation
  } | null>(null)
  const cancel = useCallback((id?: string) => {
    abortPage(id)
    if (!id || operating.current?.pageId === id) operating.current = null
    setWorking((now) => !id || now?.id === id ? null : now)
  }, [abortPage])
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<number | null>(null)
  const [tool, setTool] = useState<Tool>('boxes')
  const [applying, setApplying] = useState(false)
  const [packing, setPacking] = useState<{ done: number; total: number } | null>(null)
  const [exportFailure, setExportFailure] = useState<{
    folder: GalleryFolder
    message: string
  } | null>(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [reviewing, setReviewing] = useState<string | null>(null)
  const [showCleaned, setShowCleaned] = useState(false)
  const [detailsOpen, setDetailsOpen] = useState(false)

  const [spread, setSpread] = useState(4)
  const [fill, setFill] = useState<Fill>('white')

  const snapshotProject = useCallback(async (): Promise<ProjectData> => ({
    images: images.map(({ url: _url, ...image }) => image),
    folders, analyses, lettering, cleaned: cleanedBlobs,
    masks: await snapshotMasks(),
    touchups: await snapshotTouchups(),
    activeId, openFolder, language: source.code,
    model: llamaCpp.model, target: llamaCpp.target, prompt, spread, fill,
  }), [
    images, folders, analyses, lettering, cleanedBlobs, snapshotMasks, snapshotTouchups,
    activeId, openFolder, source.code, llamaCpp.model, llamaCpp.target, prompt, spread, fill,
  ])
  const restoreProject = useCallback(async (saved: ProjectData) => {
    await restoreMasks(saved.masks)
    await restoreTouchups(saved.touchups)
    restoreLibrary(saved.images, saved.folders)
    restoreCleaned(saved.cleaned)
    setAnalyses(saved.analyses)
    setLettering(saved.lettering)
    setActiveId(saved.activeId)
    setOpenFolder(saved.openFolder)
    setCode(saved.language)
    setModel(saved.model)
    setTarget(saved.target)
    setPrompt(saved.prompt)
    setSpread(saved.spread)
    setFill(saved.fill)
  }, [
    restoreMasks, restoreTouchups, restoreLibrary, restoreCleaned, setAnalyses, setLettering,
    setCode, setModel, setTarget, setPrompt,
  ])
  const project = useProject(snapshotProject, restoreProject)
  projectChanged.current = project.changed

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

  const restored = useRef(false)
  useEffect(() => { restored.current = project.ready }, [project.ready])
  const cleanedNow = useRef(cleanedPages)
  cleanedNow.current = cleanedPages

  const activeNow = useRef(activeId)
  activeNow.current = activeId
  const onBoard = useCallback((id: string) => id === activeNow.current, [])

  const imagesNow = useRef(images)
  imagesNow.current = images
  const held = useCallback((id: string) => imagesNow.current.some((it) => it.id === id), [])

  const releaseIdle = useCallback(async (id: string) => {
    if (onBoard(id) || operating.current?.pageId === id) return
    dropLetterMask(id)
    await Promise.all([releaseMask(id), releaseTouchups(id)])
  }, [onBoard, dropLetterMask, releaseMask, releaseTouchups])

  const activeMask = pageMask(active)
  const activeTouchups = touchupMask(active)
  const [, refreshMasks] = useState(0)
  const maskPage = useRef<string | null>(null)
  const maskQueue = useRef(Promise.resolve())
  useEffect(() => {
    let current = true
    // Serialize navigation so rapid page changes cannot queue full-size decodes.
    maskQueue.current = maskQueue.current.then(async () => {
      if (!current) return
      const previous = maskPage.current
      if (previous && previous !== active?.id) await releaseIdle(previous)
      if (!current) return
      maskPage.current = active?.id ?? null
      if (!active) return
      const [mask, touchups] = await Promise.all([loadMask(active), loadTouchups(active)])
      if (!current) {
        await releaseIdle(active.id)
      } else if (mask !== activeMask || touchups !== activeTouchups) {
        refreshMasks((version) => version + 1)
      }
    }).catch((cause) => { if (current) setError(said(cause)) })
    return () => { current = false }
  }, [active, activeMask, activeTouchups, loadMask, loadTouchups, releaseIdle])


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
      cancel(id)
      dropTouchups(id)
      dropMask(id)
      dropCleaned(id)
      dropLetterMask(id)
      setLettering((current) => without(current, id))
      setAnalyses((current) => without(current, id))
    },
    [cancel, dropTouchups, dropMask, dropCleaned, dropLetterMask, setLettering, setAnalyses],
  )

  const removeImage = useCallback(
    (id: string) => {
      remove(id)
      forget(id)
    },
    [remove, forget],
  )

  const lastFailure = useRef<string | null>(null)

  const workOn = useCallback(
    async <T,>(
      page: GalleryImage,
      step: (operation: PageOperation) => Promise<T>,
      signal?: AbortSignal,
    ): Promise<T | null> => {
      if (!held(page.id) || !restored.current) return null
      const operation = begin(page.id, signal)
      operating.current = operation
      lastFailure.current = null
      setError(null)
      let result: T | null = null
      try {
        operation.check()
        result = await step(operation)
        operation.check()
      } catch (cause) {
        if (operation.signal.aborted) {
          if (signal) throw cause
        } else {
          lastFailure.current = said(cause)
          setError(lastFailure.current)
        }
        result = null
      } finally {
        operation.finish()
        if (operating.current === operation) operating.current = null
        try {
          await releaseIdle(page.id)
        } catch (cause) {
          if (!operating.current) {
            lastFailure.current = said(cause)
            setError(lastFailure.current)
          }
          result = null
        }
        setWorking((now) => now?.operation === operation ? null : now)
      }
      return result
    },
    [begin, held, releaseIdle],
  )

  const during = useCallback(
    async <T,>(operation: PageOperation, stage: Stage, step: () => Promise<T>): Promise<T> => {
      operation.check()
      setWorking({ id: operation.pageId, stage, operation })
      const result = await step()
      operation.check()
      return result
    },
    [],
  )

  const reread = useCallback(
    async (page: GalleryImage, boxes: Box[], ids: string[], operation: PageOperation) => {
      const every = analysesNow.current[page.id]?.detection.regions ?? []
      const everyId = every.map((region) => region.id)

      const [texts, balloons] = await during(operation, 'reading', () =>
        Promise.all([
          read(page.file, boxes, source.code, operation.signal),
          bubbles(
            page.file,
            every.map((region) => region.box),
            operation.signal,
          ),
        ]),
      )

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
    [during, source.code, setAnalyses],
  )

  const detectAndRead = useCallback(
    async (page: GalleryImage, operation: PageOperation): Promise<Analysis> => {
      const { id, file } = page
      if (onBoard(id)) setSelected(null)
      let found = analysesNow.current[id]
      if (!found) {
        const detection = await during(operation, 'detecting', () =>
          detect(file, source.code, operation.signal),
        )
        found = {
          detection,
          texts: detection.regions.length === 0 ? [] : null,
          excluded: detection.regions.flatMap((region, index) =>
            region.confidence < UNSURE ? [index] : [],
          ),
        }
        setAnalyses((current) => ({ ...current, [id]: found }))
      }
      if (found.texts === null) {
        const texts = await during(operation, 'reading', () =>
          read(file, found.detection.regions.map((region) => region.box), source.code, operation.signal),
        )
        found = { ...found, texts }
        setAnalyses((current) => ({ ...current, [id]: found }))
      }
      return found
    },
    [during, onBoard, source.code, setAnalyses],
  )

  const tracePage = useCallback(
    async (page: GalleryImage, operation: PageOperation): Promise<ImageBitmap> => {
      operation.check()
      const held = letterMaskAt(page.id, spread)
      if (held) return held

      return during(operation, 'tracing', async () => {
        const bitmap = await createImageBitmap(await letterMask(page.file, spread, operation.signal))
        if (operation.signal.aborted) {
          bitmap.close()
          operation.check()
        }
        keepLetterMask(page.id, spread, bitmap)
        return bitmap
      })
    },
    [during, letterMaskAt, keepLetterMask, spread],
  )

  const traceLetters = useCallback(
    () => active ? workOn(active, (operation) => tracePage(active, operation)) : Promise.resolve(null),
    [active, workOn, tracePage],
  )

  const markLetters = useCallback(
    async (page: GalleryImage, boxes: Box[], operation: PageOperation) => {
      const mask = await loadMask(page)
      operation.check()
      if (!mask || mask.empty) return
      const letters = await tracePage(page, operation)
      operation.check()
      if (!mask.empty) mark(mask, boxes, letters)
    },
    [loadMask, tracePage],
  )

  const toggleExcluded = useCallback(
    async (index: number) => {
      if (!active) return
      const held = analyses[active.id]
      const box = held?.detection.regions[index]?.box
      if (!held || !box) return
      cancel(active.id)

      const putBack = held.excluded.includes(index)
      setAnalyses((current) => ({
        ...current,
        [active.id]: blocks.toggledExcluded(held, index),
      }))

      if (putBack) await workOn(active, (operation) => markLetters(active, [box], operation))
      else await workOn(active, async (operation) => {
        const mask = await loadMask(active)
        operation.check()
        mask?.boxes([box], true)
      })
    },
    [active, analyses, cancel, loadMask, markLetters, workOn, setAnalyses],
  )

  const addRegion = useCallback(
    async (box: Box) => {
      if (!active) return
      const held = analyses[active.id]
      if (!held) return
      cancel(active.id)

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

      await workOn(active, async (operation) => {
        await markLetters(active, [box], operation)
        await reread(active, [box], [added.id], operation)
      })
    },
    [active, analyses, cancel, markLetters, reread, source.rtl, workOn, setAnalyses, setLettering],
  )

  const setRegionBox = useCallback(
    (index: number, box: Box) => {
      if (!active) return
      cancel(active.id)
      setAnalyses((current) => {
        const held = current[active.id]
        if (!held) return current
        return { ...current, [active.id]: blocks.withBox(held, index, box) }
      })
    },
    [active, cancel, setAnalyses],
  )

  const rereadRegion = useCallback(
    async (index: number, was: Box) => {
      if (!active) return
      const held = analyses[active.id]
      const region = held?.detection.regions[index]
      if (!held || !region || region.box.join() === was.join()) return

      await workOn(active, async (operation) => {
        const mask = await loadMask(active)
        operation.check()
        mask?.boxes([was], true)
        if (!held.excluded.includes(index)) await markLetters(active, [region.box], operation)
        if (held.texts) await reread(active, [region.box], [region.id], operation)
      })
    },
    [active, analyses, loadMask, markLetters, reread, workOn],
  )

  const moveRegion = useCallback(
    (from: number, to: number) => {
      if (!active || from === to) return
      const { id } = active
      cancel(id)

      setAnalyses((current) =>
        current[id] ? { ...current, [id]: blocks.moved(current[id], from, to) } : current,
      )
      setLettering((current) =>
        current[id] ? { ...current, [id]: moveAt(current[id], from, to) } : current,
      )
      setSelected((now) => (now === null ? now : movedIndex(now, from, to)))
    },
    [active, cancel, setAnalyses, setLettering],
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
      cancel(active.id)

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

      await workOn(active, (operation) =>
        reread(active, [firstBox, secondBox], [region.id, added.id], operation),
      )
    },
    [active, analyses, lettering, cancel, reread, source.rtl, workOn, setAnalyses, setLettering],
  )

  const marksFor = useCallback(
    async (page: GalleryImage, found: Analysis, operation: PageOperation): Promise<Blob | null> => {
      const mask = await loadMask(page)
      operation.check()
      if (!mask) return null

      if (mask.empty) {
        const boxes = blocks.toClean(found)
        if (boxes.length === 0) return null
        const letters = await tracePage(page, operation)
        operation.check()
        mark(mask, boxes, letters)
      }
      return mask.toBlob()
    },
    [loadMask, tracePage],
  )

  const cleanPage = useCallback(
    async (page: GalleryImage, marks: Blob, operation: PageOperation): Promise<void> => {
      const cleaned = await during(operation, 'cleaning', () =>
        clean(page.file, marks, fill, operation.signal),
      )
      setCleaned(page.id, cleaned)
    },
    [during, setCleaned, fill],
  )

  const translatePage = useCallback(
    async (
      page: GalleryImage,
      found: Analysis,
      operation: PageOperation,
      chapter?: ChapterContext,
      resume = false,
    ): Promise<void> => {
      if (!llamaCpp.model || !found.texts) return

      const skip = new Set(found.excluded)
      const wanted = found.texts
        .map((text, index) => ({ text, index }))
        .filter(({ text, index }) =>
          text.trim() && !skip.has(index) && (!resume || !letteringNow.current[page.id]?.[index]),
        )
      if (wanted.length === 0) return

      const got = await during(operation, 'translating', async () => {
        await ready()
        operation.check()
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
          context: chapter ? chapterReference(chapter, page.id) : undefined,
        }, operation.signal)
      })

      const set: Lines = found.detection.regions.map((_, index) =>
        resume ? letteringNow.current[page.id]?.[index] ?? null : null,
      )
      wanted.forEach((line, at) => {
        const text = (got[at] ?? '').trim()
        if (text) set[line.index] = lines.laidOut(found, line.index, text)
      })
      setLettering((current) => ({ ...current, [page.id]: set }))
      if (onBoard(page.id)) setTool('text')
    },
    [
      during,
      onBoard,
      llamaCpp.model,
      llamaCpp.target,
      prompt,
      source.language?.name,
      setLettering,
    ],
  )

  const runDetect = useCallback(async () => {
    if (!active || !pageMask(active)) return
    if (
      (analysesNow.current[active.id] || cleanedNow.current[active.id] || pageMask(active)?.empty === false) &&
      !window.confirm('Find text again? This discards this page’s regions, translations, masks, and cleanup.')
    ) return
    forget(active.id)
    await workOn(active, (operation) => detectAndRead(active, operation))
  }, [active, forget, pageMask, workOn, detectAndRead])

  const runClean = useCallback(
    async () => {
      if (!active) return
      const page = active
      const base = cleanedNow.current[page.id]
      await workOn(page, async (operation) => {
        const mask = await (base ? loadTouchups(page) : loadMask(page))
        operation.check()
        if (!mask || mask.empty) return
        const cleaned = await during(operation, 'cleaning', async () => {
          const file = base
            ? new File([await (await fetch(base, { signal: operation.signal })).blob()], page.name, { type: 'image/png' })
            : page.file
          return clean(file, await mask.toBlob(), fill, operation.signal)
        })
        if (base) mask.clear()
        setCleaned(page.id, cleaned)
      })
    },
    [active, during, fill, workOn, loadTouchups, loadMask, setCleaned],
  )

  const runTranslate = useCallback(async () => {
    const found = active ? analyses[active.id] : null
    if (active && found) {
      await workOn(active, (operation) => translatePage(active, found, operation))
    }
  }, [active, analyses, workOn, translatePage])


  const hidePage = useCallback(
    async (page: GalleryImage, found: Analysis, operation: PageOperation): Promise<void> => {
      const marks = await marksFor(page, found, operation)
      operation.check()
      if (marks) await cleanPage(page, marks, operation)
    },
    [marksFor, cleanPage],
  )


  const runAll = useCallback(async () => {
    if (!active) return
    await workOn(active, async (operation) => {
      const found = await detectAndRead(active, operation)
      await translatePage(active, found, operation, undefined, true)
      if (!cleanedNow.current[active.id]) await hidePage(active, found, operation)
    })
  }, [active, workOn, detectAndRead, translatePage, hidePage])

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
    (folder: GalleryFolder, reprocess = false) => {
      if (batch && !batch.finished) return
      if (reprocess && !window.confirm(
        `Reprocess all pages in ${folder.name}? This discards saved regions, translations, masks, and cleanup. Originals are kept.`,
      )) return
      const pages = images
        .filter((image) => image.folder === folder.id)
        .sort((one, other) => one.file.name.localeCompare(other.file.name, undefined, {
          numeric: true,
          sensitivity: 'base',
        }))
      if (reprocess) for (const page of pages) forget(page.id)
      const chapter: ChapterContext = { pages, readings: {}, translations: {} }
      const phases: Phase[] = [{
        name: 'Reading',
        blocking: true,
        each: async (page, signal) => {
          if (!held(page.id)) return null
          const found = await workOn(page, (operation) => detectAndRead(page, operation), signal)
          if (found) {
            chapter.readings[page.id] = found
            chapter.translations[page.id] = (found.texts ?? []).flatMap((source, index) => {
              const translated = letteringNow.current[page.id]?.[index]?.text
              return translated?.trim() && !found.excluded.includes(index) ? [{ source, translated }] : []
            })
          }
          return lastFailure.current
        },
      }]
      if (llamaCpp.model) {
        phases.push({
          name: 'Translating',
          blocking: true,
          each: async (page, signal) => {
            const found = chapter.readings[page.id]
            if (!held(page.id) || !found) return null
            await workOn(page, (operation) => translatePage(page, found, operation, chapter, true), signal)
            chapter.translations[page.id] = (found.texts ?? []).flatMap((source, index) => {
              const translated = letteringNow.current[page.id]?.[index]?.text
              return translated?.trim() && !found.excluded.includes(index) ? [{ source, translated }] : []
            })
            return lastFailure.current
          },
        })
      }
      phases.push({
        name: 'Cleaning',
        each: async (page, signal) => {
          const found = chapter.readings[page.id]
          if (!held(page.id) || !found || cleanedNow.current[page.id]) return null
          await workOn(page, (operation) => hidePage(page, found, operation), signal)
          return lastFailure.current
        },
      })
      void startBatch(folder, pages, phases)
    },
    [batch, images, forget, held, workOn, detectAndRead, translatePage, hidePage, startBatch, llamaCpp.model],
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
    cancel()
    clear()
    clearMasks()
    clearTouchups()
    clearCleaned()
    clearLetterMasks()
    setLettering({})
    setAnalyses({})
    setActiveId(null)
    setReviewing(null)
    setOpenFolder(null)
    setExportFailure(null)
  }, [
    stopBatch,
    cancel,
    setAnalyses,
    setLettering,
    clear,
    clearMasks,
    clearCleaned,
    clearTouchups,
    clearLetterMasks,
  ])

  const changeLettering = useCallback(
    (index: number, patch: Partial<Lettering>) => {
      if (!active) return
      const { id } = active
      cancel(id)
      setLettering((current) =>
        current[id]
          ? { ...current, [id]: lines.withLine(current[id], index, patch) }
          : current,
      )
    },
    [active, cancel, setLettering],
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
      const base = cleanedPage ?? active.url
      const page = await compose(base, active.width, active.height, set)
      save(page, `${stem(active.name)}-lettered.png`)
    } catch (cause) {
      setError(said(cause))
    } finally {
      setApplying(false)
    }
  }, [active, lettering, cleanedPage])

  const downloadFolder = useCallback(
    async (folder: GalleryFolder) => {
      const pages = imagesNow.current.filter((image) => image.folder === folder.id)
      if (pages.length === 0 || packing) return

      setPacking({ done: 0, total: pages.length })
      setError(null)
      setExportFailure(null)
      try {
        let completed = 0
        const failed: string[] = []
        let anyLettered = false

        async function* rendered(): AsyncGenerator<Packed> {
          for (const [at, page] of pages.entries()) {
            try {
              const made = await finished(
                page,
                lettering[page.id],
                cleanedNow.current[page.id] ?? null,
              )
              anyLettered ||= made.reached === 'lettered'
              completed += 1
              yield made
            } catch (cause) {
              failed.push(`${page.name}: ${said(cause)}`)
            }
            setPacking({ done: at + 1, total: pages.length })
          }
        }
        const archive = await pack(rendered())

        if (failed.length > 0) {
          const message = `Could not export ${failed.length} page(s):\n${failed.join('\n')}`
          setExportFailure({ folder, message })
          if (
            completed === 0 ||
            !window.confirm(
              `${message}\n\nDownload only the ${completed} successful page(s)? ` +
              'Failed pages will be omitted, not replaced with originals. ' +
              'Cancel to fix the pages or retry the export.',
            )
          ) return
        }

        const name = archiveName(folder, llamaCpp.target, anyLettered)
        save(
          archive,
          failed.length > 0 ? name.replace(/\.(zip|cbz)$/i, '-partial.$1') : name,
        )
      } catch (cause) {
        setExportFailure({ folder, message: said(cause) })
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

  if (!project.ready) {
    return (
      <main className="flex h-screen items-center justify-center bg-canvas p-6 text-ink">
        {project.error ? (
          <div role="alert">
            <p>{project.error}</p>
            <Button className="mt-3" onClick={project.retry}>Retry loading project</Button>
          </div>
        ) : <p role="status">Restoring project…</p>}
      </main>
    )
  }

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-canvas text-ink">
      <header className="flex shrink-0 items-center justify-between gap-4 border-b border-line bg-surface px-4 py-2.5">
        <h1 className="text-sm font-semibold tracking-tight text-ink">manga-trans</h1>
        <div className="flex min-w-0 items-center gap-2">
          <p className="truncate font-mono text-[11px] text-faint">{API_BASE}</p>
          <span role="status" className="text-[11px] text-faint">
            {project.error ? 'Project not saved' : project.saving ? 'Saving project…' : 'Project saved'}
          </span>
          {working && (
            <Button onClick={() => { stopBatch(); cancel() }}>Cancel operation</Button>
          )}
          <IconButton label="Settings" onClick={() => setSettingsOpen(true)}>
            <GearIcon />
          </IconButton>
        </div>
      </header>

      {project.error && (
        <div role="alert" className="shrink-0 border-b border-warn/30 bg-warn/10 px-4 py-3 text-xs text-warn">
          <p>{project.error} Keep this tab open until the project is saved.</p>
          <Button className="mt-2" onClick={project.retry}>Retry saving project</Button>
        </div>
      )}

      {exportFailure && (
        <div role="alert" className="max-h-48 shrink-0 overflow-y-auto border-b border-warn/30 bg-warn/10 px-4 py-3">
          <p className="whitespace-pre-wrap break-words text-xs text-warn">{exportFailure.message}</p>
          <div className="mt-2 flex gap-2">
            <Button
              disabled={packing !== null || !folders.some((folder) => folder.id === exportFailure.folder.id)}
              onClick={() => void downloadFolder(exportFailure.folder)}
            >
              Retry export
            </Button>
            <Button onClick={() => setExportFailure(null)}>Dismiss</Button>
          </div>
        </div>
      )}

      {settingsOpen && (
        <Settings
          onClose={() => setSettingsOpen(false)}
          prompt={prompt}
          fallback={builtInPrompt}
          onSave={(next) => { stopBatch(); cancel(); setPrompt(next) }}
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
          onReprocessFolder={(folder) => translateFolder(folder, true)}
          onStopBatch={() => { stopBatch(); cancel() }}
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
            mask={cleanedPage ? activeTouchups : activeMask}
            cleaned={cleanedPage}
            stage={stage ?? (active && (!activeMask || !activeTouchups) && !error ? 'loading' : null)}
            error={error}
            selected={selected}
            onSelect={setSelected}
            tool={tool}
            onTool={(next) => {
              setTool(next)
              setSelected(null)
              setShowCleaned(next !== 'boxes' && Boolean(cleanedPage))
            }}
            runningFolder={batch !== null && !batch.finished}
            showCleaned={showCleaned}
            onShowCleaned={setShowCleaned}
            onRunAll={runAll}
            onDetect={runDetect}
            inspecting={{
              languages: source.offered,
              language: source.code,
              onLanguage: (next) => { stopBatch(); cancel(); source.setCode(next) },
              onAddRegion: addRegion,
              onRegionBox: setRegionBox,
              onRegionSettled: rereadRegion,
              onToggleExcluded: toggleExcluded,
            }}
            masking={{
              onClean: runClean,
              letters: letterMaskAt(active?.id, spread),
              onTrace: traceLetters,
              spread,
              onSpread: (next) => { cancel(); if (active) dropLetterMask(active.id); setSpread(next) },
              fill,
              onFill: (next) => { cancel(); setFill(next) },
            }}
            translating={{
              models: llamaCpp.models,
              model: llamaCpp.model,
              onModel: (next) => { stopBatch(); cancel(); llamaCpp.setModel(next) },
              target: llamaCpp.target,
              onTarget: (next) => { stopBatch(); cancel(); llamaCpp.setTarget(next) },
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
                  ? null
                  : pageLettering.some(Boolean) && !cleanedPage
                    ? 'this page has not been cleaned yet'
                    : null),
            }}
          />
        )}

        {!reviewFolder &&
          active &&
          analysis &&
          tool !== 'mask' &&
          (
            <div className="flex shrink-0 flex-col lg:contents">
              <button
                type="button"
                aria-expanded={detailsOpen}
                aria-controls="page-details"
                onClick={() => setDetailsOpen(!detailsOpen)}
                className="border-t border-line bg-surface px-4 py-2 text-left text-xs font-medium text-muted focus-visible:outline-2 focus-visible:outline-accent lg:hidden"
              >
                {detailsOpen ? 'Hide' : 'Show'} {tool === 'text' ? 'translation editor' : 'detected text'}
              </button>
              <div id="page-details" className={detailsOpen ? 'contents' : 'hidden lg:contents'}>
          {tool === 'text' ? (
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
          )}
              </div>
            </div>
          )}
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
