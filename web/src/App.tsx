import { useCallback, useEffect, useRef, useState } from 'react'
import { Board } from './components/Board'
import { Dropzone } from './components/Dropzone'
import { Gallery } from './components/Gallery'
import { RegionsPanel } from './components/RegionsPanel'
import { Settings } from './components/Settings'
import { TranslationsPanel } from './components/TranslationsPanel'
import { IconButton } from './components/ui'
import { useFileDrop } from './hooks/useFileDrop'
import { useImageLibrary } from './hooks/useImageLibrary'
import { useMasks } from './hooks/useMasks'
import { useObjectUrls } from './hooks/useObjectUrls'
import { useSettings } from './hooks/useSettings'
import type { Analysis, BoardMode, Box, Fill, Lettering, Stage } from './lib/api'
import {
  API_BASE,
  UNSURE,
  bubbles,
  clean,
  defaultPrompt,
  detect,
  letterMask,
  models as listModels,
  read,
  translate,
} from './lib/api'
import { compose, save } from './lib/compose'
import { SIZE_MAX, SIZE_MIN, fitSize, originalSize, ready } from './lib/fit'
import type { GalleryImage } from './lib/images'
import { formatBytes, plural } from './lib/images'
import { halves, insertAt, insertionFor, moveAt, movedIndex } from './lib/order'

const said = (cause: unknown) =>
  cause instanceof Error ? cause.message : String(cause)

/**
 * A traced page is held under how far it was grown as well as which page it is:
 * ask for more spread and it is a different tracing, not the same one again.
 */
const traceKey = (id: string, spread: number) => `${id}@${spread}`

/**
 * Where a translation of this block goes: the balloon it was written in when one
 * was found, and the block itself when none was.
 *
 * The block is where the Japanese is, and Japanese runs down the page — a line
 * of a dozen characters is a column forty pixels across. English set in that
 * column wraps to about a letter a line, which is why every line used to have to
 * be dragged out to its balloon before it could be read.
 */
const room = (region: { box: Box; bubble?: Box | null }): Box =>
  region.bubble ?? region.box

/**
 * How large a translation is set in `box`: as large as it will go there, but no
 * larger than the page is lettered, which is worked out from the block the
 * original was in and what it said.
 *
 * A balloon is drawn around the words rather than to them, so the largest type
 * that fits one is much bigger than the type it was drawn around whenever the
 * line is short. Both halves are needed: without the box, a long line overruns
 * its balloon; without the ceiling, a short one fills it.
 */
function sizeFor(text: string, box: Box, original: string, block: Box): number {
  return fitSize(
    text,
    box[2] - box[0],
    box[3] - box[1],
    originalSize(original, block[2] - block[0], block[3] - block[1]),
  )
}

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
  // How far past the ink the traced mask reaches. What is enough depends on the
  // scan, so it is the reader's to raise when edges are being left behind.
  const [spread, setSpread] = useState(4)
  // What goes where the lettering was: the art around it, filled in, or flat
  // white. The art is right for anything drawn over a tone or a line, which is
  // most of a page; white is for when the ground has to be clear.
  const [fill, setFill] = useState<Fill>('art')

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
      // One page may have been traced at several spreads; all of them go.
      for (const [key, bitmap] of Object.entries(lettersHeld.current)) {
        if (key.startsWith(`${id}@`)) bitmap.close()
      }
      setLetters((current) =>
        Object.fromEntries(
          Object.entries(current).filter(([key]) => !key.startsWith(`${id}@`)),
        ),
      )
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
          // Whatever the detector is not sure of starts left alone. It is still
          // read and still listed, so what it says can be seen before deciding;
          // it is only kept out of the cleaning until it is put back.
          excluded: detection.regions.flatMap((region, index) =>
            region.confidence < UNSURE ? [index] : [],
          ),
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
   * A block the detector missed, drawn by hand.
   *
   * It goes in at the point reading order puts it, not on the end, so the page
   * still translates as one conversation in the right sequence. Everything held
   * against a block by its position — what was read, what was left alone, what
   * was lettered — is moved along with it.
   */
  const addRegion = useCallback(
    async (box: Box) => {
      if (!active) return
      const { id, file } = active
      const held = analyses[id]
      if (!held) return

      const at = insertionFor(
        held.detection.regions.map((region) => region.box),
        box,
      )
      const added = { id: crypto.randomUUID(), box, confidence: 1, manual: true }
      const regions = insertAt(held.detection.regions, at, added)
      const texts = insertAt(
        held.texts ?? held.detection.regions.map(() => ''),
        at,
        '',
      )

      setAnalyses((current) => ({
        ...current,
        [id]: {
          ...held,
          detection: { ...held.detection, regions },
          texts,
          // Indices at or past the new one all moved up by one.
          excluded: held.excluded.map((index) => (index >= at ? index + 1 : index)),
        },
      }))
      setLettering((current) =>
        current[id] ? { ...current, [id]: insertAt(current[id], at, null) } : current,
      )
      setSelected(at)

      // Marked for hiding along with the rest, if the rest already are.
      const mask = forPage(active)
      if (mask && !mask.empty) {
        const traced = lettersHeld.current[traceKey(id, spread)]
        if (traced) mask.letters(traced, [box])
        else mask.boxes([box])
      }

      setError(null)
      setWorking({ id, stage: 'reading' })
      try {
        // What it says and what room it was said in, asked for together: a block
        // drawn by hand should end up knowing both, the same as a detected one.
        const [[text], [balloon]] = await Promise.all([
          read(file, [box]),
          bubbles(file, [box]),
        ])
        setAnalyses((current) => {
          const now = current[id]
          if (!now?.texts) return current
          // Found again by name: the list may have been added to or reordered
          // while the reader was working.
          const where = now.detection.regions.findIndex(
            (region) => region.id === added.id,
          )
          if (where === -1) return current
          const said = [...now.texts]
          said[where] = text ?? ''
          const regions = [...now.detection.regions]
          regions[where] = { ...regions[where], bubble: balloon }
          return {
            ...current,
            [id]: {
              ...now,
              detection: { ...now.detection, regions },
              texts: said,
            },
          }
        })
      } catch (cause) {
        setError(said(cause))
      } finally {
        setWorking(null)
      }
    },
    [active, analyses, forPage, spread],
  )

  /**
   * A block's box, while it is being dragged or pulled by an edge.
   *
   * Only the box moves here. What is inside it is read again once the drag is
   * over rather than on every frame of it, so this stays cheap enough to run at
   * the speed of a pointer.
   */
  const setRegionBox = useCallback(
    (index: number, box: Box) => {
      if (!active) return
      const { id } = active
      setAnalyses((current) => {
        const held = current[id]
        const region = held?.detection.regions[index]
        if (!held || !region) return current
        const regions = [...held.detection.regions]
        // The balloon this block was written in was worked out from where the
        // block was. It is dropped rather than dragged along: a block pulled
        // onto its neighbour is in that neighbour's balloon now, and a stale
        // answer would letter the translation into the one it came from.
        // `rereadRegion` asks again once the drag is over.
        regions[index] = { ...region, box, bubble: null }
        return {
          ...current,
          [id]: { ...held, detection: { ...held.detection, regions } },
        }
      })
    },
    [active],
  )

  /**
   * That drag, once it is over: read what the block says now, and keep the mask
   * in step — the rectangle it used to cover comes back out, and where it is
   * now goes in.
   *
   * This is what splitting two bubbles the detector ran together comes down to:
   * pull this one off the second, and what it says is no longer both of them.
   */
  const rereadRegion = useCallback(
    async (index: number, was: Box) => {
      if (!active) return
      const { id, file } = active
      const held = analyses[id]
      const region = held?.detection.regions[index]
      if (!held || !region) return

      const { box } = region
      if (box.join() === was.join()) return

      const mask = forPage(active)
      if (mask && !mask.empty) {
        mask.boxes([was], true)
        if (!held.excluded.includes(index)) {
          const traced = lettersHeld.current[traceKey(id, spread)]
          if (traced) mask.letters(traced, [box])
          else mask.boxes([box])
        }
      }

      // Nothing has been read on this page yet, so there is nothing to bring up
      // to date: detecting will read the lot.
      if (!held.texts) return

      setError(null)
      setWorking({ id, stage: 'reading' })
      try {
        const [[text], [balloon]] = await Promise.all([
          read(file, [box]),
          bubbles(file, [box]),
        ])
        setAnalyses((current) => {
          const now = current[id]
          if (!now?.texts) return current
          const where = now.detection.regions.findIndex(
            (block) => block.id === region.id,
          )
          if (where === -1) return current
          const said = [...now.texts]
          said[where] = text ?? ''
          const regions = [...now.detection.regions]
          regions[where] = { ...regions[where], bubble: balloon }
          return {
            ...current,
            [id]: {
              ...now,
              detection: { ...now.detection, regions },
              texts: said,
            },
          }
        })
      } catch (cause) {
        setError(said(cause))
      } finally {
        setWorking(null)
      }
    },
    [active, analyses, forPage, spread],
  )

  /**
   * A block dragged to a different place in the list.
   *
   * The order is a guess the detector makes — down the page, then right to left
   * across it — and an inset panel or a caption is enough to throw it. It is
   * also the order the page is translated in, one conversation at a time, so it
   * is worth being able to say. Everything held against a block by its position
   * moves along with it.
   */
  const moveRegion = useCallback(
    (from: number, to: number) => {
      if (!active || from === to) return
      const { id } = active

      setAnalyses((current) => {
        const held = current[id]
        if (!held?.detection.regions[from] || !held.detection.regions[to]) {
          return current
        }
        return {
          ...current,
          [id]: {
            ...held,
            detection: {
              ...held.detection,
              regions: moveAt(held.detection.regions, from, to),
            },
            texts: held.texts ? moveAt(held.texts, from, to) : null,
            excluded: held.excluded.map((index) => movedIndex(index, from, to)),
          },
        }
      })
      setLettering((current) =>
        current[id] ? { ...current, [id]: moveAt(current[id], from, to) } : current,
      )
      setSelected((now) => (now === null ? now : movedIndex(now, from, to)))
    },
    [active],
  )

  /**
   * One block that turned out to be two bubbles, cut in two at `at` — a place
   * in the translated line, which is where the join between them shows.
   *
   * The detector runs neighbouring bubbles together often enough to matter, and
   * it is usually noticed with the lettering already set, which is why the cut
   * can be made from here and not only back among the blocks. The line is cut
   * where the cursor was, each half is resized to the box it now sits in, and
   * both boxes are read again so the originals still say what is inside them.
   *
   * The block is what is really being cut: the translations are held one per
   * block, in step with them, and two bubbles were always two blocks.
   */
  const splitRegion = useCallback(
    async (index: number, at: number) => {
      if (!active) return
      const { id, file } = active
      const held = analyses[id]
      const region = held?.detection.regions[index]
      const line = lettering[id]?.[index]
      if (!held || !region || !line) return

      const before = line.text.slice(0, at).trim()
      const rest = line.text.slice(at).trim()
      // Neither half may be empty: that is a cursor at one end, not a cut.
      if (!before || !rest) return

      const [firstBox, secondBox] = halves(region.box, at / line.text.length)
      const added = {
        id: crypto.randomUUID(),
        box: secondBox,
        confidence: region.confidence,
        manual: region.manual,
      }

      setAnalyses((current) => {
        const now = current[id]
        const target = now?.detection.regions[index]
        if (!now || target?.id !== region.id) return current

        const regions = [...now.detection.regions]
        // Two bubbles, so two balloons: whichever one the block as a whole was
        // said to be in was the wrong answer for at least one of the halves.
        regions[index] = { ...target, box: firstBox, bubble: null }
        const texts = now.texts ?? now.detection.regions.map(() => '')

        return {
          ...current,
          [id]: {
            ...now,
            detection: {
              ...now.detection,
              regions: insertAt(regions, index + 1, added),
            },
            // The original stays on the first half until it has been read
            // again, which is a moment away.
            texts: insertAt(texts, index + 1, ''),
            excluded: [
              // Indices at or past the new one all moved up by one, and a block
              // left alone is left alone on both sides of the cut.
              ...now.excluded.map((was) => (was >= index + 1 ? was + 1 : was)),
              ...(now.excluded.includes(index) ? [index + 1] : []),
            ],
          },
        }
      })

      setLettering((current) => {
        const page = current[id]
        const was = page?.[index]
        if (!page || !was) return current

        // Held to the size of the block before the cut, not of the half: the
        // page is lettered at one size, and half a block holds half the
        // characters in half the room, which says nothing new about it.
        const most = originalSize(
          held.texts?.[index] ?? '',
          region.box[2] - region.box[0],
          region.box[3] - region.box[1],
        )
        const put = (text: string, box: Box): Lettering => ({
          ...was,
          text,
          box,
          size: fitSize(text, box[2] - box[0], box[3] - box[1], most),
        })

        const next = [...page]
        next[index] = put(before, firstBox)
        return { ...current, [id]: insertAt(next, index + 1, put(rest, secondBox)) }
      })
      setSelected(index)

      setError(null)
      setWorking({ id, stage: 'reading' })
      try {
        const [readings, balloons] = await Promise.all([
          read(file, [firstBox, secondBox]),
          bubbles(file, [firstBox, secondBox]),
        ])
        setAnalyses((current) => {
          const now = current[id]
          if (!now?.texts) return current
          // Found again by name: the list may have been added to or reordered
          // while the reader was working.
          const first = now.detection.regions.findIndex(
            (block) => block.id === region.id,
          )
          const second = now.detection.regions.findIndex(
            (block) => block.id === added.id,
          )
          if (first === -1 || second === -1) return current
          const texts = [...now.texts]
          texts[first] = readings[0] ?? ''
          texts[second] = readings[1] ?? ''
          const regions = [...now.detection.regions]
          regions[first] = { ...regions[first], bubble: balloons[0] }
          regions[second] = { ...regions[second], bubble: balloons[1] }
          return {
            ...current,
            [id]: { ...now, detection: { ...now.detection, regions }, texts },
          }
        })
      } catch (cause) {
        setError(said(cause))
      } finally {
        setWorking(null)
      }
    },
    [active, analyses, lettering],
  )

  /**
   * The lettering itself, traced pixel by pixel, so a clean can hide the words
   * and leave the art they were drawn over. Another pass of the detector, so it
   * is asked for once per page and then kept.
   */
  const tracePage = useCallback(
    async (page: GalleryImage): Promise<ImageBitmap | null> => {
      const key = traceKey(page.id, spread)
      const held = lettersHeld.current[key]
      if (held) return held

      setError(null)
      setWorking({ id: page.id, stage: 'tracing' })
      try {
        const traced = await createImageBitmap(await letterMask(page.file, spread))
        setLetters((current) => ({ ...current, [key]: traced }))
        return traced
      } catch (cause) {
        setError(said(cause))
        return null
      } finally {
        setWorking(null)
      }
    },
    [spread],
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
        setCleaned(page.id, await clean(page.file, marks, fill))
        return true
      } catch (cause) {
        setError(said(cause))
        return false
      } finally {
        setWorking(null)
      }
    },
    [setCleaned, fill],
  )

  const pageLettering = active ? (lettering[active.id] ?? []) : []

  /**
   * Translate the page: every block that was read and not left alone, sent
   * together. Each line lands in the balloon its original was written in, set at
   * whatever size fits it — not in the box the original came out of, which for
   * vertical Japanese is a column too narrow to set a word of English across.
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
          const block = found.detection.regions[line.index]
          const box = room(block)
          set[line.index] = {
            text,
            box,
            // `line.text` is the original this was translated from, which is
            // what says how large this page is lettered.
            size: sizeFor(text, box, line.text, block.box),
            angle: 0,
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

  /** How far the line is turned, kept to one full turn so it reads plainly. */
  const setLetteringAngle = useCallback(
    (index: number, angle: number) =>
      changeLettering(index, { angle: ((angle % 360) + 360) % 360 }),
    [changeLettering],
  )

  const fitOne = useCallback(
    (index: number) => {
      const held = active ? analyses[active.id] : null
      const line = active ? lettering[active.id]?.[index] : null
      const block = held?.detection.regions[index]
      if (!line || !block) return
      changeLettering(index, {
        size: sizeFor(line.text, line.box, held?.texts?.[index] ?? '', block.box),
      })
    },
    [active, analyses, lettering, changeLettering],
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
    const held = analyses[id]
    setLettering((current) => {
      const page = current[id]
      if (!page) return current
      return {
        ...current,
        [id]: page.map((line, at) => {
          const block = held?.detection.regions[at]
          if (line === null || !block) return line
          return {
            ...line,
            size: sizeFor(line.text, line.box, held?.texts?.[at] ?? '', block.box),
          }
        }),
      }
    })
  }, [active, analyses])

  /**
   * Put every line back in the balloon its block was written in, and resize it
   * to suit.
   *
   * Detecting already answers with those balloons, so this is for the blocks it
   * never saw: one drawn by hand, one cut in two, one pulled off the bubble
   * beside it. It is also the way back after a box has been dragged about by
   * hand and made worse rather than better.
   *
   * A block with no balloon around it — a sound effect over artwork, a caption
   * in the margin — is left exactly where it is. There is nowhere better to put
   * it, and moving it somewhere arbitrary would be worse than leaving it.
   */
  const fitBoxes = useCallback(async () => {
    if (!active) return
    const { id, file } = active
    const found = analyses[id]
    if (!found) return

    setError(null)
    setWorking({ id, stage: 'fitting' })
    try {
      const [balloons] = await Promise.all([
        bubbles(
          file,
          found.detection.regions.map((region) => region.box),
        ),
        // Sizes are about to be worked out by measuring: the face has to be in.
        ready(),
      ])

      // Both lists are held by block position, so an answer only means anything
      // while the page still has the blocks it was asked about.
      setAnalyses((current) => {
        const now = current[id]
        if (now?.detection.regions.length !== balloons.length) return current
        return {
          ...current,
          [id]: {
            ...now,
            detection: {
              ...now.detection,
              regions: now.detection.regions.map((region, at) => ({
                ...region,
                bubble: balloons[at],
              })),
            },
          },
        }
      })
      setLettering((current) => {
        const page = current[id]
        if (page?.length !== balloons.length) return current
        return {
          ...current,
          [id]: page.map((line, at) => {
            const box = balloons[at]
            const block = found.detection.regions[at]
            if (!line || !box || !block) return line
            return {
              ...line,
              box,
              size: sizeFor(line.text, box, found.texts?.[at] ?? '', block.box),
            }
          }),
        }
      })
    } catch (cause) {
      setError(said(cause))
    } finally {
      setWorking(null)
    }
  }, [active, analyses])

  const total = images.reduce((sum, image) => sum + image.size, 0)

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-canvas text-ink">
      <header className="flex shrink-0 items-center justify-between gap-4 border-b border-line bg-surface px-4 py-2.5">
        <h1 className="text-sm font-semibold tracking-tight text-ink">manga-trans</h1>
        <div className="flex min-w-0 items-center gap-2">
          <p className="truncate font-mono text-[11px] text-faint">{API_BASE}</p>
          <IconButton label="Settings" onClick={() => setSettingsOpen(true)}>
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
          models={ollamaModels}
        />
      )}

      <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
        <aside className="flex shrink-0 flex-col border-line bg-surface max-lg:h-64 max-lg:border-b lg:w-60 lg:border-r xl:w-72">
          <div className="shrink-0 p-3">
            <Dropzone onFiles={add} dragging={dragging} busy={busy} />
          </div>

          {notice && (
            <div className="mx-3 mb-3 flex shrink-0 items-start justify-between gap-2 rounded-lg border border-warn/30 bg-warn/10 px-2.5 py-2 text-[11px] leading-snug text-warn">
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
            <div className="flex shrink-0 items-center justify-between gap-2 border-t border-line px-3 py-2 text-[11px] text-faint">
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
          onAddRegion={addRegion}
          onRegionBox={setRegionBox}
          onRegionSettled={rereadRegion}
          onToggleExcluded={toggleExcluded}
          letters={active ? (letters[traceKey(active.id, spread)] ?? null) : null}
          onTrace={traceLetters}
          spread={spread}
          onSpread={setSpread}
          fill={fill}
          onFill={setFill}
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
            onFitBoxes: fitBoxes,
            lettering: pageLettering,
            onBox: setLetteringBox,
            onTurn: setLetteringAngle,
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
            onMove={moveRegion}
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
            onSplit={splitRegion}
          />
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
      className={`shrink-0 rounded-md px-2 py-1 font-medium transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-danger ${
        armed
          ? 'bg-danger text-white'
          : 'text-faint hover:bg-danger/15 hover:text-danger'
      }`}
    >
      {armed ? 'Sure?' : 'Clear all'}
    </button>
  )
}

export default App
