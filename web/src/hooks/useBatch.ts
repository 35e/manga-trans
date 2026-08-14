import { useCallback, useRef, useState } from 'react'
import { said } from '../lib/api'
import type { GalleryFolder, GalleryImage } from '../lib/images'

/** How far through a folder a run has got, which is what the bar shows. */
export type BatchRun = {
  /** The folder being worked through, so deleting it can call the run off. */
  folder: string
  label: string
  /** Which of the two runs this is, since a folder is now run twice. */
  phase: string
  total: number
  /** Pages finished, whether they came out or fell over. */
  done: number
  /** The page in hand, named so the bar can say which and link to it. */
  page: { id: string; name: string } | null
  /** What is being done that is not a page — the chapter being read whole. */
  note: string | null
  failed: { name: string; why: string }[]
  /** Asked to stop: still true once it has, which is how the bar knows. */
  stopping: boolean
  finished: boolean
}

/**
 * A whole folder run one page at a time, so the page in hand is the page named.
 *
 * `step` hands back why a page fell over, or null when it came out; a run is
 * never taken down by one page. It is an argument rather than the hook's,
 * because a folder is run twice and the two want one card and one Stop.
 */
export function useBatch() {
  const [run, setRun] = useState<BatchRun | null>(null)

  // Read between pages, not depended on: a request already in the air is left
  // to land, so stopping means "after this one".
  const stopping = useRef(false)
  const going = useRef(false)

  const start = useCallback(
    async (
      folder: GalleryFolder,
      pages: GalleryImage[],
      phase: string,
      step: (page: GalleryImage) => Promise<string | null>,
      /**
       * Work on the whole chapter rather than any one page, run at the end of
       * this one and inside it, so there is one card and one Stop.
       */
      after?: (say: (note: string | null) => void) => Promise<string | null>,
    ) => {
      if (going.current || pages.length === 0) return
      going.current = true
      stopping.current = false
      setRun({
        folder: folder.id,
        label: folder.name,
        phase,
        total: pages.length,
        done: 0,
        page: null,
        note: null,
        failed: [],
        stopping: false,
        finished: false,
      })

      try {
        for (const page of pages) {
          if (stopping.current) break
          setRun((now) => now && { ...now, page: { id: page.id, name: page.name } })

          let why: string | null
          try {
            why = await step(page)
          } catch (cause) {
            why = said(cause)
          }

          setRun(
            (now) =>
              now && {
                ...now,
                done: now.done + 1,
                failed: why ? [...now.failed, { name: page.name, why }] : now.failed,
              },
          )
        }

        if (!stopping.current && after) {
          setRun((now) => now && { ...now, page: null })
          const say = (note: string | null) =>
            setRun((now) => now && { ...now, note })
          let why: string | null
          try {
            why = await after(say)
          } catch (cause) {
            why = said(cause)
          }
          setRun(
            (now) =>
              now && {
                ...now,
                note: null,
                failed: why ? [...now.failed, { name: folder.name, why }] : now.failed,
              },
          )
        }
      } finally {
        setRun((now) => now && { ...now, page: null, note: null, finished: true })
        going.current = false
      }
    },
    [],
  )

  /**
   * Carry on no further, and how a deleted folder calls its run off. The page in
   * hand is still let land.
   */
  const stop = useCallback(() => {
    stopping.current = true
    setRun((now) => now && { ...now, stopping: true })
  }, [])

  const dismiss = useCallback(() => setRun(null), [])

  return { run, start, stop, dismiss }
}
