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
 * A whole folder run one page at a time.
 *
 * One at a time and not all at once: the API holds one detector and one reader
 * behind a lock each, so pages sent together queue there anyway — and queued
 * there they arrive in no particular order, which leaves a progress bar with
 * nothing true to say. This way the page in hand is always the page named.
 *
 * `step` hands back why a page fell over, or null when it came out. A run is
 * never taken down by one page: a throw is caught and counted like any refusal.
 * It is an argument rather than the hook's, because a folder is run twice — once
 * to read it and once to letter it — and the two want different steps under one
 * card, one Stop and one thing that is true.
 */
export function useBatch() {
  const [run, setRun] = useState<BatchRun | null>(null)

  // Whether to carry on is read between pages rather than depended on: a request
  // already in the air is left to land. Stopping is therefore "after this one".
  const stopping = useRef(false)
  const going = useRef(false)

  const start = useCallback(
    async (
      folder: GalleryFolder,
      pages: GalleryImage[],
      phase: string,
      step: (page: GalleryImage) => Promise<string | null>,
      /**
       * Work on the whole chapter rather than on any one page, run at the end of
       * this one. Inside the run rather than after it so there is one card and
       * one Stop: reading the chapter is part of reading it, not a second job.
       * It says what it is doing through `say` and gives up the same way a page
       * does.
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
   * Carry on no further. Also how a run is called off by its folder being deleted
   * or the gallery emptied — the page in hand is still let land, so the run winds
   * down the one way rather than being torn out from under itself.
   */
  const stop = useCallback(() => {
    stopping.current = true
    setRun((now) => now && { ...now, stopping: true })
  }, [])

  const dismiss = useCallback(() => setRun(null), [])

  return { run, start, stop, dismiss }
}
