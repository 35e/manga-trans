import { useCallback, useRef, useState } from 'react'
import { said } from '../lib/api'
import type { GalleryFolder, GalleryImage } from '../lib/images'

export type BatchRun = {
  folder: string
  label: string
  phase: string
  total: number
  done: number
  page: { id: string; name: string } | null
  note: string | null
  failed: { name: string; why: string }[]
  stopping: boolean
  finished: boolean
}

export function useBatch() {
  const [run, setRun] = useState<BatchRun | null>(null)

  const stopping = useRef(false)
  const going = useRef(false)

  const start = useCallback(
    async (
      folder: GalleryFolder,
      pages: GalleryImage[],
      phase: string,
      step: (page: GalleryImage) => Promise<string | null>,
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

  const stop = useCallback(() => {
    stopping.current = true
    setRun((now) => now && { ...now, stopping: true })
  }, [])

  const dismiss = useCallback(() => setRun(null), [])

  return { run, start, stop, dismiss }
}
