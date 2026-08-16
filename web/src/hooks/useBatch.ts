import { useCallback, useRef, useState } from 'react'
import { said } from '../lib/api'
import type { GalleryFolder, GalleryImage } from '../lib/images'

export type Phase =
  | {
      name: string
      each: (page: GalleryImage) => Promise<string | null>
      blocking?: boolean
    }
  | {
      name: string
      whole: (say: (note: string | null) => void) => Promise<string | null>
    }

export type Failure = { id: string; name: string; why: string }

export type BatchRun = {
  folder: string
  label: string
  phase: string
  phaseAt: number
  phases: number
  total: number
  done: number
  page: { id: string; name: string } | null
  note: string | null
  failed: Failure[]
  stopping: boolean
  finished: boolean
}

export function useBatch() {
  const [run, setRun] = useState<BatchRun | null>(null)

  const stopping = useRef(false)
  const going = useRef(false)

  const start = useCallback(
    async (folder: GalleryFolder, pages: GalleryImage[], phases: Phase[]) => {
      if (going.current || pages.length === 0 || phases.length === 0) return
      going.current = true
      stopping.current = false

      const first = phases[0]
      setRun({
        folder: folder.id,
        label: folder.name,
        phase: first.name,
        phaseAt: 0,
        phases: phases.length,
        total: 'each' in first ? pages.length : 0,
        done: 0,
        page: null,
        note: null,
        failed: [],
        stopping: false,
        finished: false,
      })

      // A page that falls over in a blocking phase cannot be carried by the
      // phases after it: without regions there is nothing to translate or mask.
      const broken = new Set<string>()

      try {
        for (const [at, phase] of phases.entries()) {
          if (stopping.current) break

          setRun(
            (now) =>
              now && {
                ...now,
                phase: phase.name,
                phaseAt: at,
                total: 'each' in phase ? pages.length : 0,
                done: 0,
                page: null,
                note: null,
              },
          )

          if ('whole' in phase) {
            const say = (note: string | null) => setRun((now) => now && { ...now, note })
            let why: string | null
            try {
              why = await phase.whole(say)
            } catch (cause) {
              why = said(cause)
            }
            setRun(
              (now) =>
                now && {
                  ...now,
                  note: null,
                  failed: why
                    ? [...now.failed, { id: folder.id, name: folder.name, why }]
                    : now.failed,
                },
            )
            continue
          }

          for (const page of pages) {
            if (stopping.current) break

            if (broken.has(page.id)) {
              setRun((now) => now && { ...now, done: now.done + 1 })
              continue
            }

            setRun((now) => now && { ...now, page: { id: page.id, name: page.name } })

            let why: string | null
            try {
              why = await phase.each(page)
            } catch (cause) {
              why = said(cause)
            }

            if (why && phase.blocking) broken.add(page.id)

            setRun(
              (now) =>
                now && {
                  ...now,
                  done: now.done + 1,
                  failed: why
                    ? [...now.failed, { id: page.id, name: page.name, why }]
                    : now.failed,
                },
            )
          }
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
