import { useCallback, useEffect, useRef, useState } from 'react'
import { said } from '../lib/api'
import type { GalleryFolder, GalleryImage } from '../lib/images'

export type Phase = {
  name: string
  each: (page: GalleryImage, signal: AbortSignal) => Promise<string | null>
  blocking?: boolean
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
  failed: Failure[]
  stopping: boolean
  finished: boolean
}

export function useBatch() {
  const [run, setRun] = useState<BatchRun | null>(null)

  const current = useRef<{ controller: AbortController; finished: boolean } | null>(null)
  const mounted = useRef(true)

  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      current.current?.controller.abort()
      current.current = null
    }
  }, [])

  const start = useCallback(
    async (folder: GalleryFolder, pages: GalleryImage[], phases: Phase[]) => {
      if (
        !mounted.current ||
        (current.current && !current.current.finished && !current.current.controller.signal.aborted) ||
        pages.length === 0 ||
        phases.length === 0
      ) return
      const job = { controller: new AbortController(), finished: false }
      current.current = job
      const { signal } = job.controller
      const update = (change: (now: BatchRun) => BatchRun) => {
        if (!mounted.current || current.current !== job) return
        setRun((now) => mounted.current && current.current === job && now ? change(now) : now)
      }

      setRun({
        folder: folder.id,
        label: folder.name,
        phase: phases[0].name,
        phaseAt: 0,
        phases: phases.length,
        total: pages.length,
        done: 0,
        page: null,
        failed: [],
        stopping: false,
        finished: false,
      })

      // A page that falls over in a blocking phase cannot be carried by the
      // phases after it: without regions there is nothing to translate or mask.
      const broken = new Set<string>()

      try {
        for (const [at, phase] of phases.entries()) {
          if (signal.aborted) break

          update((now) => ({
            ...now,
            phase: phase.name,
            phaseAt: at,
            total: pages.length,
            done: 0,
            page: null,
          }))

          for (const page of pages) {
            if (signal.aborted) break

            if (broken.has(page.id)) {
              update((now) => ({ ...now, done: now.done + 1 }))
              continue
            }

            update((now) => ({ ...now, page: { id: page.id, name: page.name } }))

            let why: string | null
            try {
              why = await phase.each(page, signal)
            } catch (cause) {
              if (signal.aborted || (cause instanceof Error && cause.name === 'AbortError')) {
                job.controller.abort()
                break
              }
              why = said(cause)
            }
            if (signal.aborted) break

            if (why && phase.blocking) broken.add(page.id)

            update((now) => ({
              ...now,
              done: now.done + 1,
              failed: why
                ? [...now.failed, { id: page.id, name: page.name, why }]
                : now.failed,
            }))
          }
        }
      } finally {
        job.finished = true
        update((now) => ({
          ...now,
          page: null,
          stopping: signal.aborted,
          finished: true,
        }))
      }
    },
    [],
  )

  const stop = useCallback(() => {
    const job = current.current
    if (!mounted.current || !job || job.finished) return
    job.controller.abort()
    setRun((now) => current.current === job && now ? { ...now, stopping: true } : now)
  }, [])

  const dismiss = useCallback(() => {
    if (mounted.current) setRun(null)
  }, [])

  return { run, start, stop, dismiss }
}
