import { useCallback, useEffect, useRef, useState } from 'react'
import { said } from '../lib/api'
import { loadProject, saveProject } from '../lib/project'
import type { ProjectData } from '../lib/project'

export function useProject(
  snapshot: () => Promise<ProjectData>,
  restore: (data: ProjectData) => Promise<void>,
) {
  const [ready, setReady] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const callbacks = useRef({ snapshot, restore })
  callbacks.current = { snapshot, restore }
  const previous = useRef<typeof snapshot | null>(null)
  const actions = useRef({ changed: () => {}, retry: () => {} })
  const changed = useCallback(() => actions.current.changed(), [])
  const retry = useCallback(() => actions.current.retry(), [])

  useEffect(() => {
    let alive = true
    let restored = false
    let loading = false
    let writing = false
    let revision = 0
    let saved = 0
    let timer: number | undefined
    previous.current = null
    setReady(false)

    const flush = async () => {
      if (!alive || !restored || writing || revision === saved) return
      writing = true
      setSaving(true)
      try {
        while (alive && revision !== saved) {
          const version = revision
          const data = await callbacks.current.snapshot()
          if (!alive) return
          await saveProject(data)
          if (!alive) return
          saved = version
          setError(null)
        }
      } catch (cause) {
        if (alive) setError(`Project not saved: ${said(cause)}. Keep this tab open and retry.`)
      } finally {
        writing = false
        if (alive) setSaving(false)
      }
    }

    const load = async () => {
      if (!alive || loading || restored) return
      loading = true
      setError(null)
      try {
        const data = await loadProject()
        if (!alive) return
        if (data) await callbacks.current.restore(data)
        if (!alive) return
        restored = true
        setReady(true)
      } catch (cause) {
        if (alive) setError(`Project could not be restored: ${said(cause)}. Retry to load it; the saved project will not be overwritten.`)
      } finally {
        loading = false
      }
    }

    actions.current = {
      changed: () => {
        if (!alive || !restored) return
        revision += 1
        setSaving(true)
        clearTimeout(timer)
        timer = window.setTimeout(() => void flush(), 300)
      },
      retry: () => {
        clearTimeout(timer)
        if (restored) void flush()
        else void load()
      },
    }

    const warn = (event: BeforeUnloadEvent) => {
      if (revision === saved) return
      event.preventDefault()
      event.returnValue = ''
    }
    window.addEventListener('beforeunload', warn)
    void load()
    return () => {
      alive = false
      clearTimeout(timer)
      window.removeEventListener('beforeunload', warn)
    }
  }, [])

  useEffect(() => {
    if (!ready) return
    if (previous.current && previous.current !== snapshot) changed()
    previous.current = snapshot
  }, [snapshot, ready, changed])

  return { ready, saving, error, changed, retry }
}
