import { useCallback, useEffect, useRef, useState } from 'react'

export function useObjectUrls() {
  const [state, setState] = useState<{ urls: Record<string, string>; blobs: Record<string, Blob> }>({ urls: {}, blobs: {} })
  const latest = useRef(state)
  const mounted = useRef(true)

  const restore = useCallback((blobs: Record<string, Blob>) => {
    if (!mounted.current) return
    const urls: Record<string, string> = {}
    try {
      for (const [key, blob] of Object.entries(blobs)) urls[key] = URL.createObjectURL(blob)
    } catch (error) {
      for (const url of Object.values(urls)) URL.revokeObjectURL(url)
      throw error
    }
    for (const url of Object.values(latest.current.urls)) URL.revokeObjectURL(url)
    latest.current = { urls, blobs: { ...blobs } }
    setState(latest.current)
  }, [])

  useEffect(() => {
    mounted.current = true
    if (Object.keys(latest.current.blobs).length > 0) restore(latest.current.blobs)
    return () => {
      mounted.current = false
      for (const url of Object.values(latest.current.urls)) URL.revokeObjectURL(url)
    }
  }, [restore])

  const set = useCallback((key: string, blob: Blob) => {
    if (!mounted.current) return
    const url = URL.createObjectURL(blob)
    const previous = latest.current.urls[key]
    latest.current = {
      urls: { ...latest.current.urls, [key]: url },
      blobs: { ...latest.current.blobs, [key]: blob },
    }
    setState(latest.current)
    if (previous) URL.revokeObjectURL(previous)
  }, [])

  const drop = useCallback((key: string) => {
    const going = latest.current.urls[key]
    if (!going) return
    const urls = { ...latest.current.urls }
    const blobs = { ...latest.current.blobs }
    delete urls[key]
    delete blobs[key]
    latest.current = { urls, blobs }
    if (mounted.current) setState(latest.current)
    URL.revokeObjectURL(going)
  }, [])

  const clear = useCallback(() => restore({}), [restore])

  return { ...state, set, drop, clear, restore }
}
