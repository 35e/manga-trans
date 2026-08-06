import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * Blobs held under a key and handed out as URLs, each released the moment it is
 * replaced, dropped or the page goes away. Nothing here happens inside a state
 * updater: those run twice under StrictMode, and a URL made twice is a URL
 * leaked once.
 */
export function useObjectUrls() {
  const [urls, setUrls] = useState<Record<string, string>>({})

  const latest = useRef(urls)
  latest.current = urls

  useEffect(() => {
    return () => {
      for (const url of Object.values(latest.current)) URL.revokeObjectURL(url)
    }
  }, [])

  const set = useCallback((key: string, blob: Blob) => {
    const url = URL.createObjectURL(blob)
    const previous = latest.current[key]
    setUrls((current) => ({ ...current, [key]: url }))
    if (previous) URL.revokeObjectURL(previous)
  }, [])

  const drop = useCallback((key: string) => {
    const going = latest.current[key]
    if (!going) return
    setUrls((current) => {
      const rest = { ...current }
      delete rest[key]
      return rest
    })
    URL.revokeObjectURL(going)
  }, [])

  const clear = useCallback(() => {
    const going = Object.values(latest.current)
    setUrls({})
    for (const url of going) URL.revokeObjectURL(url)
  }, [])

  return { urls, set, drop, clear }
}
