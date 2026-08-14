import { useCallback, useEffect, useRef, useState } from 'react'

/** A different spread is a different tracing, not the same one again. */
const keyFor = (id: string, spread: number) => `${id}@${spread}`

export type LetterMasks = {
  /** The tracing held for this page at this spread, if one has been asked for. */
  at: (id: string | null | undefined, spread: number) => ImageBitmap | null
  keep: (id: string, spread: number, bitmap: ImageBitmap) => void
  drop: (id: string) => void
  clear: () => void
}

/**
 * The lettering traced pixel by pixel, one bitmap per page and spread. They are
 * explicitly `close()`d: decoded pixels the collector will not hurry over.
 */
export function useLetterMasks(): LetterMasks {
  const [held, setHeld] = useState<Record<string, ImageBitmap>>({})

  // Read through a ref, so the cleanup does not re-run on every tracing.
  const now = useRef(held)
  now.current = held

  useEffect(() => {
    return () => {
      for (const bitmap of Object.values(now.current)) bitmap.close()
    }
  }, [])

  const at = useCallback(
    (id: string | null | undefined, spread: number) =>
      id ? (now.current[keyFor(id, spread)] ?? null) : null,
    [],
  )

  // The ref is written before the state so a tracing can be marked into a mask
  // the moment it arrives, rather than a render later.
  const hold = (next: Record<string, ImageBitmap>) => {
    now.current = next
    setHeld(next)
  }

  const keep = useCallback((id: string, spread: number, bitmap: ImageBitmap) => {
    hold({ ...now.current, [keyFor(id, spread)]: bitmap })
  }, [])

  const drop = useCallback((id: string) => {
    const mine = `${id}@`
    // One page may have been traced at several spreads; all of them go.
    for (const [key, bitmap] of Object.entries(now.current)) {
      if (key.startsWith(mine)) bitmap.close()
    }
    hold(
      Object.fromEntries(
        Object.entries(now.current).filter(([key]) => !key.startsWith(mine)),
      ),
    )
  }, [])

  const clear = useCallback(() => {
    for (const bitmap of Object.values(now.current)) bitmap.close()
    hold({})
  }, [])

  return { at, keep, drop, clear }
}
