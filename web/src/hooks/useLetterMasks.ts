import { useCallback, useEffect, useRef, useState } from 'react'

const keyFor = (id: string, spread: number) => `${id}@${spread}`

export type LetterMasks = {
  at: (id: string | null | undefined, spread: number) => ImageBitmap | null
  keep: (id: string, spread: number, bitmap: ImageBitmap) => void
  drop: (id: string) => void
  clear: () => void
}

export function useLetterMasks(): LetterMasks {
  const [held, setHeld] = useState<Record<string, ImageBitmap>>({})

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

  const hold = (next: Record<string, ImageBitmap>) => {
    now.current = next
    setHeld(next)
  }

  const keep = useCallback((id: string, spread: number, bitmap: ImageBitmap) => {
    hold({ ...now.current, [keyFor(id, spread)]: bitmap })
  }, [])

  const drop = useCallback((id: string) => {
    const mine = `${id}@`
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
