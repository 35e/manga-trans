import { useCallback, useRef, useState } from 'react'
import type { Bible } from '../lib/api'
import { folded, withText } from '../lib/bible'

export function useChapter() {
  const [bibles, setBibles] = useState<Record<string, Bible>>({})

  const now = useRef(bibles)
  now.current = bibles

  const learn = useCallback((folder: string, said: Bible, first: number) => {
    setBibles((current) => ({
      ...current,
      [folder]: folded(current[folder], said, first),
    }))
  }, [])

  const correct = useCallback(
    (folder: string, field: 'synopsis' | 'register', value: string) => {
      setBibles((current) => ({
        ...current,
        [folder]: withText(current[folder], field, value),
      }))
    },
    [],
  )

  const forget = useCallback((folder: string) => {
    setBibles((current) => {
      if (!(folder in current)) return current
      const next = { ...current }
      delete next[folder]
      return next
    })
  }, [])

  const clear = useCallback(() => setBibles({}), [])

  return { bibles, now, learn, correct, forget, clear }
}
