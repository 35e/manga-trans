import { useCallback, useRef, useState } from 'react'
import type { Bible } from '../lib/api'
import { folded, withText } from '../lib/bible'

/**
 * What each chapter turned out to be, read whole before any of it was
 * translated. Every folding rule lives in `lib/bible.ts`, as `useStory` does.
 */
export function useChapter() {
  const [bibles, setBibles] = useState<Record<string, Bible>>({})

  // Read through a ref, so translating a page does not depend on the record.
  const now = useRef(bibles)
  now.current = bibles

  /** One window of the survey folded in, its pages starting at `first`. */
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
