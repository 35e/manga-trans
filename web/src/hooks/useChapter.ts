import { useCallback, useRef, useState } from 'react'
import type { Bible } from '../lib/api'
import { folded, withText } from '../lib/bible'

/**
 * What each chapter turned out to be, read whole before any of it was
 * translated. Sent over with every page of the folder, so page three is
 * translated knowing what page forty reveals — which is the one thing the
 * running story and glossary cannot do, being built forwards.
 *
 * Per folder, because that is what a chapter is here. A loose page has none: a
 * page on its own is not a chapter and there is nothing to survey.
 *
 * What is *held* here and what is *decided* are apart, the same as `useStory`:
 * every rule about folding a window in lives in `lib/bible.ts`.
 */
export function useChapter() {
  const [bibles, setBibles] = useState<Record<string, Bible>>({})

  // Read as a page is translated, from a callback that deliberately does not
  // depend on the record — the same reason the analyses are read through a ref.
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
