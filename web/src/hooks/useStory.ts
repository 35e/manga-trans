import { useCallback, useRef, useState } from 'react'
import type { Fact, Story } from '../lib/api'
import { merged, withFact } from '../lib/story'

/**
 * Where each chapter has got to: what is going on, and who it is going on
 * between. Sent over with the next page of the folder so a page opening on two
 * people mid-argument is translated as that rather than as strangers meeting, and
 * so a character keeps one gender across pages no model sees together.
 *
 * Per folder, because that is what a chapter is here — a loose page has none and
 * contributes none.
 *
 * What is *held* here and what is *decided* are deliberately apart: every rule
 * about which of two answers wins lives in `lib/story.ts`. All this does is keep
 * one story per folder and hand the two of them to it.
 */
export function useStory() {
  const [stories, setStories] = useState<Record<string, Story>>({})

  // Read as a page is translated, from a callback that deliberately does not
  // depend on the record — the same reason the analyses are read through a ref.
  const now = useRef(stories)
  now.current = stories

  const learn = useCallback((folder: string, said: Story) => {
    setStories((current) => ({ ...current, [folder]: merged(current[folder], said) }))
  }, [])

  /** A fact set by hand, which the model is then told is not its to change. */
  const correct = useCallback(
    (folder: string, name: string, fact: Fact, value: string) => {
      setStories((current) => ({
        ...current,
        [folder]: withFact(current[folder], name, fact, value),
      }))
    },
    [],
  )

  const forget = useCallback((folder: string) => {
    setStories((current) => {
      if (!(folder in current)) return current
      const next = { ...current }
      delete next[folder]
      return next
    })
  }, [])

  const clear = useCallback(() => setStories({}), [])

  return { stories, now, learn, correct, forget, clear }
}
