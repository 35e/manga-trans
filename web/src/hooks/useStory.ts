import { useCallback, useRef, useState } from 'react'
import type { Fact, Story } from '../lib/api'
import { merged, withFact } from '../lib/story'

export function useStory() {
  const [stories, setStories] = useState<Record<string, Story>>({})

  const now = useRef(stories)
  now.current = stories

  const learn = useCallback((folder: string, said: Story) => {
    setStories((current) => ({ ...current, [folder]: merged(current[folder], said) }))
  }, [])

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
