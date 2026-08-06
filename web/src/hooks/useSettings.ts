import { useCallback, useState } from 'react'

const PROMPT_KEY = 'manga-trans:prompt'

function held(key: string): string | null {
  try {
    return window.localStorage.getItem(key)
  } catch {
    // A browser that will not remember anything is still a browser to work in.
    return null
  }
}

function keep(key: string, value: string | null) {
  try {
    if (value === null) window.localStorage.removeItem(key)
    else window.localStorage.setItem(key, value)
  } catch {
    /* nothing to be done, and nothing worth stopping for */
  }
}

/**
 * What this browser remembers between visits.
 *
 * The API keeps nothing — every request carries what it needs — so anything
 * meant to outlast a reload is kept here and sent along.
 */
export function useSettings() {
  // null means "whatever the API's own is", so a change there is picked up
  // rather than frozen into a copy made once.
  const [prompt, remember] = useState<string | null>(() => held(PROMPT_KEY))

  const setPrompt = useCallback((next: string | null) => {
    remember(next)
    keep(PROMPT_KEY, next)
  }, [])

  return { prompt, setPrompt }
}
