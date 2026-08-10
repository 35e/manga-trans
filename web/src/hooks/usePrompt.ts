import { useCallback, useEffect, useState } from 'react'
import { defaultPrompt } from '../lib/api'
import { held, keep } from '../lib/dom'

const PROMPT_KEY = 'manga-trans:prompt'

/**
 * What the model is told, remembered in this browser because the API keeps no
 * settings. `prompt` is null for "whatever the API's own is", so a change there
 * is picked up rather than frozen into a copy; `builtIn` is that own prompt.
 */
export function usePrompt() {
  const [prompt, remember] = useState<string | null>(() => held(PROMPT_KEY))
  const [builtIn, setBuiltIn] = useState<string | null>(null)

  useEffect(() => {
    let dropped = false
    defaultPrompt().then(
      (found) => {
        if (!dropped) setBuiltIn(found)
      },
      () => undefined,
    )
    return () => {
      dropped = true
    }
  }, [])

  const setPrompt = useCallback((next: string | null) => {
    remember(next)
    keep(PROMPT_KEY, next)
  }, [])

  return { prompt, setPrompt, builtIn }
}
