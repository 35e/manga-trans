import { useCallback, useEffect, useState } from 'react'
import type { Language } from '../lib/api'
import { LANGUAGE_DEFAULT, languages as listLanguages } from '../lib/api'
import { held, keep } from '../lib/dom'

const LANGUAGE_KEY = 'manga-trans:language'

/**
 * What the pages are lettered in, remembered in this browser because the API
 * keeps no settings. The list is the API's. Until it arrives the code stands on
 * its own and the page is read right to left.
 */
export function useLanguage() {
  const [offered, setOffered] = useState<Language[]>([])
  const [code, remember] = useState(() => held(LANGUAGE_KEY) ?? LANGUAGE_DEFAULT)

  useEffect(() => {
    let dropped = false
    listLanguages().then(
      (found) => {
        if (dropped) return
        setOffered(found)
        // A code kept from an API that read something this one does not would
        // otherwise be sent on every page and refused.
        remember((chosen) =>
          found.some((language) => language.code === chosen)
            ? chosen
            : (found[0]?.code ?? chosen),
        )
      },
      () => undefined,
    )
    return () => {
      dropped = true
    }
  }, [])

  const setCode = useCallback((next: string) => {
    remember(next)
    keep(LANGUAGE_KEY, next)
  }, [])

  const language = offered.find((held) => held.code === code) ?? null
  return { offered, code, setCode, language, rtl: language?.rtl ?? true }
}
