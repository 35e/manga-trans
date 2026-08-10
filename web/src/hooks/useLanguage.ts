import { useCallback, useEffect, useState } from 'react'
import type { Language } from '../lib/api'
import { LANGUAGE_DEFAULT, languages as listLanguages } from '../lib/api'
import { held, keep } from '../lib/dom'

const LANGUAGE_KEY = 'manga-trans:language'

/**
 * What the pages are lettered in — which decides who reads them, which way round
 * the blocks are put, and what the translator is told the page is in.
 *
 * Remembered in this browser, because the API keeps no settings and because it
 * is a property of what is being worked on rather than of a page: a chapter is
 * all one language, and picking it once a chapter is once too often already.
 *
 * The list is the API's rather than held here: which reader exists for what is
 * its business. Until it arrives — or if it never does — the code stands on its
 * own and the page is read right to left, which is what the API does with a
 * request that names no language at all.
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
        // A code kept from a version of the API that read something this one
        // does not would otherwise be sent on every page and refused.
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
