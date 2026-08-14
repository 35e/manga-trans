import { useCallback, useRef, useState } from 'react'
import type { Term } from '../lib/api'

/** How many terms a chapter carries. Past this it is vocabulary, not its cast. */
const LIMIT = 40

/** What each chapter has settled on, per folder. A loose page has none. */
export function useGlossary() {
  const [terms, setTerms] = useState<Record<string, Term[]>>({})

  // Read through a ref, so translating a page does not depend on the record.
  const now = useRef(terms)
  now.current = terms

  /**
   * Fold what a page named into its folder's list. **The first rendering of a
   * term wins and is never overwritten** — consistency is the whole point, and a
   * later answer is not better for being later.
   */
  const learn = useCallback((folder: string, found: Term[]) => {
    if (found.length === 0) return
    setTerms((current) => {
      const held = current[folder] ?? []
      if (held.length >= LIMIT) return current
      const seen = new Set(held.map((term) => term.source))
      const fresh = found.filter((term) => {
        if (seen.has(term.source)) return false
        seen.add(term.source)
        return true
      })
      if (fresh.length === 0) return current
      return { ...current, [folder]: held.concat(fresh).slice(0, LIMIT) }
    })
  }, [])

  /**
   * A term's wording put right by hand. No `settled` mark and none needed: a
   * term already held is never overwritten.
   */
  const correct = useCallback((folder: string, source: string, target: string) => {
    setTerms((current) => {
      const held = current[folder]
      if (!held) return current
      return {
        ...current,
        [folder]: held.map((term) =>
          term.source === source ? { ...term, target: target.trim() } : term,
        ),
      }
    })
  }, [])

  const forget = useCallback((folder: string) => {
    setTerms((current) => {
      if (!(folder in current)) return current
      const next = { ...current }
      delete next[folder]
      return next
    })
  }, [])

  const clear = useCallback(() => setTerms({}), [])

  return { terms, now, learn, correct, forget, clear }
}
