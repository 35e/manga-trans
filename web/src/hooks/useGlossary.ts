import { useCallback, useRef, useState } from 'react'
import type { Term } from '../lib/api'

/** How many terms a chapter carries. Past this it is vocabulary, not its cast. */
const LIMIT = 40

/**
 * What each chapter has settled on: the names, places and coinages a folder's
 * pages have already been translated with, sent back over with the next page so
 * a character keeps one spelling across a chapter no model ever sees at once.
 *
 * Per folder, because that is what a chapter is here. A loose page has none, and
 * contributes none: there is nothing for it to be consistent with.
 */
export function useGlossary() {
  const [terms, setTerms] = useState<Record<string, Term[]>>({})

  // Read as a page is translated, from a callback that deliberately does not
  // depend on the record — the same reason the analyses are read through a ref.
  const now = useRef(terms)
  now.current = terms

  /**
   * Fold what a page named into its folder's list.
   *
   * The first rendering of a term wins and is never overwritten: consistency is
   * the whole point, and the second answer is not better for being later. Past
   * the cap new terms are dropped rather than rotating old ones out — the ones
   * found first are the recurring cast.
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
   * A term's wording put right by hand.
   *
   * No `settled` mark and none needed: a term already held is never overwritten,
   * so a hand-set one is immovable by the same rule that makes the first
   * rendering win. Worth most in the gap between reading a chapter and
   * translating it, where a name put right is put right on every page rather
   * than on the pages after the one that noticed.
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
