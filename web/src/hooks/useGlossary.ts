import { useCallback, useRef, useState } from 'react'
import type { Term } from '../lib/api'

const LIMIT = 40

export function useGlossary() {
  const [terms, setTerms] = useState<Record<string, Term[]>>({})

  const now = useRef(terms)
  now.current = terms

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
