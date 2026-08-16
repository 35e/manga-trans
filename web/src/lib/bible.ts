import type { Bible, Story, Term } from './api'
import { merged } from './story'

export const SURVEY_PAGES = 8

const TERM_LIMIT = 40

export const UNREAD: Bible = {
  synopsis: '',
  register: '',
  beats: [],
  cast: [],
  terms: [],
}

export function isUnread(bible: Bible | null | undefined): boolean {
  return !bible || (bible.synopsis === '' && bible.beats.length === 0)
}

export function folded(held: Bible | undefined, said: Bible, first: number): Bible {
  const was = held ?? UNREAD

  const beats = [...was.beats]
  while (beats.length < first) beats.push('')
  said.beats.forEach((beat, at) => {
    beats[first + at] = beat
  })

  const cast = merged({ scene: '', cast: was.cast }, { scene: '', cast: said.cast }).cast

  const terms: Term[] = [...was.terms]
  for (const term of said.terms) {
    const at = terms.findIndex((held) => held.source === term.source)
    if (at !== -1) terms[at] = term
    else if (terms.length < TERM_LIMIT) terms.push(term)
  }

  return {
    synopsis: said.synopsis.trim() === '' ? was.synopsis : said.synopsis,
    register: said.register.trim() === '' ? was.register : said.register,
    beats,
    cast,
    terms,
  }
}

export function asStory(bible: Bible): Story {
  return { scene: '', cast: bible.cast }
}

export function withText(
  bible: Bible | undefined,
  field: 'synopsis' | 'register',
  value: string,
): Bible {
  return { ...(bible ?? UNREAD), [field]: value.trim() }
}

export function fits(bible: Bible | null | undefined, pages: number): boolean {
  return !!bible && bible.beats.length === pages
}
