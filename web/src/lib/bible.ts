import type { Bible, Story, Term } from './api'
import { merged } from './story'

/**
 * How many pages go over in one window. A chapter that will not survey wants
 * this lowered rather than the API's window raised.
 */
export const SURVEY_PAGES = 8

/** As many terms as a chapter carries — `useGlossary`'s own cap. */
const TERM_LIMIT = 40

export const UNREAD: Bible = {
  synopsis: '',
  register: '',
  beats: [],
  cast: [],
  terms: [],
}

/** Nothing has been read yet, so there is nothing to translate against. */
export function isUnread(bible: Bible | null | undefined): boolean {
  return !bible || (bible.synopsis === '' && bible.beats.length === 0)
}

/**
 * The chapter with one window's answer folded in, its pages starting at `first`.
 * Last wins, the opposite way round from `useGlossary` — nothing is translated
 * yet, so a window that has read further genuinely knows better.
 */
export function folded(held: Bible | undefined, said: Bible, first: number): Bible {
  const was = held ?? UNREAD

  const beats = [...was.beats]
  while (beats.length < first) beats.push('')
  said.beats.forEach((beat, at) => {
    beats[first + at] = beat
  })

  // Through the story's rules, so `unknown` never overwrites something known.
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

/** The bible's cast as a story, to seed the chapter with once the sweep is over. */
export function asStory(bible: Bible): Story {
  return { scene: '', cast: bible.cast }
}

/** A synopsis or a register put right by hand. */
export function withText(
  bible: Bible | undefined,
  field: 'synopsis' | 'register',
  value: string,
): Bible {
  return { ...(bible ?? UNREAD), [field]: value.trim() }
}

/**
 * Whether a bible still describes this folder. A page added or deleted since the
 * sweep leaves every beat after it describing the wrong one, silently.
 */
export function fits(bible: Bible | null | undefined, pages: number): boolean {
  return !!bible && bible.beats.length === pages
}
