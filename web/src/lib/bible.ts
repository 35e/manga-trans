/**
 * What a chapter turns out to be, as pure transforms on one `Bible` — here
 * rather than in `useChapter` for the reason `lib/story.ts` is here: this is the
 * whole of what decides whether a chapter's own account of itself is right.
 *
 * The terms are merged the opposite way round from `useGlossary`, and that is
 * deliberate. Nothing has been translated when a survey runs, so there is no page
 * to stay consistent with, and a window that has read further genuinely knows
 * better — so the last answer wins. The moment the sweep is over the finished
 * list seeds the glossary, and from there the first rendering is the one that
 * holds for the rest of the chapter.
 */

import type { Bible, Story, Term } from './api'
import { merged } from './story'

/**
 * How many pages go over in one window. Eight pages of manga is a couple of
 * thousand characters of Japanese, comfortably inside the window the API asks
 * for with the running chapter and the answer beside it. A chapter that will not
 * survey wants this lowered rather than the API's window raised.
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
 *
 * The synopsis and the register are replaced where the window wrote one and left
 * alone where it did not — an empty answer is a window that saw nothing to say,
 * not news that the chapter is about nothing. The beats are laid down where their
 * pages are, growing the list with blanks for pages no window has reached, so a
 * beat is always at the index of the page it describes.
 */
export function folded(held: Bible | undefined, said: Bible, first: number): Bible {
  const was = held ?? UNREAD

  const beats = [...was.beats]
  while (beats.length < first) beats.push('')
  said.beats.forEach((beat, at) => {
    beats[first + at] = beat
  })

  // Through the story's rules, so `unknown` never overwrites something known and
  // a fact settled by hand is not moved by a later window either.
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

/**
 * The bible's cast as a story, to seed the chapter with once the sweep is over.
 * No scene: what is going on is a position in the chapter, and a survey has read
 * all of it rather than stopped anywhere.
 */
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
 * Whether a bible still describes this folder.
 *
 * The beats are indexed by a page's place in the folder, so a page added or
 * deleted since the sweep leaves every beat after it describing the wrong page —
 * silently, since nothing downstream can tell. Translating against that is worse
 * than translating against nothing.
 */
export function fits(bible: Bible | null | undefined, pages: number): boolean {
  return !!bible && bible.beats.length === pages
}
