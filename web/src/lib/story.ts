/**
 * What a chapter has worked out about itself, as pure transforms on one `Story`.
 *
 * A chapter tells you who someone is when it is ready to, which is often several
 * pages after it first needs the answer. So the story is built up rather than
 * asserted: a page that has not seen something says `unknown`, a later page fills
 * it in, and a page that shows the earlier one was wrong corrects it.
 *
 * Which is what the rules below are for. They live here rather than in
 * `useStory` for the same reason `lib/regions.ts` holds the block edits: this is
 * the whole of what decides whether a chapter's facts are right, and it is worth
 * being able to read it in one place.
 */

import type { CastMember, Fact, Gender, Story } from './api'

/** As many people as the API carries — `ollama.CAST_LIMIT`. */
const CAST_LIMIT = 12

export const NOBODY: Story = { scene: '', cast: [] }

/** Nothing has been worked out yet — a page that settled nothing leaves this. */
export function isEmpty(story: Story | null | undefined): boolean {
  return !story || (story.scene === '' && story.cast.length === 0)
}

function settledIn(person: CastMember): Fact[] {
  return person.settled ?? []
}

/**
 * One fact, merged.
 *
 * `unknown` never overwrites something known, and an empty note never clears one:
 * a page that did not show a character must not undo the page that did, and most
 * pages show most characters not at all. Anything else known replaces what was
 * held — that is a page saying it saw something, which is the whole point of
 * asking again. A settled fact is not the model's to move either way.
 */
function keptFact<T>(held: T, said: T, settled: boolean, blank: T): T {
  if (settled) return held
  return said === blank ? held : said
}

function joined(held: CastMember, said: CastMember): CastMember {
  const settled = settledIn(held)
  return {
    ...held,
    gender: keptFact<Gender>(
      held.gender,
      said.gender,
      settled.includes('gender'),
      'unknown',
    ),
    note: keptFact(held.note ?? '', said.note ?? '', settled.includes('note'), ''),
  }
}

/**
 * The story as this page leaves it.
 *
 * The scene is replaced where the page wrote one — it is a description of now,
 * and now has moved. The cast is merged person by person, since a page speaks
 * about whoever is on it and says nothing either way about the rest.
 */
export function merged(held: Story | undefined, said: Story): Story {
  const was = held ?? NOBODY
  const cast = was.cast.map((person) => {
    const now = said.cast.find((other) => other.name === person.name)
    return now ? joined(person, now) : person
  })

  const known = new Set(cast.map((person) => person.name))
  for (const person of said.cast) {
    if (cast.length >= CAST_LIMIT) break
    if (person.name.trim() === '' || known.has(person.name)) continue
    known.add(person.name)
    cast.push(person)
  }

  return { scene: said.scene.trim() === '' ? was.scene : said.scene, cast }
}

/**
 * A fact set by hand, which settles it: the model is told so on every page after
 * this, and {@link merged} will not move it.
 *
 * Clearing one unsettles it and hands it back — an empty value is "you work it
 * out" rather than "the answer is nothing". Someone not in the cast yet is added,
 * which is how a chapter is told about a character before it has read the page
 * that introduces them.
 */
export function withFact(
  story: Story | undefined,
  name: string,
  fact: Fact,
  value: string,
): Story {
  const was = story ?? NOBODY
  const cleared = value.trim() === '' || (fact === 'gender' && value === 'unknown')

  const change = (person: CastMember): CastMember => {
    const settled = settledIn(person).filter((held) => held !== fact)
    return {
      ...person,
      [fact]: fact === 'gender' ? ((value || 'unknown') as Gender) : value,
      settled: cleared ? settled : [...settled, fact],
    }
  }

  if (was.cast.some((person) => person.name === name)) {
    return {
      ...was,
      cast: was.cast.map((person) => (person.name === name ? change(person) : person)),
    }
  }
  if (was.cast.length >= CAST_LIMIT || name.trim() === '') return was
  return {
    ...was,
    cast: [...was.cast, change({ name, gender: 'unknown' })],
  }
}
