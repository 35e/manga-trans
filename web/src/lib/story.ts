import type { CastMember, Fact, Gender, Story } from './api'

const CAST_LIMIT = 12

export const NOBODY: Story = { scene: '', cast: [] }

export function isEmpty(story: Story | null | undefined): boolean {
  return !story || (story.scene === '' && story.cast.length === 0)
}

function settledIn(person: CastMember): Fact[] {
  return person.settled ?? []
}

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
