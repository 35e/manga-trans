import type { Analysis } from './api'
import { UNSURE } from './api'
import type { Lines } from './lettering'

export type Trouble = 'failed' | 'unsure' | 'empty'

export type PageReview = {
  lines: string[]
  troubles: Trouble[]
}

export const SAYS: Record<Trouble, string> = {
  failed: 'fell over',
  unsure: 'unsure block',
  empty: 'nothing came back',
}

/** What a finished page has to show for itself, and what is wrong with it. */
export function reviewed(
  analysis: Analysis | undefined,
  lettering: Lines | undefined,
  why: string | undefined,
): PageReview {
  const set = lettering ?? []
  const lines = set
    .map((line) => line?.text.trim() ?? '')
    .filter((text) => text !== '')

  const troubles: Trouble[] = []
  if (why) troubles.push('failed')

  const regions = analysis?.detection.regions ?? []
  if (regions.some((region) => region.confidence < UNSURE)) troubles.push('unsure')

  // A block that had words in it and came back with none is a hole in the page,
  // not an empty balloon. Blocks left out of the clean are meant to be silent.
  const skip = new Set(analysis?.excluded ?? [])
  const missed = (analysis?.texts ?? []).some(
    (text, at) => text.trim() !== '' && !skip.has(at) && !set[at]?.text.trim(),
  )
  if (missed) troubles.push('empty')

  return { lines, troubles }
}
