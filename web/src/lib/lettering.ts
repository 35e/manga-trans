/**
 * The translated lines on a page, held one per block and in step with
 * `analysis.detection.regions` — see `lib/regions.ts`.
 */

import type { Analysis, Box, Lettering, Region } from './api'
import { fitSize, originalSize, roomInCharacters } from './fit'
import { insertAt, moveAt } from './order'

export type Lines = (Lettering | null)[]

/**
 * Where a translation of this block goes: the balloon it was written in, and the
 * block itself when none was found.
 */
export function roomFor(region: Pick<Region, 'box' | 'bubble'>): Box {
  return region.bubble ?? region.box
}

/**
 * How large a translation is set in `box`: as large as it will go there, but no
 * larger than the page is lettered. Both halves are needed — without the box a
 * long line overruns its balloon, without the ceiling a short one fills it.
 */
export function sizeFor(
  text: string,
  box: Box,
  original: string,
  block: Box,
): number {
  return fitSize(
    text,
    box[2] - box[0],
    box[3] - box[1],
    originalSize(original, block[2] - block[0], block[3] - block[1]),
  )
}

/**
 * Below this a budget is not worth sending: two or three words is not a length a
 * line can be written to, it is one that comes out as a telegram.
 */
const BUDGET_MIN = 12

/**
 * About how many characters a translation of this block has room for, at the
 * size the rest of the page is lettered at, or 0 for "not worth saying".
 */
export function budgetFor(
  region: Pick<Region, 'box' | 'bubble'>,
  original: string,
): number {
  const room = roomFor(region)
  const block = region.box
  if (!original.trim()) return 0
  const fits = roomInCharacters(
    room[2] - room[0],
    room[3] - room[1],
    originalSize(original, block[2] - block[0], block[3] - block[1]),
  )
  return fits < BUDGET_MIN ? 0 : fits
}

/** That size for the block at `index`, or null where there is no such block. */
function sizeAt(
  analysis: Analysis,
  index: number,
  text: string,
  box: Box,
): number | null {
  const block = analysis.detection.regions[index]
  if (!block) return null
  return sizeFor(text, box, analysis.texts?.[index] ?? '', block.box)
}

/** Room for a block put in at `at`, with nothing lettered in it yet. */
export function inserted(lines: Lines, at: number): Lines {
  return insertAt(lines, at, null)
}

/** In step with a block dragged to a different place in the list. */
export function moved(lines: Lines, from: number, to: number): Lines {
  return moveAt(lines, from, to)
}

/** One line changed, and the rest left exactly as they were. */
export function withLine(
  lines: Lines,
  index: number,
  patch: Partial<Lettering>,
): Lines {
  const line = lines[index]
  if (!line) return lines
  const next = [...lines]
  next[index] = { ...line, ...patch }
  return next
}

/** One line's translation and box, laid out afresh. */
export function laidOut(
  analysis: Analysis,
  index: number,
  text: string,
  angle = 0,
): Lettering | null {
  const block = analysis.detection.regions[index]
  if (!block) return null
  const box = roomFor(block)
  const size = sizeAt(analysis, index, text, box)
  return size === null ? null : { text, box, size, angle }
}


/**
 * One line cut in two, in step with the block being cut. Both halves are held to
 * the size of the block *before* the cut.
 */
export function split(
  lines: Lines,
  index: number,
  halves: [{ text: string; box: Box }, { text: string; box: Box }],
  original: string,
  block: Box,
): Lines {
  const was = lines[index]
  if (!was) return lines

  const most = originalSize(original, block[2] - block[0], block[3] - block[1])
  const put = ({ text, box }: { text: string; box: Box }): Lettering => ({
    ...was,
    text,
    box,
    size: fitSize(text, box[2] - box[0], box[3] - box[1], most),
  })

  const next = [...lines]
  next[index] = put(halves[0])
  return insertAt(next, index + 1, put(halves[1]))
}
