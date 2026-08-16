
import type { Analysis, Box, Lettering, Region } from './api'
import { fitSize, originalSize, roomInCharacters } from './fit'
import { insertAt, moveAt } from './order'

export type Lines = (Lettering | null)[]

export function roomFor(region: Pick<Region, 'box' | 'bubble'>): Box {
  return region.bubble ?? region.box
}

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

const BUDGET_MIN = 12

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

export function inserted(lines: Lines, at: number): Lines {
  return insertAt(lines, at, null)
}

export function moved(lines: Lines, from: number, to: number): Lines {
  return moveAt(lines, from, to)
}

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
