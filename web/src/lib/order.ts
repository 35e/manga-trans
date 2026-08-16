import type { Box } from './api'

function key(box: Box, rtl: boolean): [number, number] {
  return [box[1], rtl ? -box[2] : box[0]]
}

function after(one: [number, number], other: [number, number]): boolean {
  return one[0] !== other[0] ? one[0] > other[0] : one[1] > other[1]
}

function readsBefore(one: Box, other: Box, rtl: boolean): boolean {
  return !after(key(one, rtl), key(other, rtl))
}

const SLIVER = 12

function cutAt(length: number, share: number): number {
  if (length < SLIVER * 2) return length / 2
  return Math.min(Math.max(length * share, SLIVER), length - SLIVER)
}

export function halves(box: Box, share: number, rtl: boolean): [Box, Box] {
  const [x0, y0, x1, y1] = box
  const width = x1 - x0
  const height = y1 - y0

  let one: Box
  let other: Box
  if (height >= width) {
    const cut = Math.round(y0 + cutAt(height, share))
    one = [x0, y0, x1, cut]
    other = [x0, cut, x1, y1]
  } else {
    const cut = Math.round(x0 + cutAt(width, share))
    one = [x0, y0, cut, y1]
    other = [cut, y0, x1, y1]
  }

  return readsBefore(one, other, rtl) ? [one, other] : [other, one]
}

export function insertionFor(boxes: Box[], box: Box, rtl: boolean): number {
  const mine = key(box, rtl)
  const at = boxes.findIndex((held) => after(key(held, rtl), mine))
  return at === -1 ? boxes.length : at
}

export function insertAt<T>(list: T[], at: number, value: T): T[] {
  const next = [...list]
  next.splice(at, 0, value)
  return next
}

export function moveAt<T>(list: T[], from: number, to: number): T[] {
  const next = [...list]
  const [held] = next.splice(from, 1)
  next.splice(to, 0, held)
  return next
}

export function movedIndex(index: number, from: number, to: number): number {
  if (index === from) return to
  if (from < to) return index > from && index <= to ? index - 1 : index
  return index >= to && index < from ? index + 1 : index
}
