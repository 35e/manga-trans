import type { Box } from './api'

/**
 * Where a box falls in reading order.
 *
 * The same key the detector sorts its own blocks by — down the page, and then
 * right to left across it, which is the way a Japanese page is read. A block
 * added by hand has to land where the detector would have put it, or the page
 * reads out of order and translates as a jumbled conversation.
 */
function key(box: Box): [number, number] {
  return [box[1], -box[2]]
}

function after(one: [number, number], other: [number, number]): boolean {
  return one[0] !== other[0] ? one[0] > other[0] : one[1] > other[1]
}

/** The index a box belongs at, among blocks already in reading order. */
export function insertionFor(boxes: Box[], box: Box): number {
  const mine = key(box)
  const at = boxes.findIndex((held) => after(key(held), mine))
  return at === -1 ? boxes.length : at
}

/** The same list with `value` put in at `at`. */
export function insertAt<T>(list: T[], at: number, value: T): T[] {
  const next = [...list]
  next.splice(at, 0, value)
  return next
}

/** The same list with whatever is at `from` taken out and put back in at `to`. */
export function moveAt<T>(list: T[], from: number, to: number): T[] {
  const next = [...list]
  const [held] = next.splice(from, 1)
  next.splice(to, 0, held)
  return next
}

/**
 * Where an index ends up after that same move.
 *
 * Blocks are pointed at by their place in the list — which ones are left alone,
 * which one is picked out — so every one of those pointers has to be carried
 * across a reorder rather than left aiming at whoever moved into the slot.
 */
export function movedIndex(index: number, from: number, to: number): number {
  if (index === from) return to
  if (from < to) return index > from && index <= to ? index - 1 : index
  return index >= to && index < from ? index + 1 : index
}
