/**
 * The arithmetic of a box whose contents are turned. The box itself stays square
 * to the page; only what sits in it turns.
 */

import type { Box } from './api'

function turned(degrees: number) {
  const radians = (degrees * Math.PI) / 180
  return { cos: Math.cos(radians), sin: Math.sin(radians) }
}

/**
 * A drag across the screen, read along the box's own axes: the handles turn with
 * the box, so pulling the right one means "wider" whichever way it now points.
 */
export function alongBox(angle: number, dx: number, dy: number) {
  if (!angle) return { dx, dy }
  const { cos, sin } = turned(angle)
  return { dx: dx * cos + dy * sin, dy: -dx * sin + dy * cos }
}

/**
 * A pulled box, put back so the edge that was not pulled stays where it looks.
 *
 * Writing a drawn point as `c + R(p − c)`, the shift that holds the untouched
 * edge still is `(I − R)(c − c′)` — the two middles alone, so one correction
 * serves every handle.
 */
export function anchored(angle: number, was: Box, now: Box): Box {
  if (!angle) return now
  const { cos, sin } = turned(angle)
  const x = (was[0] + was[2] - now[0] - now[2]) / 2
  const y = (was[1] + was[3] - now[1] - now[3]) / 2
  const dx = x - (x * cos - y * sin)
  const dy = y - (x * sin + y * cos)
  return [now[0] + dx, now[1] + dy, now[2] + dx, now[3] + dy]
}
