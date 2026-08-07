/**
 * The arithmetic of a box whose contents are turned.
 *
 * A turned line is drawn rotated about the middle of its box — by the board
 * with a CSS transform, by the canvas with `rotate`, both about that same
 * point. The box itself stays square to the page, so wrapping and fitting go on
 * measuring what they always did; it is only what sits in it that turns.
 *
 * That leaves two things to work out when such a box is pulled by an edge, and
 * both are here rather than in the pointer handling, which has enough to do.
 */

import type { Box } from './api'

function turned(degrees: number) {
  const radians = (degrees * Math.PI) / 180
  return { cos: Math.cos(radians), sin: Math.sin(radians) }
}

/**
 * A drag across the screen, read along the box's own axes.
 *
 * The handles are drawn turned along with what is in the box, so pulling the
 * one on the right has to mean "wider" whichever way that now points.
 */
export function alongBox(angle: number, dx: number, dy: number) {
  if (!angle) return { dx, dy }
  const { cos, sin } = turned(angle)
  return { dx: dx * cos + dy * sin, dy: -dx * sin + dy * cos }
}

/**
 * A pulled box, put back so the edge that was not pulled stays where it looks.
 *
 * Growing a box moves its middle, and the middle is what the turn is about, so
 * the far edge swings away from where it sat — the box appears to slide as it
 * is stretched. Writing the drawn position of a point as `c + R(p − c)`, the
 * shift that holds any point of the untouched edge still is `(I − R)(c − c′)`,
 * which depends on the two middles alone: one correction serves every handle.
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
