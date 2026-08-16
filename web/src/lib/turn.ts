
import type { Box } from './api'

function turned(degrees: number) {
  const radians = (degrees * Math.PI) / 180
  return { cos: Math.cos(radians), sin: Math.sin(radians) }
}

export function alongBox(angle: number, dx: number, dy: number) {
  if (!angle) return { dx, dy }
  const { cos, sin } = turned(angle)
  return { dx: dx * cos + dy * sin, dy: -dx * sin + dy * cos }
}

export function anchored(angle: number, was: Box, now: Box): Box {
  if (!angle) return now
  const { cos, sin } = turned(angle)
  const x = (was[0] + was[2] - now[0] - now[2]) / 2
  const y = (was[1] + was[3] - now[1] - now[3]) / 2
  const dx = x - (x * cos - y * sin)
  const dy = y - (x * sin + y * cos)
  return [now[0] + dx, now[1] + dy, now[2] + dx, now[3] + dy]
}
