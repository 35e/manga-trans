import { useRef } from 'react'
import type { Box } from '../lib/api'
import { alongBox, anchored } from '../lib/turn'

export type Grip = 'n' | 's' | 'e' | 'w' | 'se'

const SMALLEST = 12

const SLOP = 3

type Options = {
  box: Box
  page: { width: number; height: number }
  scale: number
  onBox: (box: Box) => void
  onSettled?: (was: Box) => void
  angle?: number
  onAngle?: (angle: number) => void
}

const TURN_STEP = 1
const TURN_STEP_HELD = 15

export type BoxDrag = ReturnType<typeof useBoxDrag>

export function useBoxDrag({
  box,
  page,
  scale,
  onBox,
  onSettled,
  angle = 0,
  onAngle,
}: Options) {
  const from = useRef<{ x: number; y: number; box: Box } | null>(null)
  const dragged = useRef(false)

  const grab = (event: React.PointerEvent<HTMLElement>) => {
    event.stopPropagation()
    event.currentTarget.setPointerCapture(event.pointerId)
    from.current = { x: event.clientX, y: event.clientY, box }
    dragged.current = false
  }

  const since = (event: React.PointerEvent<HTMLElement>) => {
    const start = from.current
    if (!start) return null
    if (
      Math.abs(event.clientX - start.x) > SLOP ||
      Math.abs(event.clientY - start.y) > SLOP
    ) {
      dragged.current = true
    }
    return {
      dx: (event.clientX - start.x) / scale,
      dy: (event.clientY - start.y) / scale,
      box: start.box,
    }
  }

  const settle = (edges: Box): Box => [
    Math.max(0, Math.round(edges[0])),
    Math.max(0, Math.round(edges[1])),
    Math.min(page.width, Math.round(edges[2])),
    Math.min(page.height, Math.round(edges[3])),
  ]

  const shift = (event: React.PointerEvent<HTMLElement>) => {
    const drag = since(event)
    if (!drag) return
    const [bx0, by0, bx1, by1] = drag.box
    const width = bx1 - bx0
    const height = by1 - by0
    const left = Math.min(Math.max(0, bx0 + drag.dx), page.width - width)
    const top = Math.min(Math.max(0, by0 + drag.dy), page.height - height)
    onBox(settle([left, top, left + width, top + height]))
  }

  const move = (grip: Grip) => (event: React.PointerEvent<HTMLElement>) => {
    const drag = since(event)
    if (!drag) return
    const [bx0, by0, bx1, by1] = drag.box
    const { dx, dy } = alongBox(angle, drag.dx, drag.dy)

    const pulled: Box = [
      grip === 'w' ? Math.min(bx0 + dx, bx1 - SMALLEST) : bx0,
      grip === 'n' ? Math.min(by0 + dy, by1 - SMALLEST) : by0,
      grip === 'e' || grip === 'se' ? Math.max(bx1 + dx, bx0 + SMALLEST) : bx1,
      grip === 's' || grip === 'se' ? Math.max(by1 + dy, by0 + SMALLEST) : by1,
    ]
    onBox(settle(anchored(angle, drag.box, pulled)))
  }

  const spin = (event: React.PointerEvent<HTMLElement>) => {
    if (!onAngle || !from.current) return
    const around = event.currentTarget.parentElement?.getBoundingClientRect()
    if (!around) return

    dragged.current = true
    const degrees =
      (Math.atan2(
        event.clientY - (around.top + around.height / 2),
        event.clientX - (around.left + around.width / 2),
      ) *
        180) /
        Math.PI +
      90
    const step = event.shiftKey ? TURN_STEP_HELD : TURN_STEP
    onAngle((((Math.round(degrees / step) * step) % 360) + 360) % 360)
  }

  const release = (event: React.PointerEvent<HTMLElement>) => {
    const start = from.current
    from.current = null
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
    if (start && dragged.current) onSettled?.(start.box)
  }

  return { grab, shift, move, spin, release, dragged, turnable: Boolean(onAngle) }
}
