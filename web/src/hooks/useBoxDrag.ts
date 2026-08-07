import { useRef } from 'react'
import type { Box } from '../lib/api'
import { alongBox, anchored } from '../lib/turn'

/** Which edges a handle moves. */
export type Grip = 'n' | 's' | 'e' | 'w' | 'se'

/** No drag may pull a box smaller than this, in page pixels. */
const SMALLEST = 12

/** Move less than this and it was a click, not a drag. */
const SLOP = 3

type Options = {
  /** The box as it stands, which is what a drag starts from. */
  box: Box
  /** The page's own size, which nothing may be dragged outside of. */
  page: { width: number; height: number }
  /** Drawn pixels per page pixel, for turning a drag into page pixels. */
  scale: number
  onBox: (box: Box) => void
  /** Once, when the drag ends, with the box as it was before it began. */
  onSettled?: (was: Box) => void
  /** How far what is in the box is turned, in degrees clockwise. */
  angle?: number
  /** Given, the box can be turned as well as moved and pulled. */
  onAngle?: (angle: number) => void
}

/** Whole degrees, or a sixth of a right angle while shift is held. */
const TURN_STEP = 1
const TURN_STEP_HELD = 15

export type BoxDrag = ReturnType<typeof useBoxDrag>

/**
 * Dragging a box about the page, and pulling it by its edges.
 *
 * Used by everything laid over the page in the page's own coordinates — the
 * blocks the detector found, and the lettering set back over them. A drag
 * arrives in screen pixels and is handed back in page pixels, which is what
 * everything that holds a box is written in.
 */
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

  /** How far the pointer has come, in the page's own pixels. */
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

  /** Drag the box itself: it goes where it is put, the same size it was. */
  const shift = (event: React.PointerEvent<HTMLElement>) => {
    const drag = since(event)
    if (!drag) return
    const [bx0, by0, bx1, by1] = drag.box
    const width = bx1 - bx0
    const height = by1 - by0
    // Clamping where it lands rather than its edges is what keeps the size:
    // pushed against the edge of the page it stops, it does not squash.
    const left = Math.min(Math.max(0, bx0 + drag.dx), page.width - width)
    const top = Math.min(Math.max(0, by0 + drag.dy), page.height - height)
    onBox(settle([left, top, left + width, top + height]))
  }

  /** Pull one edge, or the corner, and leave the others where they are. */
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

  /**
   * Turn what is in the box: the handle follows the pointer round the middle.
   *
   * The middle is read off the element every time rather than kept from where
   * the drag began, which it can be because turning about the middle is the one
   * thing that cannot move it.
   */
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
      // The handle stands above the box, so pointing straight up is straight.
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
    // Only a drag that went somewhere is worth telling anyone about: a click
    // lands here too, having moved nothing.
    if (start && dragged.current) onSettled?.(start.box)
  }

  return { grab, shift, move, spin, release, dragged, turnable: Boolean(onAngle) }
}
