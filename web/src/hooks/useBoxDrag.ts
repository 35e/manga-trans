import { useRef } from 'react'
import type { Box } from '../lib/api'

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
}

export type BoxDrag = ReturnType<typeof useBoxDrag>

/**
 * Dragging a box about the page, and pulling it by its edges.
 *
 * Used by everything laid over the page in the page's own coordinates — the
 * blocks the detector found, and the lettering set back over them. A drag
 * arrives in screen pixels and is handed back in page pixels, which is what
 * everything that holds a box is written in.
 */
export function useBoxDrag({ box, page, scale, onBox, onSettled }: Options) {
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

    onBox(
      settle([
        grip === 'w' ? Math.min(bx0 + drag.dx, bx1 - SMALLEST) : bx0,
        grip === 'n' ? Math.min(by0 + drag.dy, by1 - SMALLEST) : by0,
        grip === 'e' || grip === 'se' ? Math.max(bx1 + drag.dx, bx0 + SMALLEST) : bx1,
        grip === 's' || grip === 'se' ? Math.max(by1 + drag.dy, by0 + SMALLEST) : by1,
      ]),
    )
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

  return { grab, shift, move, release, dragged }
}
