import { useRef, useState } from 'react'
import type { Box } from '../lib/api'
import type { Point } from '../lib/mask'

/** A stray click is not a block: it takes a real drag, this many page pixels. */
const SMALLEST = 6

type Props = {
  page: { width: number; height: number }
  onAdd: (box: Box) => void
}

/**
 * The whole page as a drawing surface, for a block the detector missed.
 *
 * Laid over the blocks rather than under them: while one is being drawn, a
 * pointer down anywhere means the start of it.
 */
export function DrawRegion({ page, onAdd }: Props) {
  const from = useRef<Point | null>(null)
  const [drawn, setDrawn] = useState<Box | null>(null)

  const at = (event: React.PointerEvent<HTMLDivElement>): Point => {
    const rect = event.currentTarget.getBoundingClientRect()
    return {
      x: ((event.clientX - rect.left) / rect.width) * page.width,
      y: ((event.clientY - rect.top) / rect.height) * page.height,
    }
  }

  /** The box between two corners, in page pixels, corners put in order. */
  const between = (one: Point, other: Point): Box => [
    Math.max(0, Math.round(Math.min(one.x, other.x))),
    Math.max(0, Math.round(Math.min(one.y, other.y))),
    Math.min(page.width, Math.round(Math.max(one.x, other.x))),
    Math.min(page.height, Math.round(Math.max(one.y, other.y))),
  ]

  return (
    <div
      className="absolute inset-0 z-30 cursor-crosshair touch-none"
      onPointerDown={(event) => {
        event.currentTarget.setPointerCapture(event.pointerId)
        from.current = at(event)
      }}
      onPointerMove={(event) => {
        if (from.current) setDrawn(between(from.current, at(event)))
      }}
      onPointerUp={(event) => {
        if (event.currentTarget.hasPointerCapture(event.pointerId)) {
          event.currentTarget.releasePointerCapture(event.pointerId)
        }
        from.current = null
        if (drawn && drawn[2] - drawn[0] > SMALLEST && drawn[3] - drawn[1] > SMALLEST) {
          onAdd(drawn)
        }
        setDrawn(null)
      }}
    >
      {drawn && (
        <span
          aria-hidden="true"
          style={{
            left: `${(drawn[0] / page.width) * 100}%`,
            top: `${(drawn[1] / page.height) * 100}%`,
            width: `${((drawn[2] - drawn[0]) / page.width) * 100}%`,
            height: `${((drawn[3] - drawn[1]) / page.height) * 100}%`,
          }}
          className="absolute border-2 border-dashed border-accent bg-accent/20"
        />
      )}
    </div>
  )
}
