import { useEffect, useRef } from 'react'
import type { Brush, Mask, Point } from '../lib/mask'

type Props = {
  page: { width: number; height: number }
  mask: Mask
  brush: Brush
  panning: boolean
  onStroke: () => void
}

export function MaskCanvas({ page, mask, brush, panning, onStroke }: Props) {
  const overlay = useRef<HTMLCanvasElement>(null)
  const cursor = useRef<HTMLDivElement>(null)
  const drawing = useRef<Point | null>(null)

  const repaint = () => {
    if (overlay.current) mask.showOn(overlay.current)
  }
  useEffect(repaint)

  const at = (event: React.PointerEvent<HTMLCanvasElement>): Point => {
    const rect = event.currentTarget.getBoundingClientRect()
    return {
      x: ((event.clientX - rect.left) / rect.width) * page.width,
      y: ((event.clientY - rect.top) / rect.height) * page.height,
    }
  }

  const moveDot = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const dot = cursor.current
    if (!dot) return
    const rect = event.currentTarget.getBoundingClientRect()
    const size = brush.radius * 2 * (rect.width / page.width)
    dot.style.width = `${size}px`
    dot.style.height = `${size}px`
    dot.style.transform = `translate(${event.clientX - rect.left - size / 2}px, ${
      event.clientY - rect.top - size / 2
    }px)`
    dot.style.opacity = '1'
  }

  return (
    <>
      <canvas
        ref={overlay}
        width={page.width}
        height={page.height}
        className={`absolute inset-0 h-full w-full touch-none ${
          panning ? '' : 'cursor-none'
        }`}
        onPointerDown={(event) => {
          event.currentTarget.setPointerCapture(event.pointerId)
          const point = at(event)
          drawing.current = point
          mask.dot(point, brush)
          repaint()
        }}
        onPointerMove={(event) => {
          moveDot(event)
          if (!drawing.current) return
          const point = at(event)
          mask.stroke(drawing.current, point, brush)
          drawing.current = point
          repaint()
        }}
        onPointerUp={(event) => {
          if (!drawing.current) return
          drawing.current = null
          if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            event.currentTarget.releasePointerCapture(event.pointerId)
          }
          onStroke()
        }}
        onPointerLeave={() => {
          if (cursor.current) cursor.current.style.opacity = '0'
        }}
      />

      <div
        ref={cursor}
        aria-hidden="true"
        className="pointer-events-none absolute top-0 left-0 rounded-full border-2 border-white opacity-0 ring-1 ring-black/70"
      />
    </>
  )
}
