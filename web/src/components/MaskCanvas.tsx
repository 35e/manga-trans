import { useCallback, useEffect, useLayoutEffect, useRef } from 'react'
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
  const pointer = useRef<number | null>(null)
  const frame = useRef<number | null>(null)

  const repaint = useCallback(() => {
    if (frame.current !== null) return
    frame.current = requestAnimationFrame(() => {
      frame.current = null
      if (overlay.current) mask.showOn(overlay.current)
    })
  }, [mask])
  useEffect(repaint)
  useLayoutEffect(() => {
    const canvas = overlay.current
    const dot = cursor.current
    return () => {
      if (frame.current !== null) cancelAnimationFrame(frame.current)
      frame.current = null
      drawing.current = null
      const id = pointer.current
      pointer.current = null
      if (id !== null && canvas?.hasPointerCapture(id)) canvas.releasePointerCapture(id)
      if (dot) dot.style.opacity = '0'
    }
  }, [mask, page.width, page.height])

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
          if (panning || event.button !== 0 || pointer.current !== null) return
          event.currentTarget.setPointerCapture(event.pointerId)
          pointer.current = event.pointerId
          const point = at(event)
          drawing.current = point
          mask.dot(point, brush)
          repaint()
        }}
        onPointerMove={(event) => {
          moveDot(event)
          if (!drawing.current || pointer.current !== event.pointerId) return
          const rect = event.currentTarget.getBoundingClientRect()
          const samples = event.nativeEvent.getCoalescedEvents?.() ?? []
          for (const sample of samples.length ? samples : [event.nativeEvent]) {
            const point = {
              x: ((sample.clientX - rect.left) / rect.width) * page.width,
              y: ((sample.clientY - rect.top) / rect.height) * page.height,
            }
            mask.stroke(drawing.current!, point, brush)
            drawing.current = point
          }
          repaint()
        }}
        onPointerUp={(event) => {
          if (!drawing.current || pointer.current !== event.pointerId) return
          mask.stroke(drawing.current, at(event), brush)
          drawing.current = null
          pointer.current = null
          repaint()
          if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            event.currentTarget.releasePointerCapture(event.pointerId)
          }
          onStroke()
        }}
        onPointerCancel={(event) => {
          if (!drawing.current || pointer.current !== event.pointerId) return
          drawing.current = null
          pointer.current = null
          repaint()
          onStroke()
        }}
        onLostPointerCapture={() => {
          if (!drawing.current) return
          drawing.current = null
          pointer.current = null
          repaint()
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
