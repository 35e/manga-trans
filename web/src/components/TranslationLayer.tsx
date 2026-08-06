import { useRef } from 'react'
import type { Box, Lettering } from '../lib/api'
import { FONT_STACK, FONT_WEIGHT, LINE_HEIGHT } from '../lib/fit'

/** Which edges a handle moves. */
type Grip = 'n' | 's' | 'e' | 'w' | 'se'

const GRIPS: { grip: Grip; className: string; cursor: string }[] = [
  { grip: 'n', className: 'top-0 left-1/2 h-1.5 w-6 -translate-x-1/2 -translate-y-1/2', cursor: 'ns-resize' },
  { grip: 's', className: 'bottom-0 left-1/2 h-1.5 w-6 -translate-x-1/2 translate-y-1/2', cursor: 'ns-resize' },
  { grip: 'w', className: 'top-1/2 left-0 h-6 w-1.5 -translate-x-1/2 -translate-y-1/2', cursor: 'ew-resize' },
  { grip: 'e', className: 'top-1/2 right-0 h-6 w-1.5 translate-x-1/2 -translate-y-1/2', cursor: 'ew-resize' },
  { grip: 'se', className: 'right-0 bottom-0 size-2.5 translate-x-1/2 translate-y-1/2', cursor: 'nwse-resize' },
]

const SMALLEST = 12
/** Move less than this and it was a click, not a drag. */
const SLOP = 3

type Props = {
  page: { width: number; height: number }
  /** Drawn pixels per page pixel, for turning a drag into page pixels. */
  scale: number
  lettering: (Lettering | null)[]
  selected: number | null
  onSelect: (index: number | null) => void
  onBox: (index: number, box: Box) => void
}

/** The translated lines, set over the page where the originals were. */
export function TranslationLayer({
  page,
  scale,
  lettering,
  selected,
  onSelect,
  onBox,
}: Props) {
  return (
    <>
      {lettering.map((set, index) =>
        set === null ? null : (
          <Line
            key={index}
            index={index}
            set={set}
            page={page}
            scale={scale}
            active={selected === index}
            onSelect={() => onSelect(selected === index ? null : index)}
            onBox={(box) => onBox(index, box)}
          />
        ),
      )}
    </>
  )
}

function Line({
  index,
  set,
  page,
  scale,
  active,
  onSelect,
  onBox,
}: {
  index: number
  set: Lettering
  page: { width: number; height: number }
  scale: number
  active: boolean
  onSelect: () => void
  onBox: (box: Box) => void
}) {
  const from = useRef<{ x: number; y: number; box: Box } | null>(null)
  const dragged = useRef(false)
  const [x0, y0, x1, y1] = set.box

  const grab = (event: React.PointerEvent<HTMLElement>) => {
    event.stopPropagation()
    event.currentTarget.setPointerCapture(event.pointerId)
    from.current = { x: event.clientX, y: event.clientY, box: set.box }
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

  const settle = (box: Box): Box => [
    Math.max(0, Math.round(box[0])),
    Math.max(0, Math.round(box[1])),
    Math.min(page.width, Math.round(box[2])),
    Math.min(page.height, Math.round(box[3])),
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
    from.current = null
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
  }

  return (
    <div
      style={{
        left: `${(x0 / page.width) * 100}%`,
        top: `${(y0 / page.height) * 100}%`,
        width: `${((x1 - x0) / page.width) * 100}%`,
        height: `${((y1 - y0) / page.height) * 100}%`,
      }}
      className={`absolute ${active ? 'z-20' : 'z-10'}`}
    >
      <button
        type="button"
        onPointerDown={grab}
        onPointerMove={shift}
        onPointerUp={release}
        onPointerCancel={release}
        onClick={() => {
          // The click that ends a drag is not a click on the box.
          if (dragged.current) {
            dragged.current = false
            return
          }
          onSelect()
        }}
        aria-label={`Translation ${index + 1}: ${set.text}`}
        aria-pressed={active}
        className={`flex h-full w-full cursor-move touch-none items-center justify-center bg-white/85 px-0.5 text-center wrap-anywhere text-black transition-colors ${
          active
            ? 'ring-2 ring-indigo-500'
            : 'ring-1 ring-indigo-500/30 hover:ring-indigo-500/70'
        }`}
        style={{
          fontFamily: FONT_STACK,
          fontWeight: FONT_WEIGHT,
          fontSize: `${set.size * scale}px`,
          lineHeight: LINE_HEIGHT,
        }}
      >
        {set.text}
      </button>

      {active &&
        GRIPS.map(({ grip, className, cursor }) => (
          <span
            key={grip}
            role="presentation"
            onPointerDown={grab}
            onPointerMove={move(grip)}
            onPointerUp={release}
            onPointerCancel={release}
            style={{ cursor }}
            className={`absolute touch-none rounded-xs bg-indigo-500 ring-1 ring-white ${className}`}
          />
        ))}
    </div>
  )
}
