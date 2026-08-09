import { useMemo } from 'react'
import { useBoxDrag } from '../hooks/useBoxDrag'
import { useLetteringFont } from '../hooks/useLetteringFont'
import type { Box, Lettering } from '../lib/api'
import {
  FONT_STACK,
  FONT_WEIGHT,
  LINE_HEIGHT,
  linesFor,
  strokeFor,
} from '../lib/fit'
import { BoxGrips } from './BoxGrips'

type Props = {
  page: { width: number; height: number }
  /** Drawn pixels per page pixel, for turning a drag into page pixels. */
  scale: number
  lettering: (Lettering | null)[]
  selected: number | null
  onSelect: (index: number | null) => void
  onBox: (index: number, box: Box) => void
  onTurn: (index: number, angle: number) => void
}

/** The translated lines, set over the page where the originals were. */
export function TranslationLayer({
  page,
  scale,
  lettering,
  selected,
  onSelect,
  onBox,
  onTurn,
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
            onTurn={(angle) => onTurn(index, angle)}
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
  onTurn,
}: {
  index: number
  set: Lettering
  page: { width: number; height: number }
  scale: number
  active: boolean
  onSelect: () => void
  onBox: (box: Box) => void
  onTurn: (angle: number) => void
}) {
  const drag = useBoxDrag({
    box: set.box,
    page,
    scale,
    onBox,
    angle: set.angle,
    onAngle: onTurn,
  })
  const [x0, y0, x1, y1] = set.box

  // Broken here rather than by the browser: only this knows to leave a hyphen
  // behind, and the page is drawn from the same call.
  // Until the face is in there is nothing worth measuring against — the breaks
  // would be the fallback's, not this font's — so it stays one line until then.
  const fontIn = useLetteringFont()
  const lines = useMemo(
    () => (fontIn ? linesFor(set.text, x1 - x0, set.size) : [set.text.trim()]),
    [set.text, set.size, x1, x0, fontIn],
  )

  return (
    <div
      data-box
      style={{
        left: `${(x0 / page.width) * 100}%`,
        top: `${(y0 / page.height) * 100}%`,
        width: `${((x1 - x0) / page.width) * 100}%`,
        height: `${((y1 - y0) / page.height) * 100}%`,
        // About the middle, which is where the canvas turns it too.
        transform: set.angle ? `rotate(${set.angle}deg)` : undefined,
      }}
      className={`absolute ${active ? 'z-20' : 'z-10'}`}
    >
      <button
        type="button"
        onPointerDown={drag.grab}
        onPointerMove={drag.shift}
        onPointerUp={drag.release}
        onPointerCancel={drag.release}
        onClick={() => {
          // The click that ends a drag is not a click on the box.
          if (drag.dragged.current) {
            drag.dragged.current = false
            return
          }
          onSelect()
        }}
        aria-label={`Translation ${index + 1}: ${set.text}`}
        aria-pressed={active}
        className={`flex h-full w-full cursor-move touch-none items-center justify-center text-center text-black ring-1 transition-colors ${
          active ? 'ring-accent' : 'ring-accent/30 hover:ring-accent/70'
        }`}
        style={{
          fontFamily: FONT_STACK,
          fontWeight: FONT_WEIGHT,
          fontSize: `${set.size * scale}px`,
          lineHeight: LINE_HEIGHT,
          // White laid under the letters, not over them: without this the words
          // are unreadable anywhere the art behind them is dark.
          WebkitTextStrokeWidth: `${strokeFor(set.size) * scale}px`,
          WebkitTextStrokeColor: '#fff',
          paintOrder: 'stroke fill',
        }}
      >
        <span className="block w-full">
          {lines.map((line, at) => (
            // Already broken to fit, so it must not be broken again.
            <span key={at} className="block whitespace-pre">
              {line}
            </span>
          ))}
        </span>
      </button>

      {active && <BoxGrips drag={drag} />}
    </div>
  )
}
