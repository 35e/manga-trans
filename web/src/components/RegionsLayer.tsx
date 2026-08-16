import { useBoxDrag } from '../hooks/useBoxDrag'
import type { Analysis, Box, Region } from '../lib/api'
import { UNSURE } from '../lib/api'
import { BoxGrips } from './BoxGrips'

type Props = {
  analysis: Analysis
  scale: number
  selected: number | null
  onSelect: (index: number | null) => void
  onBox: (index: number, box: Box) => void
  onSettled: (index: number, was: Box) => void
}

export function RegionsLayer({
  analysis,
  scale,
  selected,
  onSelect,
  onBox,
  onSettled,
}: Props) {
  const { detection, texts } = analysis
  const excluded = new Set(analysis.excluded)

  return (
    <>
      {detection.regions.map((region, index) => (
        <RegionBox
          key={region.id}
          region={region}
          index={index}
          page={detection}
          scale={scale}
          text={texts?.[index] ?? null}
          excluded={excluded.has(index)}
          active={selected === index}
          onSelect={() => onSelect(selected === index ? null : index)}
          onBox={(box) => onBox(index, box)}
          onSettled={(was) => onSettled(index, was)}
        />
      ))}
    </>
  )
}

function RegionBox({
  region,
  index,
  page,
  scale,
  text,
  excluded,
  active,
  onSelect,
  onBox,
  onSettled,
}: {
  region: Region
  index: number
  page: { width: number; height: number }
  scale: number
  text: string | null
  excluded: boolean
  active: boolean
  onSelect: () => void
  onBox: (box: Box) => void
  onSettled: (was: Box) => void
}) {
  const drag = useBoxDrag({ box: region.box, page, scale, onBox, onSettled })
  const [x0, y0, x1, y1] = region.box
  const unsure = region.confidence < UNSURE

  return (
    <div
      data-box
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
        onPointerDown={drag.grab}
        onPointerMove={drag.shift}
        onPointerUp={drag.release}
        onPointerCancel={drag.release}
        onClick={() => {
          if (drag.dragged.current) {
            drag.dragged.current = false
            return
          }
          onSelect()
        }}
        title={excluded ? `${text ?? ''} — left alone`.trim() : text || undefined}
        aria-label={
          excluded
            ? `Block ${index + 1}, left alone`
            : text
              ? `Block ${index + 1}: ${text}`
              : `Text block ${index + 1}`
        }
        aria-pressed={active}
        className={`h-full w-full cursor-move touch-none border transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white ${
          excluded
            ? 'border-dashed border-faint/60 hover:border-faint'
            : active
              ? 'border-accent bg-accent/10'
              : unsure
                ? 'border-warn/60 hover:border-warn hover:bg-warn/10'
                : 'border-accent/40 hover:border-accent hover:bg-accent/10'
        }`}
      >
        <span
          className={`absolute -top-px -left-px rounded-br px-1 text-[9px] leading-4 font-medium text-white tabular-nums transition-colors ${
            excluded
              ? 'bg-faint/70 line-through'
              : active
                ? 'bg-accent'
                : unsure
                  ? 'bg-warn/80'
                  : 'bg-accent/60'
          }`}
        >
          {index + 1}
        </span>
      </button>

      {active && <BoxGrips drag={drag} />}
    </div>
  )
}
