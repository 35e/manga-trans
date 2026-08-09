import type { Fill } from '../lib/api'
import type { Brush } from '../lib/mask'
import { Button, Divider, Field, Note, Segmented, Select, Toolbar } from './ui'

type Props = {
  brush: Brush
  onBrush: (brush: Brush) => void
  /** Mark the whole box around every block, or the lettering inside them. */
  onMarkBlocks: () => void
  onMarkLetters: () => void
  canMark: boolean
  tracing: boolean
  onClear: () => void
  canClear: boolean
  spread: number
  onSpread: (spread: number) => void
  /** What the clean puts where the marked lettering was. */
  fill: Fill
  onFill: (fill: Fill) => void
  note: string | null
}

const SPREADS = [0, 2, 4, 6, 8, 12, 16]

const SIZES = { min: 4, max: 160 }

/** The brush, and the two shortcuts worth having beside it. */
export function MaskTools({
  brush,
  onBrush,
  onMarkBlocks,
  onMarkLetters,
  canMark,
  tracing,
  onClear,
  canClear,
  spread,
  onSpread,
  fill,
  onFill,
  note,
}: Props) {
  return (
    <Toolbar>
      <Segmented
        label="Brush"
        value={brush.erase ? 'erase' : 'draw'}
        onChange={(tool) => onBrush({ ...brush, erase: tool === 'erase' })}
        options={[
          { value: 'draw', label: 'Draw' },
          { value: 'erase', label: 'Erase' },
        ]}
      />

      <Field label="Size">
        <input
          type="range"
          min={SIZES.min}
          max={SIZES.max}
          value={brush.radius}
          onChange={(event) => onBrush({ ...brush, radius: Number(event.target.value) })}
          className="w-24 accent-accent"
          aria-label="Brush size"
        />
        <span className="w-10 text-right tabular-nums">{brush.radius * 2}px</span>
      </Field>

      <Divider />

      <Field
        label="Hide under"
        title="What goes where the lettering was: the page around it, filled in, or flat white"
      >
        <Segmented
          label="What to hide the lettering under"
          value={fill}
          onChange={onFill}
          options={[
            { value: 'art', label: 'The art' },
            { value: 'white', label: 'White' },
          ]}
        />
      </Field>

      {note && <Note>{note}</Note>}

      <div className="ml-auto flex shrink-0 items-center gap-2">
        <span className="text-xs font-medium text-faint">
          {tracing ? 'Tracing the lettering…' : 'Mark'}
        </span>
        <Button
          onClick={onMarkLetters}
          disabled={!canMark || tracing}
          title="Mark the lettering itself, leaving the art it sits on"
        >
          Letters
        </Button>
        <Field
          label="+"
          title="How far past the ink the mask reaches, in page pixels. Raise it if edges are left behind."
        >
          <Select
            value={spread}
            onChange={(event) => onSpread(Number(event.target.value))}
            aria-label="How far past the ink to mark"
          >
            {SPREADS.map((size) => (
              <option key={size} value={size}>
                {size}px
              </option>
            ))}
          </Select>
        </Field>
        <Button
          onClick={onMarkBlocks}
          disabled={!canMark}
          title="Mark the whole box around every block the detector found"
        >
          Blocks
        </Button>
        <Button onClick={onClear} disabled={!canClear}>
          Clear mask
        </Button>
      </div>
    </Toolbar>
  )
}
