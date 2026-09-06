import type { Fill } from '../lib/api'
import type { Brush } from '../lib/mask'
import { Button, Divider, Field, Note, Segmented, Select, Toolbar } from './ui'

type Props = {
  brush: Brush
  onBrush: (brush: Brush) => void
  onMarkLetters: () => void
  canMark: boolean
  tracing: boolean
  onClear: () => void
  canClear: boolean
  spread: number
  onSpread: (spread: number) => void
  fill: Fill
  onFill: (fill: Fill) => void
  note: string | null
  onClean: () => void
  canClean: boolean
  cleaned: boolean
  busy: boolean
}

const SPREADS = [0, 2, 4, 6, 8, 12, 16]

const SIZES = { min: 4, max: 160 }

export function MaskTools({
  brush,
  onBrush,
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
  onClean,
  canClean,
  cleaned,
  busy,
}: Props) {
  return (
    <Toolbar>
      <fieldset disabled={busy} className="flex min-w-0 flex-wrap items-center gap-2.5 disabled:opacity-60">
      <Segmented
        label="Brush"
        value={brush.erase ? 'erase' : 'draw'}
        onChange={(tool) => onBrush({ ...brush, erase: tool === 'erase' })}
        options={[
          { value: 'draw', label: 'Paint to remove' },
          { value: 'erase', label: 'Unmark' },
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
        label="Fill with"
        title="Use white for plain speech bubbles, or restore the surrounding artwork"
      >
        <Segmented
          label="Cleanup fill method"
          value={fill}
          onChange={onFill}
          options={[
            { value: 'white', label: 'White' },
            { value: 'art', label: 'Restore artwork' },
            { value: 'telea', label: 'Fast fill' },
          ]}
        />
      </Field>

      {note && <Note>{note}</Note>}

      {!cleaned && (
        <>
        <Button
          onClick={onMarkLetters}
          disabled={!canMark || tracing}
          title="Mark the lettering itself, leaving the art it sits on"
        >
          {tracing ? 'Marking text…' : 'Auto-mark text'}
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
        </>
      )}
        <Button onClick={onClear} disabled={!canClear}>
          Clear marks
        </Button>
        <Button
          variant="primary"
          onClick={onClean}
          disabled={busy || !canClean}
          title={canClean ? 'Remove marked pixels without changing your translations' : 'Paint over the text you want to remove first'}
        >
          {busy ? 'Cleaning…' : cleaned ? 'Apply touch-up' : 'Clean page'}
        </Button>
      </fieldset>
    </Toolbar>
  )
}
