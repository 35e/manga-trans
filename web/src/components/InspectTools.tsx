import type { Language } from '../lib/api'
import { Divider, Field, Hint, Select, Toggle, Toolbar } from './ui'

type Props = {
  /** Every language the API can read a page in, as it answered. */
  offered: Language[]
  language: string
  onLanguage: (code: string) => void
  /** Whether the page has been through the detector. */
  found: boolean
  showBoxes: boolean
  onShowBoxes: (showing: boolean) => void
  adding: boolean
  onAdding: (adding: boolean) => void
}

/**
 * What the page is written in, and what can be done to the blocks found in it.
 * The language sits here because this is the step it bears on.
 */
export function InspectTools({
  offered,
  language,
  onLanguage,
  found,
  showBoxes,
  onShowBoxes,
  adding,
  onAdding,
}: Props) {
  return (
    <Toolbar>
      <Field
        label="Page is in"
        title="What the pages are lettered in: who reads them, which way across the page they are read, and what the translator is told they are in. Remembered for next time"
      >
        <Select
          value={language}
          onChange={(event) => onLanguage(event.target.value)}
          disabled={offered.length === 0}
        >
          {offered.length === 0 && <option value={language}>{language}</option>}
          {offered.map((held) => (
            <option key={held.code} value={held.code}>
              {held.name}
            </option>
          ))}
        </Select>
      </Field>

      {found && (
        <>
          <Divider />
          <Toggle
            on={showBoxes}
            onChange={onShowBoxes}
            title="Show the blocks the detector found"
          >
            Boxes
          </Toggle>
          <Toggle
            on={adding}
            onChange={onAdding}
            title="Draw a block the detector missed"
          >
            {adding ? 'Drawing a block…' : 'Add a block'}
          </Toggle>
          <Divider />
          <Hint>
            {adding
              ? 'Drag across the bubble it missed. It is read and put in reading order.'
              : 'Click a block to pick it out, drag to move, pull an edge to resize; delete drops it from the clean.'}
          </Hint>
        </>
      )}
    </Toolbar>
  )
}
