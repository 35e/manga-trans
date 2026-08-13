import { Button, Field, Note, Select, TextInput, Toolbar } from './ui'

type Props = {
  models: string[]
  model: string
  onModel: (model: string) => void
  target: string
  onTarget: (target: string) => void
  onFitAll: () => void
  canFit: boolean
  /** Translate the page over again, against the blocks as they stand now. */
  onTranslate: () => void
  canTranslate: boolean
  /** Whether there is anything set on the page yet: until there is, the button
   * on the header is the one that translates, and a second would only confuse. */
  lettered: boolean
  note: string | null
}

/** Which model, into what, and the button that sets it going. */
export function TranslateTools({
  models,
  model,
  onModel,
  target,
  onTarget,
  onFitAll,
  canFit,
  onTranslate,
  canTranslate,
  lettered,
  note,
}: Props) {
  return (
    <Toolbar>
      <Field label="Model">
        <Select
          value={model}
          onChange={(event) => onModel(event.target.value)}
          disabled={models.length === 0}
        >
          {models.length === 0 && <option value="">none found</option>}
          {models.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </Select>
      </Field>

      <Field label="Into">
        <TextInput
          value={target}
          onChange={(event) => onTarget(event.target.value)}
          spellCheck={false}
          className="w-28"
        />
      </Field>

      {note && <Note>{note}</Note>}

      <div className="ml-auto flex shrink-0 items-center gap-2">
        {lettered && (
          <Button
            onClick={onTranslate}
            disabled={!canTranslate}
            title="Translate the page again, against the blocks as they stand now — one added or dropped since is taken in, and every line already set is replaced"
          >
            Translate again
          </Button>
        )}
        <Button
          onClick={onFitAll}
          disabled={!canFit}
          title="Set every line at the largest size that lands in its box, held to the size this page is lettered at"
        >
          Fit all
        </Button>
      </div>
    </Toolbar>
  )
}
