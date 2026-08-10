import { useEffect, useState } from 'react'
import { Button, FOCUS, IconButton } from './ui'
import { CloseIcon } from './icons'

type Props = {
  onClose: () => void
  /** What is being used now: null when it is the API's own. */
  prompt: string | null
  /** The API's own, to start from and to go back to. */
  fallback: string | null
  onSave: (prompt: string | null) => void
  apiBase: string
  models: string[]
}

/**
 * Settings, such as they are: what the model is told, and what it is being told
 * it by. Kept in this browser rather than in the API, which stores nothing.
 */
export function Settings({ onClose, prompt, fallback, onSave, apiBase, models }: Props) {
  const [draft, setDraft] = useState(prompt ?? fallback ?? '')

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  // The default may still be on its way in when this is opened.
  useEffect(() => {
    if (prompt === null && fallback !== null) setDraft((now) => now || fallback)
  }, [prompt, fallback])

  const changed = draft !== (prompt ?? fallback ?? '')
  const isDefault = fallback !== null && draft.trim() === fallback.trim()

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Settings"
      onClick={onClose}
      className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/60 p-4 backdrop-blur-sm sm:p-8"
    >
      <div
        onClick={(event) => event.stopPropagation()}
        className="w-full max-w-2xl rounded-2xl border border-line bg-surface shadow-2xl"
      >
        <header className="flex items-start justify-between gap-4 border-b border-line px-5 py-4">
          <div>
            <h2 className="text-base font-semibold text-ink">Settings</h2>
            <p className="mt-0.5 text-xs text-faint">
              Kept in this browser. The API stores nothing and is sent these with each
              page.
            </p>
          </div>
          <IconButton label="Close settings" onClick={onClose}>
            <CloseIcon />
          </IconButton>
        </header>

        <div className="space-y-5 px-5 py-5">
          <section>
            <label htmlFor="prompt" className="text-sm font-medium text-ink">
              What the model is told
            </label>
            <p className="mt-0.5 text-xs text-faint">
              The system prompt every translation is asked with.{' '}
              <code className="rounded bg-raised px-1 py-0.5 text-muted">
                {'{target}'}
              </code>{' '}
              is replaced by the language you are translating into, and{' '}
              <code className="rounded bg-raised px-1 py-0.5 text-muted">
                {'{source}'}
              </code>{' '}
              by the one the page is in. Leave the second out and the model
              guesses — which for a page of Chinese usually means Japanese.
            </p>

            <textarea
              id="prompt"
              value={draft}
              rows={9}
              spellCheck={false}
              onChange={(event) => setDraft(event.target.value)}
              placeholder={fallback ?? 'Loading the default…'}
              className={`mt-2 w-full resize-y rounded-lg border border-line bg-canvas px-3 py-2 font-mono text-xs leading-relaxed text-ink ${FOCUS}`}
            />

            <p className="mt-1.5 text-xs text-faint">
              Asking for one translation per line, in order, is what keeps them lined
              up with the blocks they came from. Lose that and each line gets asked
              about on its own instead — slower, and every line loses the rest of the
              page as context. Asking for a voice — casual, formal, terse — comes
              through; asking the model to reformat what it hands back mostly does
              not, since the answers are held to a fixed shape.
            </p>
          </section>

          <section className="rounded-lg border border-line bg-canvas px-3 py-2.5 text-xs">
            <dl className="space-y-1">
              <div className="flex justify-between gap-4">
                <dt className="text-faint">API</dt>
                <dd className="truncate text-muted">{apiBase}</dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-faint">Models on its Ollama</dt>
                <dd className="truncate text-muted">
                  {models.length > 0 ? models.join(', ') : 'none found'}
                </dd>
              </div>
            </dl>
          </section>
        </div>

        <footer className="flex flex-wrap items-center justify-end gap-2 border-t border-line px-5 py-3">
          <Button
            variant="ghost"
            size="md"
            className="mr-auto"
            onClick={() => setDraft(fallback ?? '')}
            disabled={fallback === null || isDefault}
          >
            Back to the default
          </Button>
          <Button size="md" onClick={onClose}>
            Cancel
          </Button>
          <Button
            variant="primary"
            size="md"
            onClick={() => {
              // Saved as "the API's own" when it is word for word the API's own,
              // so a change there is still picked up later.
              onSave(isDefault || !draft.trim() ? null : draft)
              onClose()
            }}
            disabled={!changed}
          >
            Save
          </Button>
        </footer>
      </div>
    </div>
  )
}
