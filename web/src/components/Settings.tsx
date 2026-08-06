import { useEffect, useRef, useState } from 'react'

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
export function Settings({
  onClose,
  prompt,
  fallback,
  onSave,
  apiBase,
  models,
}: Props) {
  const [draft, setDraft] = useState(prompt ?? fallback ?? '')
  const box = useRef<HTMLTextAreaElement>(null)

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
      className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-slate-950/50 p-4 backdrop-blur-sm sm:p-8"
    >
      <div
        onClick={(event) => event.stopPropagation()}
        className="w-full max-w-2xl rounded-2xl border border-slate-200 bg-white shadow-2xl dark:border-white/10 dark:bg-slate-950"
      >
        <header className="flex items-start justify-between gap-4 border-b border-slate-200 px-5 py-4 dark:border-white/10">
          <div>
            <h2 className="text-base font-semibold text-slate-900 dark:text-white">
              Settings
            </h2>
            <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
              Kept in this browser. The API stores nothing and is sent these with
              each page.
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close settings"
            className="rounded-lg px-2 py-1 text-sm font-medium text-slate-500 transition-colors hover:bg-slate-100 hover:text-slate-900 dark:text-slate-400 dark:hover:bg-white/10 dark:hover:text-white"
          >
            ✕
          </button>
        </header>

        <div className="space-y-5 px-5 py-5">
          <section>
            <label
              htmlFor="prompt"
              className="text-sm font-medium text-slate-900 dark:text-white"
            >
              What the model is told
            </label>
            <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
              The system prompt every translation is asked with.{' '}
              <code className="rounded bg-slate-100 px-1 py-0.5 dark:bg-white/10">
                {'{target}'}
              </code>{' '}
              is replaced by the language you are translating into.
            </p>

            <textarea
              id="prompt"
              ref={box}
              value={draft}
              rows={9}
              spellCheck={false}
              onChange={(event) => setDraft(event.target.value)}
              placeholder={fallback ?? 'Loading the default…'}
              className="mt-2 w-full resize-y rounded-lg border border-slate-300 bg-white px-3 py-2 font-mono text-xs leading-relaxed text-slate-900 focus-visible:outline-2 focus-visible:outline-indigo-500 dark:border-white/15 dark:bg-slate-900 dark:text-white"
            />

            <p className="mt-1.5 text-xs text-slate-400 dark:text-slate-500">
              Asking for one translation per line, in order, is what keeps them
              lined up with the blocks they came from. Lose that and each line
              gets asked about on its own instead — slower, and every line loses
              the rest of the page as context. Asking for a voice — casual,
              formal, terse — comes through; asking the model to reformat what it
              hands back mostly does not, since the answers are held to a fixed
              shape.
            </p>
          </section>

          <section className="rounded-lg bg-slate-50 px-3 py-2.5 text-xs dark:bg-white/5">
            <dl className="space-y-1">
              <div className="flex justify-between gap-4">
                <dt className="text-slate-500 dark:text-slate-400">API</dt>
                <dd className="truncate text-slate-700 dark:text-slate-300">
                  {apiBase}
                </dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-slate-500 dark:text-slate-400">
                  Models on its Ollama
                </dt>
                <dd className="truncate text-slate-700 dark:text-slate-300">
                  {models.length > 0 ? models.join(', ') : 'none found'}
                </dd>
              </div>
            </dl>
          </section>
        </div>

        <footer className="flex flex-wrap items-center justify-end gap-2 border-t border-slate-200 px-5 py-3 dark:border-white/10">
          <button
            type="button"
            onClick={() => setDraft(fallback ?? '')}
            disabled={fallback === null || isDefault}
            className="mr-auto rounded-lg px-3 py-1.5 text-sm font-medium text-slate-500 transition-colors hover:bg-slate-100 hover:text-slate-900 disabled:opacity-40 dark:text-slate-400 dark:hover:bg-white/10 dark:hover:text-white"
          >
            Back to the default
          </button>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 transition-colors hover:bg-slate-50 dark:border-white/15 dark:text-slate-200 dark:hover:bg-white/10"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={() => {
              // Saved as "the API's own" when it is word for word the API's
              // own, so a change there is still picked up later.
              onSave(isDefault || !draft.trim() ? null : draft)
              onClose()
            }}
            disabled={!changed}
            className="rounded-lg bg-indigo-600 px-3.5 py-1.5 text-sm font-medium text-white transition-colors hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Save
          </button>
        </footer>
      </div>
    </div>
  )
}
