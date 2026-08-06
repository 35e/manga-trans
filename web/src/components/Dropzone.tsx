import { useRef } from 'react'

type Props = {
  onFiles: (files: FileList | File[] | null) => void
  dragging: boolean
  busy: boolean
}

export function Dropzone({ onFiles, dragging, busy }: Props) {
  const input = useRef<HTMLInputElement>(null)

  return (
    <div
      className={`relative rounded-2xl border-2 border-dashed p-10 text-center transition-colors ${
        dragging
          ? 'border-indigo-500 bg-indigo-50 dark:border-indigo-400 dark:bg-indigo-500/10'
          : 'border-slate-300 bg-white hover:border-slate-400 dark:border-white/15 dark:bg-white/5 dark:hover:border-white/25'
      }`}
    >
      <input
        ref={input}
        type="file"
        accept="image/*"
        multiple
        className="sr-only"
        onChange={(event) => {
          onFiles(event.target.files)
          // Let the same file be chosen twice in a row.
          event.target.value = ''
        }}
      />

      <svg
        aria-hidden="true"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        className={`mx-auto size-10 transition-colors ${
          dragging
            ? 'text-indigo-500 dark:text-indigo-400'
            : 'text-slate-400 dark:text-slate-500'
        }`}
      >
        <path d="M12 16V4m0 0L8 8m4-4 4 4" />
        <path d="M3 15v3a3 3 0 0 0 3 3h12a3 3 0 0 0 3-3v-3" />
      </svg>

      <p className="mt-4 text-base font-medium text-slate-900 dark:text-white">
        {dragging ? 'Drop to add them' : 'Drag pages in from anywhere'}
      </p>
      <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
        or paste from the clipboard
      </p>

      <button
        type="button"
        onClick={() => input.current?.click()}
        disabled={busy}
        className="mt-5 inline-flex items-center gap-2 rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-slate-700 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-500 disabled:opacity-60 dark:bg-white dark:text-slate-900 dark:hover:bg-slate-200"
      >
        {busy ? 'Reading…' : 'Browse files'}
      </button>
    </div>
  )
}
