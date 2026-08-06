import { useRef } from 'react'

type Props = {
  onFiles: (files: FileList | File[] | null) => void
  dragging: boolean
  busy: boolean
}

export function Dropzone({ onFiles, dragging, busy }: Props) {
  const input = useRef<HTMLInputElement>(null)

  return (
    <button
      type="button"
      onClick={() => input.current?.click()}
      disabled={busy}
      className={`w-full rounded-xl border-2 border-dashed px-3 py-4 text-center transition-colors ${
        dragging
          ? 'border-indigo-500 bg-indigo-50 dark:border-indigo-400 dark:bg-indigo-500/10'
          : 'border-slate-300 hover:border-slate-400 hover:bg-slate-50 dark:border-white/15 dark:hover:border-white/25 dark:hover:bg-white/5'
      }`}
    >
      <input
        ref={input}
        type="file"
        accept="image/*,.zip,.cbz,application/zip"
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
        className={`mx-auto size-6 transition-colors ${
          dragging
            ? 'text-indigo-500 dark:text-indigo-400'
            : 'text-slate-400 dark:text-slate-500'
        }`}
      >
        <path d="M12 16V4m0 0L8 8m4-4 4 4" />
        <path d="M3 15v3a3 3 0 0 0 3 3h12a3 3 0 0 0 3-3v-3" />
      </svg>

      <span className="mt-2 block text-xs font-medium text-slate-900 dark:text-white">
        {busy ? 'Reading…' : dragging ? 'Drop to add' : 'Drop pages or browse'}
      </span>
      <span className="mt-0.5 block text-[11px] text-slate-500 dark:text-slate-400">
        a zip of them works too, and so does pasting
      </span>
    </button>
  )
}
