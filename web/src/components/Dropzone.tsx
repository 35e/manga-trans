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
      className={`flex w-full items-center gap-3 rounded-xl border border-dashed px-3 py-3 text-left transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:opacity-60 ${
        dragging
          ? 'border-accent bg-accent/10'
          : 'border-line hover:border-faint hover:bg-raised'
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
        className={`size-5 shrink-0 transition-colors ${
          dragging ? 'text-accent-lit' : 'text-faint'
        }`}
      >
        <path d="M12 16V4m0 0L8 8m4-4 4 4" />
        <path d="M3 15v3a3 3 0 0 0 3 3h12a3 3 0 0 0 3-3v-3" />
      </svg>

      <span className="min-w-0">
        <span className="block truncate text-xs font-medium text-ink">
          {busy ? 'Reading…' : dragging ? 'Drop to add' : 'Drop pages or browse'}
        </span>
        <span className="block truncate text-[11px] text-faint">
          a zip works too, and so does pasting
        </span>
      </span>
    </button>
  )
}
