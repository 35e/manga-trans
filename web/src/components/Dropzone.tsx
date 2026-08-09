import { useRef } from 'react'
import { UploadIcon } from './icons'

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

      <UploadIcon
        className={`size-5 shrink-0 transition-colors ${
          dragging ? 'text-accent-lit' : 'text-faint'
        }`}
      />

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
