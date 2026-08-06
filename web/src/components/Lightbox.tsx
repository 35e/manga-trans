import { useEffect } from 'react'
import type { GalleryImage } from '../lib/images'
import { formatBytes } from '../lib/images'

type Props = {
  image: GalleryImage
  position: { index: number; total: number }
  onClose: () => void
  onStep: (delta: number) => void
  onRemove: () => void
}

export function Lightbox({ image, position, onClose, onStep, onRemove }: Props) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
      if (event.key === 'ArrowLeft') onStep(-1)
      if (event.key === 'ArrowRight') onStep(1)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose, onStep])

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={image.name}
      className="fixed inset-0 z-50 flex flex-col bg-slate-950/90 backdrop-blur-sm"
      onClick={onClose}
    >
      <header
        className="flex items-center justify-between gap-4 px-4 py-3 text-white"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="min-w-0">
          <p className="truncate text-sm font-medium">{image.name}</p>
          <p className="text-xs text-slate-400 tabular-nums">
            {position.index + 1} of {position.total} · {image.width} ×{' '}
            {image.height} · {formatBytes(image.size)}
          </p>
        </div>

        <div className="flex shrink-0 items-center gap-2">
          <button
            type="button"
            onClick={onRemove}
            className="rounded-lg px-3 py-1.5 text-sm font-medium text-slate-300 transition-colors hover:bg-red-600 hover:text-white focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-red-500"
          >
            Delete
          </button>
          <button
            type="button"
            onClick={onClose}
            autoFocus
            aria-label="Close"
            className="rounded-lg px-3 py-1.5 text-sm font-medium text-slate-300 transition-colors hover:bg-white/10 hover:text-white focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
          >
            Close
          </button>
        </div>
      </header>

      <div className="flex min-h-0 flex-1 items-center gap-2 px-2 pb-6">
        {position.total > 1 && (
          <Step direction={-1} onStep={onStep} label="Previous image" />
        )}
        <img
          src={image.url}
          alt={image.name}
          onClick={(event) => event.stopPropagation()}
          className="mx-auto max-h-full min-h-0 max-w-full object-contain"
        />
        {position.total > 1 && (
          <Step direction={1} onStep={onStep} label="Next image" />
        )}
      </div>
    </div>
  )
}

function Step({
  direction,
  onStep,
  label,
}: {
  direction: 1 | -1
  onStep: (delta: number) => void
  label: string
}) {
  return (
    <button
      type="button"
      aria-label={label}
      onClick={(event) => {
        event.stopPropagation()
        onStep(direction)
      }}
      className="shrink-0 rounded-full bg-white/10 p-2 text-white transition-colors hover:bg-white/20 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
    >
      <svg
        aria-hidden="true"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="size-5"
      >
        <path d={direction === -1 ? 'm14 6-6 6 6 6' : 'm10 6 6 6-6 6'} />
      </svg>
    </button>
  )
}
