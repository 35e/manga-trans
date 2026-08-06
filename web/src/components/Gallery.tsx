import type { GalleryImage } from '../lib/images'
import { formatBytes } from '../lib/images'

type Props = {
  images: GalleryImage[]
  onRemove: (id: string) => void
  onOpen: (id: string) => void
}

export function Gallery({ images, onRemove, onOpen }: Props) {
  if (images.length === 0) return <EmptyState />

  return (
    <ul className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
      {images.map((image) => (
        <Card
          key={image.id}
          image={image}
          onRemove={() => onRemove(image.id)}
          onOpen={() => onOpen(image.id)}
        />
      ))}
    </ul>
  )
}

function Card({
  image,
  onRemove,
  onOpen,
}: {
  image: GalleryImage
  onRemove: () => void
  onOpen: () => void
}) {
  return (
    <li className="group relative overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm transition-shadow hover:shadow-md dark:border-white/10 dark:bg-white/5">
      <button
        type="button"
        onClick={onOpen}
        className="block w-full cursor-zoom-in bg-slate-100 focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-indigo-500 dark:bg-slate-950/40"
        aria-label={`View ${image.name}`}
      >
        <img
          src={image.url}
          alt={image.name}
          loading="lazy"
          className="aspect-3/4 w-full object-contain transition-transform duration-200 group-hover:scale-[1.02]"
        />
      </button>

      <button
        type="button"
        onClick={onRemove}
        aria-label={`Delete ${image.name}`}
        className="absolute top-2 right-2 rounded-lg bg-white/85 p-1.5 text-slate-600 opacity-0 shadow-sm backdrop-blur transition group-hover:opacity-100 hover:bg-red-600 hover:text-white focus-visible:opacity-100 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-red-500 max-sm:opacity-100 dark:bg-slate-900/80 dark:text-slate-300"
      >
        <svg
          aria-hidden="true"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          strokeLinecap="round"
          className="size-4"
        >
          <path d="M4 7h16M10 11v6M14 11v6M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
        </svg>
      </button>

      <div className="border-t border-slate-200 px-3 py-2 dark:border-white/10">
        <p
          className="truncate text-sm font-medium text-slate-900 dark:text-white"
          title={image.name}
        >
          {image.name}
        </p>
        <p className="mt-0.5 text-xs text-slate-500 tabular-nums dark:text-slate-400">
          {image.width} × {image.height} · {formatBytes(image.size)}
        </p>
      </div>
    </li>
  )
}

function EmptyState() {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white/60 px-6 py-16 text-center dark:border-white/10 dark:bg-white/5">
      <svg
        aria-hidden="true"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
        className="mx-auto size-10 text-slate-300 dark:text-slate-600"
      >
        <rect x="3" y="4" width="18" height="16" rx="2" />
        <path d="m4 16 4.5-4.5a2 2 0 0 1 2.8 0L16 16" />
        <path d="m14 14 1.5-1.5a2 2 0 0 1 2.8 0L20 14" />
        <circle cx="9" cy="9" r="1.2" />
      </svg>
      <p className="mt-3 text-sm font-medium text-slate-900 dark:text-white">
        No pages yet
      </p>
      <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
        Everything you drop in shows up here.
      </p>
    </div>
  )
}
