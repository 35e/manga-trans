import type { GalleryImage } from '../lib/images'
import { formatBytes } from '../lib/images'

type Props = {
  images: GalleryImage[]
  activeId: string | null
  onOpen: (id: string) => void
  onRemove: (id: string) => void
}

/** The rail: every page dropped in, small, the one on the board ringed. */
export function Gallery({ images, activeId, onOpen, onRemove }: Props) {
  if (images.length === 0) {
    return (
      <p className="px-1 py-6 text-center text-xs text-slate-500 dark:text-slate-400">
        Pages you drop in show up here.
      </p>
    )
  }

  return (
    <ul className="grid grid-cols-2 gap-2">
      {images.map((image) => (
        <Thumb
          key={image.id}
          image={image}
          active={image.id === activeId}
          onOpen={() => onOpen(image.id)}
          onRemove={() => onRemove(image.id)}
        />
      ))}
    </ul>
  )
}

function Thumb({
  image,
  active,
  onOpen,
  onRemove,
}: {
  image: GalleryImage
  active: boolean
  onOpen: () => void
  onRemove: () => void
}) {
  return (
    <li className="group relative">
      <button
        type="button"
        onClick={onOpen}
        aria-current={active}
        title={`${image.name} — ${image.width} × ${image.height}, ${formatBytes(image.size)}`}
        className={`block w-full overflow-hidden rounded-lg bg-slate-100 ring-2 transition focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-500 dark:bg-slate-900 ${
          active
            ? 'ring-indigo-500'
            : 'ring-transparent hover:ring-slate-300 dark:hover:ring-white/20'
        }`}
      >
        <img
          src={image.url}
          alt={image.name}
          loading="lazy"
          className="aspect-[3/4] w-full object-contain"
        />
        <span className="block truncate px-1.5 py-1 text-[11px] text-slate-600 dark:text-slate-400">
          {image.name}
        </span>
      </button>

      <button
        type="button"
        onClick={onRemove}
        aria-label={`Delete ${image.name}`}
        className="absolute top-1 right-1 rounded-md bg-white/85 p-1 text-slate-600 opacity-0 shadow-sm backdrop-blur transition group-hover:opacity-100 hover:bg-red-600 hover:text-white focus-visible:opacity-100 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-red-500 max-sm:opacity-100 dark:bg-slate-900/85 dark:text-slate-300"
      >
        <svg
          aria-hidden="true"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          strokeLinecap="round"
          className="size-3.5"
        >
          <path d="M4 7h16M10 11v6M14 11v6M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
        </svg>
      </button>
    </li>
  )
}
