import type { GalleryImage } from '../lib/images'
import { formatBytes } from '../lib/images'
import { TrashIcon } from './icons'

type Props = {
  images: GalleryImage[]
  activeId: string | null
  onOpen: (id: string) => void
  onRemove: (id: string) => void
}

/** The rail: every page dropped in, small, the one on the board picked out. */
export function Gallery({ images, activeId, onOpen, onRemove }: Props) {
  if (images.length === 0) {
    return (
      <p className="px-1 py-8 text-center text-xs text-faint">
        Pages you drop in show up here.
      </p>
    )
  }

  return (
    <ul className="grid grid-cols-2 gap-2">
      {images.map((image, index) => (
        <Thumb
          key={image.id}
          image={image}
          number={index + 1}
          active={image.id === activeId}
          onOpen={() => onOpen(image.id)}
          onRemove={() => onRemove(image.id)}
        />
      ))}
    </ul>
  )
}

/**
 * One page in the rail. The picture sits in a box of its own inside the card's
 * border rather than against it, or a page of the wrong shape rides over the
 * rounded corners; the two radii are concentric so the corner reads as one curve.
 */
function Thumb({
  image,
  number,
  active,
  onOpen,
  onRemove,
}: {
  image: GalleryImage
  number: number
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
        className={`block w-full rounded-xl border p-1 text-left transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent ${
          active
            ? 'border-accent bg-accent/10'
            : 'border-line bg-surface hover:border-faint hover:bg-raised'
        }`}
      >
        <span className="relative block overflow-hidden rounded-lg bg-canvas">
          <img
            src={image.url}
            alt={image.name}
            loading="lazy"
            className="block aspect-[3/4] w-full object-contain"
          />
          <span className="absolute top-1 left-1 rounded bg-black/60 px-1 text-[10px] leading-4 font-medium text-white tabular-nums backdrop-blur-sm">
            {number}
          </span>
        </span>

        <span
          className={`mt-1 block truncate px-0.5 pb-0.5 text-[11px] ${
            active ? 'text-ink' : 'text-muted'
          }`}
        >
          {image.name}
        </span>
      </button>

      <button
        type="button"
        onClick={onRemove}
        aria-label={`Delete ${image.name}`}
        title={`Delete ${image.name}`}
        className="absolute top-2 right-2 rounded-md bg-black/60 p-1 text-white/80 opacity-0 backdrop-blur-sm transition group-hover:opacity-100 hover:bg-danger hover:text-white focus-visible:opacity-100 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-danger max-sm:opacity-100"
      >
        <TrashIcon className="size-3.5" />
      </button>
    </li>
  )
}
