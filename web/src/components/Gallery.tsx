import type { GalleryFolder, GalleryImage } from '../lib/images'
import { formatBytes, plural } from '../lib/images'
import { FolderIcon, TrashIcon } from './icons'

type Props = {
  images: GalleryImage[]
  folders: GalleryFolder[]
  /** The folder being looked into, or null for everything that was dropped in. */
  open: string | null
  onOpenFolder: (id: string) => void
  activeId: string | null
  onOpen: (id: string) => void
  onRemove: (id: string) => void
  onRemoveFolder: (id: string) => void
}

/**
 * The rail: every page dropped in, small, the one on the board picked out.
 *
 * An archive is a folder here rather than fifty more thumbnails, and the pages
 * inside one are only shown once it has been opened.
 */
export function Gallery({
  images,
  folders,
  open,
  onOpenFolder,
  activeId,
  onOpen,
  onRemove,
  onRemoveFolder,
}: Props) {
  const shown = images.filter((image) =>
    open === null ? image.folder === undefined : image.folder === open,
  )

  const empty =
    open === null ? folders.length === 0 && shown.length === 0 : shown.length === 0

  if (empty) {
    return (
      <p className="px-1 py-8 text-center text-xs text-faint">
        {open === null
          ? 'Pages you drop in show up here. A zip becomes a folder, and so does New folder.'
          : 'Nothing in this folder. Pages dropped in while it is open land here.'}
      </p>
    )
  }

  return (
    <ul className="grid grid-cols-2 gap-2">
      {open === null &&
        folders.map((folder) => (
          <Folder
            key={folder.id}
            folder={folder}
            pages={images.filter((image) => image.folder === folder.id)}
            onOpen={() => onOpenFolder(folder.id)}
            onRemove={() => onRemoveFolder(folder.id)}
          />
        ))}

      {shown.map((image, index) => (
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
 * One archive, shown as its first page under a stack.
 *
 * A double-click opens it, as a folder anywhere else does — and so does a single
 * click, since a card that ignores one reads as broken and a touch has no double.
 */
function Folder({
  folder,
  pages,
  onOpen,
  onRemove,
}: {
  folder: GalleryFolder
  pages: GalleryImage[]
  onOpen: () => void
  onRemove: () => void
}) {
  const cover = pages[0]
  const total = pages.reduce((sum, page) => sum + page.size, 0)

  return (
    <li className="group relative">
      <button
        type="button"
        onClick={onOpen}
        onDoubleClick={onOpen}
        title={`${folder.name} — ${plural(pages.length, 'page')}, ${formatBytes(total)}. Open it.`}
        className="block w-full rounded-xl border border-line bg-surface p-1 text-left transition-colors hover:border-faint hover:bg-raised focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
      >
        {/* The lip of the pages behind, so a folder does not read as a page. */}
        <span className="relative block">
          <span
            aria-hidden="true"
            className="absolute inset-x-1.5 -top-0.5 block h-1 rounded-t bg-line"
          />
          <span className="relative block overflow-hidden rounded-lg bg-canvas">
            {cover ? (
              <img
                src={cover.url}
                alt=""
                loading="lazy"
                className="block aspect-[3/4] w-full object-contain opacity-70"
              />
            ) : (
              <span className="block aspect-[3/4] w-full" />
            )}
            <span className="absolute inset-0 grid place-items-center bg-black/45">
              <FolderIcon className="size-7 text-white/90" />
            </span>
            <span className="absolute bottom-1 left-1 rounded bg-black/60 px-1 text-[10px] leading-4 font-medium text-white tabular-nums backdrop-blur-sm">
              {pages.length}
            </span>
          </span>
        </span>

        <span className="mt-1 block truncate px-0.5 pb-0.5 text-[11px] text-muted">
          {folder.name}
        </span>
      </button>

      <Delete onClick={onRemove} what={`the folder ${folder.name}`} />
    </li>
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

      <Delete onClick={onRemove} what={image.name} />
    </li>
  )
}

/** The corner button that throws a card away, on the card it is over. */
function Delete({ onClick, what }: { onClick: () => void; what: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={`Delete ${what}`}
      title={`Delete ${what}`}
      className="absolute top-2 right-2 rounded-md bg-black/60 p-1 text-white/80 opacity-0 backdrop-blur-sm transition group-hover:opacity-100 hover:bg-danger hover:text-white focus-visible:opacity-100 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-danger max-sm:opacity-100"
    >
      <TrashIcon className="size-3.5" />
    </button>
  )
}
