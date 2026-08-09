import { useEffect, useState } from 'react'
import type { LibraryNotice } from '../hooks/useImageLibrary'
import type { GalleryImage } from '../lib/images'
import { formatBytes, plural } from '../lib/images'
import { Dropzone } from './Dropzone'
import { Gallery } from './Gallery'

type Props = {
  images: GalleryImage[]
  activeId: string | null
  onOpen: (id: string) => void
  onRemove: (id: string) => void
  onFiles: (files: FileList | File[] | null) => void
  dragging: boolean
  busy: boolean
  notice: LibraryNotice | null
  onDismissNotice: () => void
  onClearAll: () => void
}

/** The rail down the side: what to drop pages into, and every page dropped in. */
export function Sidebar({
  images,
  activeId,
  onOpen,
  onRemove,
  onFiles,
  dragging,
  busy,
  notice,
  onDismissNotice,
  onClearAll,
}: Props) {
  const total = images.reduce((sum, image) => sum + image.size, 0)

  return (
    <aside className="flex shrink-0 flex-col border-line bg-surface max-lg:h-64 max-lg:border-b lg:w-60 lg:border-r xl:w-72">
      <div className="shrink-0 p-3">
        <Dropzone onFiles={onFiles} dragging={dragging} busy={busy} />
      </div>

      {notice && (
        <div className="mx-3 mb-3 flex shrink-0 items-start justify-between gap-2 rounded-lg border border-warn/30 bg-warn/10 px-2.5 py-2 text-[11px] leading-snug text-warn">
          <span>{notice.text}</span>
          <button
            type="button"
            onClick={onDismissNotice}
            aria-label="Dismiss"
            className="shrink-0 font-semibold hover:underline"
          >
            ✕
          </button>
        </div>
      )}

      <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-3">
        <Gallery
          images={images}
          activeId={activeId}
          onOpen={onOpen}
          onRemove={onRemove}
        />
      </div>

      {images.length > 0 && (
        <div className="flex shrink-0 items-center justify-between gap-2 border-t border-line px-3 py-2 text-[11px] text-faint">
          <span className="tabular-nums">
            {plural(images.length, 'page')} · {formatBytes(total)}
          </span>
          <ClearAll onClear={onClearAll} />
        </div>
      )}
    </aside>
  )
}

/** Two taps to empty the gallery, without a browser dialog. */
function ClearAll({ onClear }: { onClear: () => void }) {
  const [armed, setArmed] = useState(false)

  useEffect(() => {
    if (!armed) return
    const timer = setTimeout(() => setArmed(false), 4000)
    return () => clearTimeout(timer)
  }, [armed])

  return (
    <button
      type="button"
      onClick={() => {
        if (armed) onClear()
        setArmed(!armed)
      }}
      onBlur={() => setArmed(false)}
      className={`shrink-0 rounded-md px-2 py-1 font-medium transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-danger ${
        armed ? 'bg-danger text-white' : 'text-faint hover:bg-danger/15 hover:text-danger'
      }`}
    >
      {armed ? 'Sure?' : 'Clear all'}
    </button>
  )
}
