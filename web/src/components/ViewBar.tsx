import type { BoardView } from '../hooks/useBoardView'
import { stem } from '../lib/images'
import { Button, Divider, IconButton, Segmented } from './ui'
import { DownloadIcon, ZoomInIcon, ZoomOutIcon } from './icons'

type Props = {
  view: BoardView
  /** The cleaned page as an object URL, once there is one. */
  cleaned: string | null
  name: string
  showCleaned: boolean
  onShowCleaned: (showing: boolean) => void
}

/**
 * The one bar that sits on the page rather than above it: which version is being
 * looked at, and how closely. Original-versus-cleaned belongs here and not in a
 * step of its own — comparing them is looking, done by the same hand that is
 * zooming in to see whether the clean held up.
 */
export function ViewBar({ view, cleaned, name, showCleaned, onShowCleaned }: Props) {
  return (
    <div className="pointer-events-none absolute inset-x-0 bottom-3 flex justify-center px-3">
      <div className="pointer-events-auto flex max-w-full items-center gap-1.5 overflow-x-auto rounded-xl border border-line bg-surface/85 p-1.5 shadow-lg backdrop-blur">
        {cleaned && (
          <>
            <Segmented
              label="Which version of the page to show"
              value={showCleaned ? 'cleaned' : 'original'}
              onChange={(which) => onShowCleaned(which === 'cleaned')}
              options={[
                {
                  value: 'original',
                  label: 'Original',
                  title: 'The page as it came in',
                },
                {
                  value: 'cleaned',
                  label: 'Cleaned',
                  title: 'The page with the lettering hidden',
                },
              ]}
            />
            <a
              href={cleaned}
              download={`${stem(name)}-clean.png`}
              title="Download the cleaned page"
              aria-label="Download the cleaned page"
              className="inline-grid size-7 shrink-0 place-items-center rounded-lg text-muted transition-colors hover:bg-raised hover:text-ink focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
            >
              <DownloadIcon />
            </a>
            <Divider />
          </>
        )}

        <IconButton label="Zoom out" title="Zoom out (−)" onClick={view.zoomOut}>
          <ZoomOutIcon />
        </IconButton>

        <span
          title="Hold ctrl and scroll to zoom. Middle-drag or hold space to move about."
          className="w-14 shrink-0 text-center text-xs font-medium text-muted tabular-nums"
        >
          {Math.round(view.scale * 100)}%
        </span>

        <IconButton label="Zoom in" title="Zoom in (+)" onClick={view.zoomIn}>
          <ZoomInIcon />
        </IconButton>

        <Divider />

        <Button
          onClick={view.fit}
          disabled={view.fitted}
          title="Fit the whole page on the board (0)"
        >
          Fit
        </Button>
        <Button
          onClick={view.actual}
          disabled={view.scale === 1}
          title="Show the page at its own size, pixel for pixel (1)"
        >
          1:1
        </Button>
      </div>
    </div>
  )
}
