import { useEffect, useRef, useState } from 'react'
import { useBlockKeys, useZoomKeys } from '../hooks/useBoardKeys'
import { useBoardView } from '../hooks/useBoardView'
import type { Analysis, Box, Fill, Language, Stage, Tool } from '../lib/api'
import type { GalleryImage } from '../lib/images'
import type { Lines } from '../lib/lettering'
import type { Brush, Mask } from '../lib/mask'
import { mark } from '../lib/mask'
import { toClean } from '../lib/regions'
import { DrawRegion } from './DrawRegion'
import { InspectTools } from './InspectTools'
import { MaskCanvas } from './MaskCanvas'
import { MaskTools } from './MaskTools'
import { RegionsLayer } from './RegionsLayer'
import { TranslateTools } from './TranslateTools'
import { TranslationLayer } from './TranslationLayer'
import { ViewBar } from './ViewBar'
import { PageIcon } from './icons'
import { Button, Segmented, Spinner } from './ui'

export type Inspecting = {
  languages: Language[]
  language: string
  onLanguage: (code: string) => void
  onAddRegion: (box: Box) => void
  onRegionBox: (index: number, box: Box) => void
  onRegionSettled: (index: number, was: Box) => void
  onToggleExcluded: (index: number) => void
}

export type Masking = {
  onClean: (marks: Blob) => void
  letters: ImageBitmap | null
  onTrace: () => Promise<ImageBitmap | null>
  spread: number
  onSpread: (spread: number) => void
  fill: Fill
  onFill: (fill: Fill) => void
}

export type Translating = {
  models: string[]
  model: string
  onModel: (model: string) => void
  target: string
  onTarget: (target: string) => void
  onTranslate: () => void
  lettering: Lines
  onBox: (index: number, box: Box) => void
  onTurn: (index: number, angle: number) => void
  onSize: (index: number, by: number) => void
  onApply: () => void
  applying: boolean
  note: string | null
}

type Props = {
  image: GalleryImage | null
  analysis: Analysis | null
  mask: Mask | null
  cleaned: string | null
  stage: Stage | null
  error: string | null
  selected: number | null
  onSelect: (index: number | null) => void
  tool: Tool
  onTool: (tool: Tool) => void
  runningFolder: boolean
  showCleaned: boolean
  onShowCleaned: (showing: boolean) => void
  onRunAll: () => void
  onDetect: () => void
  inspecting: Inspecting
  masking: Masking
  translating: Translating
}

const LABELS: Record<Stage, string> = {
  detecting: 'Detecting…',
  reading: 'Reading…',
  tracing: 'Tracing…',
  cleaning: 'Cleaning…',
  translating: 'Translating…',
  surveying: 'Reading the chapter…',
}

export function Board({
  image,
  analysis,
  mask,
  cleaned,
  stage,
  error,
  selected,
  onSelect,
  tool,
  onTool,
  runningFolder,
  showCleaned,
  onShowCleaned,
  onRunAll,
  onDetect,
  inspecting,
  masking,
  translating,
}: Props) {
  const surface = useRef<HTMLDivElement>(null)
  const view = useBoardView(surface, image)

  const [brush, setBrush] = useState<Brush>({ radius: 16, erase: false })
  const [showBoxes, setShowBoxes] = useState(true)
  const [adding, setAdding] = useState(false)
  const [, setEdits] = useState(0)
  const edited = () => setEdits((count) => count + 1)

  const busy = stage !== null || runningFolder
  const waiting = runningFolder ? 'wait for the folder being run to finish' : undefined
  const brushing = tool === 'mask' && !showCleaned
  const marked = Boolean(mask && !mask.empty)

  const read = analysis?.texts != null
  const lettered = translating.lettering.some(Boolean)

  useZoomKeys(view, image !== null)
  useBlockKeys({
    tool,
    selected,
    lettering: translating.lettering,
    onToggleExcluded: inspecting.onToggleExcluded,
    onSize: translating.onSize,
    onTurn: translating.onTurn,
  })

  const analysisNow = useRef(analysis)
  analysisNow.current = analysis
  const lettersNow = useRef(masking.letters)
  lettersNow.current = masking.letters
  const traceNow = useRef(masking.onTrace)
  traceNow.current = masking.onTrace

  const markBlocks = async () => {
    const found = analysisNow.current
    if (!mask || !found) return
    const letters = lettersNow.current ?? (await traceNow.current())
    if (!letters) return
    mark(mask, toClean(found), letters)
    edited()
  }

  useEffect(() => {
    if (!brushing || !mask || !mask.empty || !analysisNow.current) return
    let dropped = false

    void (async () => {
      const letters = lettersNow.current ?? (await traceNow.current())
      if (dropped || !mask.empty) return
      const found = analysisNow.current
      if (found) {
        mark(mask, toClean(found), letters)
        edited()
      }
    })()

    return () => {
      dropped = true
    }
  }, [brushing, mask, analysis?.detection])

  const canMark = Boolean(mask && analysis && toClean(analysis).length > 0)

  return (
    <section className="flex min-h-0 min-w-0 flex-1 flex-col bg-canvas">
      <header className="flex shrink-0 flex-wrap items-center gap-x-4 gap-y-2 border-b border-line bg-surface px-4 py-2.5">
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium text-ink">
            {image ? image.name : 'Nothing on the board'}
          </p>
          <p className="text-xs text-faint tabular-nums">
            {image ? `${image.width} × ${image.height}` : 'Pick a page from the gallery'}
          </p>
        </div>

        {image && (
          <>
            <Segmented<Tool>
              label="Tool"
              value={tool}
              onChange={onTool}
              options={[
                {
                  value: 'boxes',
                  label: 'Boxes',
                  title: 'Find and adjust the text blocks',
                },
                { value: 'mask', label: 'Mask', title: 'Brush over what gets hidden' },
                { value: 'text', label: 'Text', title: 'Edit the translated lettering' },
              ]}
            />

            <div className="flex shrink-0 items-center gap-2">
              <Action
                onClick={onRunAll}
                disabled={busy}
                stage={stage}
                title={waiting ?? 'Read the page, translate it, then clean it'}
              >
                Translate
              </Action>

              {lettered && (
                <Button
                  size="md"
                  onClick={translating.onApply}
                  disabled={busy || translating.applying}
                  title="Set the lettering into the page and save it"
                >
                  {translating.applying ? 'Applying…' : 'Apply to image'}
                </Button>
              )}
            </div>
          </>
        )}
      </header>

      {tool === 'boxes' && image && (
        <InspectTools
          offered={inspecting.languages}
          language={inspecting.language}
          onLanguage={inspecting.onLanguage}
          found={analysis !== null}
          showBoxes={showBoxes}
          onShowBoxes={setShowBoxes}
          adding={adding}
          onAdding={setAdding}
          onDetect={onDetect}
          busy={busy}
        />
      )}

      {brushing && (
        <MaskTools
          brush={brush}
          onBrush={setBrush}
          onMarkLetters={() => void markBlocks()}
          canMark={canMark}
          tracing={stage === 'tracing'}
          onClear={() => {
            mask?.clear()
            edited()
          }}
          canClear={marked}
          spread={masking.spread}
          onSpread={masking.onSpread}
          fill={masking.fill}
          onFill={masking.onFill}
          note={read ? null : 'find the text first, or brush the page by hand'}
          onClean={() => {
            if (mask && !mask.empty) void mask.toBlob().then(masking.onClean)
          }}
          canClean={marked}
          cleaned={Boolean(cleaned)}
          busy={busy}
        />
      )}

      {tool === 'text' && image && (
        <TranslateTools
          models={translating.models}
          model={translating.model}
          onModel={translating.onModel}
          target={translating.target}
          onTarget={translating.onTarget}
          onTranslate={translating.onTranslate}
          canTranslate={Boolean(translating.model) && read && !busy}
          lettered={lettered}
          note={translating.note}
        />
      )}

      {error && (
        <p
          role="alert"
          className="shrink-0 border-b border-danger/40 bg-danger/10 px-4 py-2 text-sm text-danger"
        >
          {error}
        </p>
      )}

      <div className="relative min-h-0 flex-1">
        <div
          ref={surface}
          onPointerDown={(event) => {
            if (selected === null) return
            if (!(event.target as Element).closest('[data-box]')) onSelect(null)
          }}
          className={`absolute inset-0 isolate overflow-auto overscroll-contain ${
            view.panning ? 'cursor-grabbing' : ''
          }`}
        >
          <div
            className="board-mat relative"
            style={
              {
                width: view.content.width,
                height: view.content.height,
                '--mat-grid': `${view.grid}px`,
              } as React.CSSProperties
            }
          >
            {image && view.page && (
              <div
                className="absolute ring-1 ring-white/10 shadow-[0_8px_40px_rgb(0_0_0/0.55)]"
                style={{
                  left: view.origin.x,
                  top: view.origin.y,
                  width: view.page.width,
                  height: view.page.height,
                }}
              >
                <img
                  src={showCleaned && cleaned ? cleaned : image.url}
                  alt={image.name}
                  className="block h-full w-full select-none"
                  style={{ imageRendering: view.crisp ? 'pixelated' : undefined }}
                  draggable={false}
                />

                {tool === 'boxes' && analysis && !showCleaned && showBoxes && (
                  <RegionsLayer
                    analysis={analysis}
                    scale={view.scale}
                    selected={selected}
                    onSelect={onSelect}
                    onBox={inspecting.onRegionBox}
                    onSettled={inspecting.onRegionSettled}
                  />
                )}

                {tool === 'boxes' && adding && (
                  <DrawRegion page={image} onAdd={inspecting.onAddRegion} />
                )}

                {tool === 'text' && (
                  <TranslationLayer
                    page={image}
                    scale={view.scale}
                    lettering={translating.lettering}
                    selected={selected}
                    onSelect={onSelect}
                    onBox={translating.onBox}
                    onTurn={translating.onTurn}
                  />
                )}

                {brushing && mask && (
                  <MaskCanvas
                    page={image}
                    mask={mask}
                    brush={brush}
                    panning={view.panning}
                    onStroke={edited}
                  />
                )}
              </div>
            )}
          </div>
        </div>

        {!image && <BoardEmpty />}

        {image && (
          <ViewBar
            view={view}
            cleaned={cleaned}
            name={image.name}
            showCleaned={showCleaned}
            onShowCleaned={onShowCleaned}
          />
        )}
      </div>
    </section>
  )
}

function Action({
  onClick,
  disabled,
  stage,
  title,
  children,
}: {
  onClick: () => void
  disabled: boolean
  stage: Stage | null
  title?: string
  children: React.ReactNode
}) {
  return (
    <Button variant="primary" size="md" onClick={onClick} disabled={disabled} title={title}>
      {stage ? (
        <>
          <Spinner />
          {LABELS[stage]}
        </>
      ) : (
        children
      )}
    </Button>
  )
}

function BoardEmpty() {
  return (
    <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
      <div className="text-center">
        <PageIcon className="mx-auto size-12 text-line" />
        <p className="mt-4 text-sm font-medium text-muted">
          Click a page in the gallery to put it on the board
        </p>
        <p className="mt-1 text-sm text-faint">
          Then: find the text, hide it, letter it back in.
        </p>
      </div>
    </div>
  )
}
