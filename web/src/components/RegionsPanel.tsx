import {
  DndContext,
  KeyboardSensor,
  PointerSensor,
  closestCenter,
  useSensor,
  useSensors,
} from '@dnd-kit/core'
import type { DragEndEvent } from '@dnd-kit/core'
import { restrictToParentElement, restrictToVerticalAxis } from '@dnd-kit/modifiers'
import {
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import { useEffect, useRef, useState } from 'react'
import type { Analysis, Region } from '../lib/api'
import { UNSURE } from '../lib/api'
import { plural } from '../lib/images'
import { CloseIcon, GripIcon, RestoreIcon } from './icons'
import { Button, FOCUS } from './ui'

type Props = {
  analysis: Analysis
  reading: boolean
  selected: number | null
  onSelect: (index: number | null) => void
  onToggleExcluded: (index: number) => void
  /** A block dragged to a different place in the list. */
  onMove: (from: number, to: number) => void
}

/** What the detector found and what the reader made of it, block by block. */
export function RegionsPanel({
  analysis,
  reading,
  selected,
  onSelect,
  onToggleExcluded,
  onMove,
}: Props) {
  const list = useRef<HTMLUListElement>(null)
  const { regions } = analysis.detection
  const { texts } = analysis
  const excluded = new Set(analysis.excluded)
  const kept = regions.length - excluded.size

  // A drag has to be told apart from a click, or picking a block out by its
  // handle would start one.
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 4 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  )

  useEffect(() => {
    if (selected === null) return
    list.current
      ?.querySelector(`[data-index="${selected}"]`)
      ?.scrollIntoView({ block: 'nearest' })
  }, [selected])

  const dropped = ({ active, over }: DragEndEvent) => {
    if (!over || active.id === over.id) return
    const from = regions.findIndex((region) => region.id === active.id)
    const to = regions.findIndex((region) => region.id === over.id)
    if (from !== -1 && to !== -1) onMove(from, to)
  }

  return (
    <aside className="flex w-full shrink-0 flex-col border-line bg-surface max-lg:h-72 max-lg:border-t lg:w-72 lg:border-l xl:w-96">
      <div className="flex shrink-0 items-start justify-between gap-2 border-b border-line px-4 py-3">
        <div className="min-w-0">
          <h2 className="text-sm font-semibold text-ink">Text</h2>
          <p className="mt-0.5 text-xs text-faint">
            {regions.length === 0
              ? 'No lettering found on this page'
              : reading
                ? `Reading ${plural(regions.length, 'block')}…`
                : excluded.size > 0
                  ? `${plural(kept, 'block')} to clean · ${excluded.size} left alone`
                  : `${plural(regions.length, 'block')}, read by manga-ocr`}
          </p>
        </div>
        {texts && texts.some((text) => text) && (
          <CopyAll texts={texts.filter((text, index) => text && !excluded.has(index))} />
        )}
      </div>

      {regions.length === 0 ? (
        <p className="px-4 py-6 text-sm text-faint">
          The detector saw no lettering here.
        </p>
      ) : (
        <>
          <p className="shrink-0 border-b border-line px-4 py-1.5 text-[11px] text-faint">
            This is the order the page is translated in. Drag one by its handle to
            move it.
          </p>

          <DndContext
            sensors={sensors}
            collisionDetection={closestCenter}
            modifiers={[restrictToVerticalAxis, restrictToParentElement]}
            onDragEnd={dropped}
          >
            <SortableContext
              items={regions.map((region) => region.id)}
              strategy={verticalListSortingStrategy}
            >
              <ul ref={list} className="min-h-0 flex-1 space-y-1.5 overflow-y-auto p-3">
                {regions.map((region, index) => (
                  <Block
                    key={region.id}
                    region={region}
                    index={index}
                    text={texts?.[index] ?? null}
                    reading={reading}
                    excluded={excluded.has(index)}
                    active={selected === index}
                    onSelect={() => onSelect(selected === index ? null : index)}
                    onToggleExcluded={() => onToggleExcluded(index)}
                  />
                ))}
              </ul>
            </SortableContext>
          </DndContext>
        </>
      )}
    </aside>
  )
}

function Block({
  region,
  index,
  text,
  reading,
  excluded,
  active,
  onSelect,
  onToggleExcluded,
}: {
  region: Region
  index: number
  text: string | null
  reading: boolean
  excluded: boolean
  active: boolean
  onSelect: () => void
  onToggleExcluded: () => void
}) {
  const {
    attributes,
    listeners,
    setNodeRef,
    setActivatorNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: region.id })
  const [x0, y0, x1, y1] = region.box
  const unsure = region.confidence < UNSURE

  return (
    <li
      ref={setNodeRef}
      data-index={index}
      style={{
        // Translate rather than the whole transform: a row being sorted should
        // slide past the others, not stretch to their heights.
        transform: CSS.Translate.toString(transform),
        transition,
      }}
      className={`group flex items-stretch gap-1 ${
        isDragging ? 'relative z-10 opacity-80' : ''
      }`}
    >
      <button
        type="button"
        ref={setActivatorNodeRef}
        {...attributes}
        {...listeners}
        title="Drag to reorder"
        aria-label={`Reorder block ${index + 1}`}
        className={`flex w-5 shrink-0 cursor-grab touch-none items-center justify-center rounded-md text-line transition-colors hover:bg-raised hover:text-muted active:cursor-grabbing ${FOCUS}`}
      >
        <GripIcon />
      </button>

      <div className="relative min-w-0 flex-1">
        <button
          type="button"
          onClick={onSelect}
          aria-pressed={active}
          className={`w-full rounded-lg border py-2 pr-9 pl-2.5 text-left transition-colors ${FOCUS} ${
            excluded
              ? 'border-dashed border-line bg-canvas'
              : active
                ? 'border-accent bg-accent/10'
                : 'border-line hover:border-faint hover:bg-raised'
          }`}
        >
          <div className="flex items-baseline gap-2">
            <span className="text-xs font-semibold text-faint tabular-nums">
              {index + 1}
            </span>

            {text === null ? (
              <span className="text-sm text-faint italic">
                {reading ? 'reading…' : 'not read'}
              </span>
            ) : text === '' ? (
              <span className="text-sm text-faint italic">nothing read here</span>
            ) : (
              <p
                lang="ja"
                className={`min-w-0 flex-1 text-sm leading-relaxed select-text ${
                  excluded ? 'text-faint line-through' : 'text-ink'
                }`}
              >
                {text}
              </p>
            )}
          </div>

          <p className="mt-1 text-[11px] text-faint tabular-nums">
            {excluded ? (
              <>
                <span className="font-medium text-muted">left alone</span>
                {/* Still says how sure the detector was: that is usually the
                    reason it is being left alone, and the reason to put it
                    back. */}
                {!region.manual && (
                  <span className={unsure ? 'text-warn' : ''}>
                    {' · '}
                    {Math.round(region.confidence * 100)}%
                  </span>
                )}
              </>
            ) : region.manual ? (
              <span className="font-medium text-accent-lit">added by hand</span>
            ) : (
              <span className={unsure ? 'text-warn' : ''}>
                {Math.round(region.confidence * 100)}%
              </span>
            )}{' '}
            · {x0}, {y0} · {x1 - x0} × {y1 - y0}
          </p>
        </button>

        <button
          type="button"
          onClick={onToggleExcluded}
          title={
            excluded ? 'Clean this block after all' : 'Leave this block alone: do not clean it'
          }
          aria-label={
            excluded
              ? `Clean block ${index + 1} after all`
              : `Leave block ${index + 1} alone`
          }
          className={`absolute top-1.5 right-1.5 rounded-md p-1.5 text-faint transition-colors hover:bg-raised hover:text-ink ${FOCUS}`}
        >
          {excluded ? (
            <RestoreIcon className="size-3.5" />
          ) : (
            <CloseIcon className="size-3.5" />
          )}
        </button>
      </div>
    </li>
  )
}

/** Every block still being cleaned, in reading order, on the clipboard. */
function CopyAll({ texts }: { texts: string[] }) {
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    if (!copied) return
    const timer = setTimeout(() => setCopied(false), 1600)
    return () => clearTimeout(timer)
  }, [copied])

  return (
    <Button
      onClick={() => {
        navigator.clipboard.writeText(texts.join('\n')).then(
          () => setCopied(true),
          () => setCopied(false),
        )
      }}
    >
      {copied ? 'Copied' : 'Copy all'}
    </Button>
  )
}
