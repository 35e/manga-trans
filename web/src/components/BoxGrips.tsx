import type { BoxDrag, Grip } from '../hooks/useBoxDrag'

const GRIPS: { grip: Grip; className: string; cursor: string }[] = [
  { grip: 'n', className: 'top-0 left-1/2 h-1.5 w-6 -translate-x-1/2 -translate-y-1/2', cursor: 'ns-resize' },
  { grip: 's', className: 'bottom-0 left-1/2 h-1.5 w-6 -translate-x-1/2 translate-y-1/2', cursor: 'ns-resize' },
  { grip: 'w', className: 'top-1/2 left-0 h-6 w-1.5 -translate-x-1/2 -translate-y-1/2', cursor: 'ew-resize' },
  { grip: 'e', className: 'top-1/2 right-0 h-6 w-1.5 translate-x-1/2 -translate-y-1/2', cursor: 'ew-resize' },
  { grip: 'se', className: 'right-0 bottom-0 size-2.5 translate-x-1/2 translate-y-1/2', cursor: 'nwse-resize' },
]

/**
 * The handles on a picked-out box: one per edge, and one at the corner for
 * both at once. Shown inside a box that is already positioned, so they sit on
 * its edges.
 */
export function BoxGrips({ drag }: { drag: BoxDrag }) {
  return (
    <>
      {GRIPS.map(({ grip, className, cursor }) => (
        <span
          key={grip}
          role="presentation"
          onPointerDown={drag.grab}
          onPointerMove={drag.move(grip)}
          onPointerUp={drag.release}
          onPointerCancel={drag.release}
          style={{ cursor }}
          className={`absolute z-10 touch-none rounded-xs bg-indigo-500 ring-1 ring-white ${className}`}
        />
      ))}
    </>
  )
}
