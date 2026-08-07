import type { BoxDrag, Grip } from '../hooks/useBoxDrag'

/**
 * Where each handle sits, and how far around it the pointer still catches it.
 *
 * The element is the target and is drawn nothing at all; what shows is the
 * small square inside it. So the handles stay out of the way of the page — a
 * box with lettering under it is meant to be read through — while still being
 * caught along the whole edge rather than only on the few pixels drawn.
 */
const GRIPS: { grip: Grip; className: string; cursor: string }[] = [
  { grip: 'n', className: 'top-0 left-1/2 h-3 w-6 -translate-x-1/2 -translate-y-1/2', cursor: 'ns-resize' },
  { grip: 's', className: 'bottom-0 left-1/2 h-3 w-6 -translate-x-1/2 translate-y-1/2', cursor: 'ns-resize' },
  { grip: 'w', className: 'top-1/2 left-0 h-6 w-3 -translate-x-1/2 -translate-y-1/2', cursor: 'ew-resize' },
  { grip: 'e', className: 'top-1/2 right-0 h-6 w-3 translate-x-1/2 -translate-y-1/2', cursor: 'ew-resize' },
  { grip: 'se', className: 'right-0 bottom-0 size-3.5 translate-x-1/2 translate-y-1/2', cursor: 'nwse-resize' },
]

/**
 * The handles on a picked-out box: one per edge, one at the corner for both at
 * once, and — where what is in the box can be turned — a round one standing off
 * the top edge. Shown inside a box that is already positioned, so they sit on
 * its edges.
 */
export function BoxGrips({ drag }: { drag: BoxDrag }) {
  return (
    <>
      {drag.turnable && (
        <span
          role="presentation"
          title="Drag to turn. Hold shift for 15° at a time."
          onPointerDown={drag.grab}
          onPointerMove={drag.spin}
          onPointerUp={drag.release}
          onPointerCancel={drag.release}
          style={{ cursor: 'grab' }}
          className="absolute bottom-full left-1/2 z-10 grid size-5 -translate-x-1/2 touch-none place-items-center"
        >
          {/* Round, where the ones that resize are square: the shape is what
              says which it is, since both are too small to say anything else. */}
          <span className="pointer-events-none size-1.5 rounded-full bg-white ring-1 ring-indigo-500" />
        </span>
      )}

      {GRIPS.map(({ grip, className, cursor }) => (
        <span
          key={grip}
          role="presentation"
          onPointerDown={drag.grab}
          onPointerMove={drag.move(grip)}
          onPointerUp={drag.release}
          onPointerCancel={drag.release}
          style={{ cursor }}
          className={`absolute z-10 grid touch-none place-items-center ${className}`}
        >
          <span className="pointer-events-none size-1.5 bg-white ring-1 ring-indigo-500" />
        </span>
      ))}
    </>
  )
}
