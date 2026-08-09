import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import type { GalleryImage } from '../lib/images'

export const ZOOM_MIN = 0.05
export const ZOOM_MAX = 8

/** One press of a zoom button, or one notch of a wheel. */
const STEP = 1.25

/** Breathing room left around the page, in screen pixels. */
const PAD = 28

/**
 * What the bar floating at the foot of the board takes up. Kept out of the room
 * the page is fitted and centred in, so the bar never lies over the page it is
 * there to work on.
 */
const BAR = 56

/** Past this the page is drawn pixel for pixel, which is the point of going in. */
const CRISP_AT = 3

const clamp = (value: number, low: number, high: number) =>
  Math.min(high, Math.max(low, value))

type Size = { width: number; height: number }

export type BoardView = {
  /** The size the page is drawn at, or null when there is no page. */
  page: Size | null
  /** The mat the page is laid on: the board's own box, or the page and its
   *  margins when those are the larger — which is what makes it scroll. */
  content: Size
  /** Where on the mat the page's top left corner goes. */
  origin: { x: number; y: number }
  /** Drawn pixels per page pixel. Everything laid over the page reads this. */
  scale: number
  /** How far apart the mat's dots are drawn, so the mat zooms with the page. */
  grid: number
  /** Sitting at the size the whole page fits at, so Fit has nothing to do. */
  fitted: boolean
  /** Being dragged about with the middle button or a held space. */
  panning: boolean
  /** Worth drawing pixel for pixel rather than smoothed. */
  crisp: boolean
  fit: () => void
  actual: () => void
  zoomIn: () => void
  zoomOut: () => void
}

/**
 * The page on the board: how large it is drawn, and how it is moved about.
 *
 * The board is a scroll box, so panning is the browser's own — a trackpad, a
 * wheel, the scrollbars and the keys all work without being taught to. What is
 * added here is the zoom: the buttons, ctrl and the wheel together, and the
 * arithmetic that keeps whatever was under the pointer under it afterwards.
 *
 * `surface` is the scroll box itself. The page is laid inside a mat of
 * `content` size, centred, which is what gives it its margins at any zoom.
 */
export function useBoardView(
  surface: React.RefObject<HTMLDivElement | null>,
  image: GalleryImage | null,
): BoardView {
  const [box, setBox] = useState<Size>({ width: 0, height: 0 })
  // Null while the page is following the board's size rather than being held at
  // a zoom of its own, which is what "fitted" means.
  const [zoom, setZoom] = useState<number | null>(null)
  const [panning, setPanning] = useState(false)

  const width = image?.width ?? 0
  const height = image?.height ?? 0

  useEffect(() => {
    const element = surface.current
    if (!element) return
    // ResizeObserver reports the box once on observe, so there is no separate
    // first measurement — and none taken a different way.
    const observer = new ResizeObserver(([entry]) => setBox(entry.contentRect))
    observer.observe(element)
    return () => observer.disconnect()
  }, [surface])

  // A different page arrives at its fitted size, whatever the last one was left
  // at: the zoom belonged to that page, not to the board.
  const id = image?.id
  useEffect(() => setZoom(null), [id])

  // Nothing is drawn until the board has been measured: laying the page out
  // against a board of no size would put it up at full size for a frame first.
  const measured = box.width > 0 && box.height > 0
  const showing = measured && width > 0 && height > 0

  const fitScale = showing
    ? clamp(
        Math.min(
          (box.width - PAD * 2) / width,
          (box.height - PAD * 2 - BAR) / height,
        ),
        ZOOM_MIN,
        1,
      )
    : 1

  const scale = zoom ?? fitScale

  // Held by identity as well as by value: what is laid over the page redraws
  // when this changes, and it must not change on every render for nothing.
  //
  // Floored, so a page at its fitted size can never round its way past the
  // board and put scrollbars up — which would shrink the board, and refit.
  const page = useMemo(
    () =>
      showing
        ? { width: Math.floor(width * scale), height: Math.floor(height * scale) }
        : null,
    [showing, width, height, scale],
  )

  const content = useMemo<Size>(
    () => ({
      width: Math.max(box.width, page ? page.width + PAD * 2 : 0),
      height: Math.max(box.height, page ? page.height + PAD * 2 + BAR : 0),
    }),
    [box.width, box.height, page],
  )

  /**
   * Where the page sits on the mat: across the middle, and centred in what is
   * left above the floating bar. Worked out here and nowhere else, because the
   * zoom keeps a point still by measuring against it.
   */
  const origin = useMemo(
    () => ({
      x: (content.width - (page?.width ?? 0)) / 2,
      y: (content.height - BAR - (page?.height ?? 0)) / 2,
    }),
    [content, page],
  )

  // Read by the wheel and pan listeners, which are subscribed once and must not
  // be resubscribed on every pixel of zoom.
  const now = useRef({ scale, origin, page })
  now.current = { scale, origin, page }

  /**
   * What was under the pointer when the zoom began, in the page's own pixels,
   * with where on the board it was. Applied once the new size has been laid
   * out, which is what keeps that point still.
   */
  const anchor = useRef<{ x: number; y: number; left: number; top: number } | null>(
    null,
  )

  useLayoutEffect(() => {
    const element = surface.current
    const held = anchor.current
    if (!element || !held) return
    anchor.current = null

    const { origin: at, page: drawn } = now.current
    if (!drawn) return
    element.scrollLeft = at.x + held.x * scale - held.left
    element.scrollTop = at.y + held.y * scale - held.top
  }, [surface, scale])

  /** Zoom about a point on the board, given in client coordinates. */
  const zoomAt = useCallback(
    (factor: number, clientX?: number, clientY?: number) => {
      const element = surface.current
      const { scale: was, origin: at, page: drawn } = now.current
      const next = clamp(was * factor, ZOOM_MIN, ZOOM_MAX)
      if (next === was) return

      if (element && drawn) {
        const rect = element.getBoundingClientRect()
        // The middle of the board when no point was given: pressing a button is
        // not aimed at anything, so it goes in on what is already in the middle.
        const left = (clientX ?? rect.left + rect.width / 2) - rect.left
        const top = (clientY ?? rect.top + rect.height / 2) - rect.top
        anchor.current = {
          x: (element.scrollLeft + left - at.x) / was,
          y: (element.scrollTop + top - at.y) / was,
          left,
          top,
        }
      }
      setZoom(next)
    },
    [surface],
  )

  // Ctrl or command with the wheel, which is also what a trackpad pinch arrives
  // as. A plain wheel is left alone: that is the board scrolling, as it should.
  useEffect(() => {
    const element = surface.current
    if (!element) return

    const onWheel = (event: WheelEvent) => {
      if (!event.ctrlKey && !event.metaKey) return
      event.preventDefault()
      zoomAt(Math.exp(-event.deltaY * 0.0015), event.clientX, event.clientY)
    }

    element.addEventListener('wheel', onWheel, { passive: false })
    return () => element.removeEventListener('wheel', onWheel)
  }, [surface, zoomAt])

  // Space held down, which turns a plain drag into a pan. Tracked rather than
  // swallowed: space still works the button it is pressed on.
  const spacing = useRef(false)

  useEffect(() => {
    const typing = () => {
      const focused = document.activeElement
      return (
        focused instanceof HTMLElement &&
        (focused.isContentEditable ||
          ['INPUT', 'TEXTAREA', 'SELECT'].includes(focused.tagName))
      )
    }

    const down = (event: KeyboardEvent) => {
      if (event.code === 'Space' && !typing()) spacing.current = true
    }
    const up = (event: KeyboardEvent) => {
      if (event.code === 'Space') spacing.current = false
    }
    const blurred = () => {
      spacing.current = false
    }

    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    window.addEventListener('blur', blurred)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
      window.removeEventListener('blur', blurred)
    }
  }, [])

  /**
   * Panning with the middle button, or with space held. Caught on the way down
   * rather than on the way up, so that a pan over the page never reaches the
   * brush or a block and starts drawing one instead.
   */
  useEffect(() => {
    const element = surface.current
    if (!element) return
    let from: { x: number; y: number; left: number; top: number } | null = null

    const down = (event: PointerEvent) => {
      const wanted = event.button === 1 || (event.button === 0 && spacing.current)
      if (!wanted) return
      event.preventDefault()
      event.stopPropagation()
      from = {
        x: event.clientX,
        y: event.clientY,
        left: element.scrollLeft,
        top: element.scrollTop,
      }
      element.setPointerCapture(event.pointerId)
      setPanning(true)
    }

    const move = (event: PointerEvent) => {
      if (!from) return
      element.scrollLeft = from.left - (event.clientX - from.x)
      element.scrollTop = from.top - (event.clientY - from.y)
    }

    const up = (event: PointerEvent) => {
      if (!from) return
      from = null
      setPanning(false)
      if (element.hasPointerCapture(event.pointerId)) {
        element.releasePointerCapture(event.pointerId)
      }
    }

    element.addEventListener('pointerdown', down, { capture: true })
    element.addEventListener('pointermove', move)
    element.addEventListener('pointerup', up)
    element.addEventListener('pointercancel', up)
    // The middle button opens a paste on some platforms and scrolls on others.
    const noAuxiliary = (event: MouseEvent) => {
      if (event.button === 1) event.preventDefault()
    }
    element.addEventListener('auxclick', noAuxiliary)

    return () => {
      element.removeEventListener('pointerdown', down, { capture: true })
      element.removeEventListener('pointermove', move)
      element.removeEventListener('pointerup', up)
      element.removeEventListener('pointercancel', up)
      element.removeEventListener('auxclick', noAuxiliary)
    }
  }, [surface])

  const fit = useCallback(() => setZoom(null), [])
  const actual = useCallback(() => zoomAt(1 / now.current.scale), [zoomAt])
  const zoomIn = useCallback(() => zoomAt(STEP), [zoomAt])
  const zoomOut = useCallback(() => zoomAt(1 / STEP), [zoomAt])

  return {
    page,
    content,
    origin,
    scale,
    grid: clamp(24 * scale, 12, 96),
    fitted: zoom === null,
    panning,
    crisp: scale >= CRISP_AT,
    fit,
    actual,
    zoomIn,
    zoomOut,
  }
}
