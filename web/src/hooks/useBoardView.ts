import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import { typingNow } from '../lib/dom'
import type { GalleryImage } from '../lib/images'

const ZOOM_MIN = 0.05
const ZOOM_MAX = 8

const STEP = 1.25

const PAD = 28

const BAR = 56

const CRISP_AT = 3

const clamp = (value: number, low: number, high: number) =>
  Math.min(high, Math.max(low, value))

type Size = { width: number; height: number }

export type BoardView = {
  page: Size | null
  content: Size
  origin: { x: number; y: number }
  scale: number
  grid: number
  fitted: boolean
  panning: boolean
  crisp: boolean
  fit: () => void
  actual: () => void
  zoomIn: () => void
  zoomOut: () => void
}

export function useBoardView(
  surface: React.RefObject<HTMLDivElement | null>,
  image: GalleryImage | null,
): BoardView {
  const [box, setBox] = useState<Size>({ width: 0, height: 0 })
  const [zoom, setZoom] = useState<number | null>(null)
  const [panning, setPanning] = useState(false)

  const width = image?.width ?? 0
  const height = image?.height ?? 0

  useEffect(() => {
    const element = surface.current
    if (!element) return
    const observer = new ResizeObserver(([entry]) => setBox(entry.contentRect))
    observer.observe(element)
    return () => observer.disconnect()
  }, [surface])

  const id = image?.id
  useEffect(() => setZoom(null), [id])

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

  const origin = useMemo(
    () => ({
      x: (content.width - (page?.width ?? 0)) / 2,
      y: (content.height - BAR - (page?.height ?? 0)) / 2,
    }),
    [content, page],
  )

  const now = useRef({ scale, origin, page })
  now.current = { scale, origin, page }

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

  const zoomAt = useCallback(
    (factor: number, clientX?: number, clientY?: number) => {
      const element = surface.current
      const { scale: was, origin: at, page: drawn } = now.current
      const next = clamp(was * factor, ZOOM_MIN, ZOOM_MAX)
      if (next === was) return

      if (element && drawn) {
        const rect = element.getBoundingClientRect()
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

  const spacing = useRef(false)

  useEffect(() => {
    const down = (event: KeyboardEvent) => {
      if (event.code === 'Space' && !typingNow()) spacing.current = true
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
