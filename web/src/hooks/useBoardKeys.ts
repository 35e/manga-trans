import { useEffect, useRef } from 'react'
import type { BoardMode } from '../lib/api'
import { typingInto } from '../lib/dom'
import type { Lines } from '../lib/lettering'
import type { BoardView } from './useBoardView'

export function useZoomKeys(view: BoardView, enabled: boolean) {
  const { zoomIn, zoomOut, fit, actual } = view

  useEffect(() => {
    if (!enabled) return

    const onKey = (event: KeyboardEvent) => {
      if (event.ctrlKey || event.metaKey || event.altKey) return
      if (typingInto(event.target)) return

      const zooming: Record<string, () => void> = {
        '+': zoomIn,
        '=': zoomIn,
        '-': zoomOut,
        _: zoomOut,
        0: fit,
        1: actual,
      }
      const go = zooming[event.key]
      if (!go) return
      event.preventDefault()
      go()
    }

    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [enabled, zoomIn, zoomOut, fit, actual])
}

type BlockKeys = {
  mode: BoardMode
  selected: number | null
  lettering: Lines
  onToggleExcluded: (index: number) => void
  onSize: (index: number, by: number) => void
  onTurn: (index: number, angle: number) => void
}

export function useBlockKeys({
  mode,
  selected,
  lettering,
  onToggleExcluded,
  onSize,
  onTurn,
}: BlockKeys) {
  const now = useRef(lettering)
  now.current = lettering

  useEffect(() => {
    if (selected === null) return

    const onKey = (event: KeyboardEvent) => {
      if (typingInto(event.target)) return

      if (mode === 'translate') {
        const sizing = event.key === 'ArrowUp' || event.key === 'ArrowDown'
        const turning = event.key === 'ArrowLeft' || event.key === 'ArrowRight'
        if (!sizing && !turning) return
        event.preventDefault()

        if (sizing) {
          const step = event.shiftKey ? 5 : 1
          onSize(selected, event.key === 'ArrowUp' ? step : -step)
          return
        }

        const line = now.current[selected]
        if (!line) return
        const step = event.shiftKey ? 15 : 1
        onTurn(selected, line.angle + (event.key === 'ArrowRight' ? step : -step))
        return
      }

      if (event.key !== 'Delete' && event.key !== 'Backspace') return
      event.preventDefault()
      onToggleExcluded(selected)
    }

    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [selected, mode, onToggleExcluded, onSize, onTurn])
}
