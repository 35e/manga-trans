import { useCallback, useEffect, useRef } from 'react'
import type { GalleryImage } from '../lib/images'
import { Mask } from '../lib/mask'

export function useMasks(onChange?: () => void) {
  const masks = useRef(new Map<string, Mask>())
  const changed = useRef(onChange)
  changed.current = onChange
  const generation = useRef(0)
  const mounted = useRef(true)

  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      generation.current += 1
    }
  }, [])

  const forPage = useCallback((image: GalleryImage | null) => {
    if (!image) return null
    const held = masks.current.get(image.id)
    if (held) return held
    const made = new Mask(image.width, image.height, () => {
      if (mounted.current && masks.current.get(image.id) === made) changed.current?.()
    })
    masks.current.set(image.id, made)
    return made
  }, [])

  const drop = useCallback((id: string) => {
    generation.current += 1
    if (masks.current.delete(id)) changed.current?.()
  }, [])

  const clear = useCallback(() => {
    generation.current += 1
    if (masks.current.size === 0) return
    masks.current.clear()
    changed.current?.()
  }, [])

  const snapshot = useCallback(async () => Object.fromEntries(
    await Promise.all([...masks.current].map(async ([id, mask]) => [id, await mask.snapshot()] as const)),
  ), [])

  const restore = useCallback(async (record: Record<string, Blob>) => {
    const turn = ++generation.current
    const restored = new Map(await Promise.all(Object.entries(record).map(async ([id, blob]) => {
      const mask = await Mask.restore(blob, () => {
        if (mounted.current && masks.current.get(id) === mask) changed.current?.()
      })
      return [id, mask] as const
    })))
    if (!mounted.current || turn !== generation.current) {
      throw new DOMException('Mask restoration was cancelled', 'AbortError')
    }
    masks.current = restored
  }, [])

  return { forPage, drop, clear, snapshot, restore }
}
