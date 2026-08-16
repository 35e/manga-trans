import { useCallback, useRef } from 'react'
import type { GalleryImage } from '../lib/images'
import { Mask } from '../lib/mask'

export function useMasks() {
  const masks = useRef(new Map<string, Mask>())

  const forPage = useCallback((image: GalleryImage | null) => {
    if (!image) return null
    const held = masks.current.get(image.id)
    if (held) return held
    const made = new Mask(image.width, image.height)
    masks.current.set(image.id, made)
    return made
  }, [])

  const drop = useCallback((id: string) => {
    masks.current.delete(id)
  }, [])

  const clear = useCallback(() => {
    masks.current.clear()
  }, [])

  return { forPage, drop, clear }
}
