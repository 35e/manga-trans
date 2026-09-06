import { useCallback, useEffect, useRef } from 'react'
import type { GalleryImage } from '../lib/images'
import { Mask } from '../lib/mask'

type Entry = {
  mask?: Mask
  blob?: Blob
  loading?: Promise<Mask | null>
  revision: number
  lease: number
}

export type Masks = {
  get(image: GalleryImage | null): Mask | null
  load(image: GalleryImage | null): Promise<Mask | null>
  release(id: string): Promise<void>
  drop(id: string): void
  clear(): void
  snapshot(): Promise<Record<string, Blob>>
  restore(blobs: Record<string, Blob>): Promise<void>
}

export function useMasks(onChange?: () => void): Masks {
  const masks = useRef(new Map<string, Entry>())
  const changed = useRef(onChange)
  changed.current = onChange
  const mounted = useRef(true)

  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      for (const entry of masks.current.values()) entry.mask?.dispose()
      masks.current.clear()
    }
  }, [])

  const get = useCallback((image: GalleryImage | null) => {
    return image ? masks.current.get(image.id)?.mask ?? null : null
  }, [])

  const load = useCallback(async (image: GalleryImage | null): Promise<Mask | null> => {
    if (!image || !mounted.current) return null
    let entry = masks.current.get(image.id)
    if (!entry) {
      entry = { revision: 0, lease: 0 }
      masks.current.set(image.id, entry)
    }
    const held = entry
    // Reacquiring an active page must invalidate serialization already in flight.
    held.lease += 1
    if (held.mask) return held.mask
    if (held.loading) return held.loading
    const notify = () => {
      if (!mounted.current || masks.current.get(image.id) !== held) return
      held.revision += 1
      held.blob = undefined
      changed.current?.()
    }
    const loading = (async () => {
      const mask = held.blob
        ? await Mask.restore(held.blob, notify)
        : new Mask(image.width, image.height, notify)
      if (!mounted.current || masks.current.get(image.id) !== held) {
        mask.dispose()
        return null
      }
      held.mask = mask
      return mask
    })()
    held.loading = loading
    try {
      return await loading
    } finally {
      if (held.loading === loading) held.loading = undefined
    }
  }, [])

  const release = useCallback(async (id: string) => {
    const entry = masks.current.get(id)
    if (!entry) return
    const lease = ++entry.lease
    if (entry.loading) await entry.loading
    if (masks.current.get(id) !== entry || entry.lease !== lease) return
    const mask = entry.mask
    if (!mask) return
    const revision = entry.revision
    const blob = entry.blob ?? await mask.snapshot()
    if (masks.current.get(id) !== entry || entry.mask !== mask ||
      entry.lease !== lease || entry.revision !== revision) return
    entry.blob = blob
    entry.mask = undefined
    mask.dispose()
  }, [])

  const drop = useCallback((id: string) => {
    const entry = masks.current.get(id)
    if (!entry) return
    masks.current.delete(id)
    entry.mask?.dispose()
    changed.current?.()
  }, [])

  const clear = useCallback(() => {
    if (masks.current.size === 0) return
    for (const entry of masks.current.values()) entry.mask?.dispose()
    masks.current.clear()
    changed.current?.()
  }, [])

  const snapshot = useCallback(async () => {
    const blobs: Record<string, Blob> = {}
    for (const [id, entry] of masks.current) {
      const mask = entry.mask
      const revision = entry.revision
      const blob = entry.blob ?? (mask ? await mask.snapshot() : undefined)
      if (!blob || masks.current.get(id) !== entry) continue
      blobs[id] = blob
      if (entry.revision === revision) entry.blob = blob
    }
    return blobs
  }, [])

  const restore = useCallback(async (record: Record<string, Blob>) => {
    if (!mounted.current) throw new DOMException('Mask restoration was cancelled', 'AbortError')
    for (const entry of masks.current.values()) entry.mask?.dispose()
    masks.current = new Map(Object.entries(record).map(([id, blob]) => [id, { blob, revision: 0, lease: 0 }]))
  }, [])

  return { get, load, release, drop, clear, snapshot, restore }
}
