import { useCallback, useEffect, useRef, useState } from 'react'
import type { GalleryImage } from '../lib/images'
import { fingerprint, isImage, loadImage, plural } from '../lib/images'
import { expand, isZip } from '../lib/zip'

export type LibraryNotice = { id: string; text: string }

/**
 * The dropped images and everything that changes them. Object URLs are released
 * as soon as an image is removed, and on unmount for whatever is left.
 */
export function useImageLibrary() {
  const [images, setImages] = useState<GalleryImage[]>([])
  const [notice, setNotice] = useState<LibraryNotice | null>(null)
  const [busy, setBusy] = useState(false)

  // Revoking on unmount reads the list through a ref so the effect need not
  // re-run — and re-revoke — on every add.
  const latest = useRef(images)
  latest.current = images

  useEffect(() => {
    return () => {
      for (const image of latest.current) URL.revokeObjectURL(image.url)
    }
  }, [])

  const say = useCallback((text: string) => {
    setNotice({ id: crypto.randomUUID(), text })
  }, [])

  const add = useCallback(
    async (incoming: FileList | File[] | null) => {
      const dropped = Array.from(incoming ?? [])
      if (dropped.length === 0) return

      setBusy(true)

      // An archive is opened where it was dropped, so a zip among loose pages
      // leaves everything in the order it arrived in.
      const files: File[] = []
      const unopenable: string[] = []
      const hollow: string[] = []

      for (const file of dropped) {
        if (!isZip(file)) {
          files.push(file)
          continue
        }
        try {
          const inside = await expand(file)
          if (inside.length === 0) hollow.push(file.name)
          files.push(...inside)
        } catch {
          unopenable.push(file.name)
        }
      }

      const rejected = files.filter((file) => !isImage(file))
      const candidates = files.filter(isImage)

      const loaded = (await Promise.all(candidates.map(loadImage))).filter(
        (image) => image !== null,
      )
      setBusy(false)

      // Dedupe out here rather than inside the updater: an updater runs twice
      // under StrictMode, and this one counts and releases as it goes.
      const seen = new Set(latest.current.map((image) => fingerprint(image.file)))
      const fresh: GalleryImage[] = []
      let duplicates = 0

      for (const image of loaded) {
        const print = fingerprint(image.file)
        if (seen.has(print)) {
          duplicates += 1
          URL.revokeObjectURL(image.url)
          continue
        }
        seen.add(print)
        fresh.push(image)
      }

      if (fresh.length > 0) setImages((current) => [...current, ...fresh])

      const broken = candidates.length - loaded.length
      const problems = [
        unopenable.length > 0 &&
          `${plural(unopenable.length, 'archive')} could not be opened`,
        hollow.length > 0 &&
          `${plural(hollow.length, 'archive')} held no images`,
        rejected.length > 0 && `${plural(rejected.length, 'file')} skipped`,
        broken > 0 && `${plural(broken, 'image')} could not be read`,
        duplicates > 0 && `${plural(duplicates, 'duplicate')} skipped`,
      ].filter((problem) => typeof problem === 'string')

      if (problems.length > 0) say(problems.join(' · '))
      else setNotice(null)
    },
    [say],
  )

  const remove = useCallback((id: string) => {
    setImages((current) => {
      const going = current.find((image) => image.id === id)
      if (going) URL.revokeObjectURL(going.url)
      return current.filter((image) => image.id !== id)
    })
  }, [])

  const clear = useCallback(() => {
    setImages((current) => {
      for (const image of current) URL.revokeObjectURL(image.url)
      return []
    })
    setNotice(null)
  }, [])

  const dismissNotice = useCallback(() => setNotice(null), [])

  return { images, add, remove, clear, busy, notice, dismissNotice }
}
