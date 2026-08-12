import { useCallback, useEffect, useRef, useState } from 'react'
import type { GalleryFolder, GalleryImage } from '../lib/images'
import { fingerprint, isImage, loadImage, plural, stem } from '../lib/images'
import { expand, isZip } from '../lib/zip'

export type LibraryNotice = { id: string; text: string }

/**
 * The dropped images and everything that changes them. Object URLs are released
 * as soon as an image is removed, and on unmount for whatever is left.
 *
 * An archive comes in as a folder of its pages rather than as fifty loose ones: a
 * chapter is worked through as a chapter, and the whole of one can be run at once.
 */
export function useImageLibrary() {
  const [images, setImages] = useState<GalleryImage[]>([])
  const [folders, setFolders] = useState<GalleryFolder[]>([])
  const [notice, setNotice] = useState<LibraryNotice | null>(null)
  const [busy, setBusy] = useState(false)

  // Revoking on unmount reads the list through a ref so the effect need not
  // re-run — and re-revoke — on every add.
  const latest = useRef(images)
  latest.current = images
  const heldFolders = useRef(folders)
  heldFolders.current = folders

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
      // leaves everything in the order it arrived in — but its pages are tagged
      // with a folder of their own on the way through.
      const taken: { file: File; folder?: string }[] = []
      const opened: GalleryFolder[] = []
      const unopenable: string[] = []
      const hollow: string[] = []

      /**
       * The folder for this archive: the one it already has, or a new one.
       *
       * Matched on the stem, so `ch01.cbz` dropped twice fills one folder — and
       * so does `ch01.zip` after it, which keeps whichever extension named the
       * folder in the first place rather than changing what it will come back as
       * halfway through a chapter.
       */
      const folderFor = (archive: string): GalleryFolder => {
        const name = stem(archive)
        const held =
          heldFolders.current.find((folder) => folder.name === name) ??
          opened.find((folder) => folder.name === name)
        if (held) return held
        const made = { id: crypto.randomUUID(), name, addedAt: Date.now(), archive }
        opened.push(made)
        return made
      }

      for (const file of dropped) {
        if (!isZip(file)) {
          taken.push({ file })
          continue
        }
        try {
          const inside = await expand(file)
          if (inside.length === 0) {
            hollow.push(file.name)
            continue
          }
          const folder = folderFor(file.name)
          for (const page of inside) taken.push({ file: page, folder: folder.id })
        } catch {
          unopenable.push(file.name)
        }
      }

      const rejected = taken.filter(({ file }) => !isImage(file))
      const candidates = taken.filter(({ file }) => isImage(file))

      const loaded = (
        await Promise.all(candidates.map(({ file, folder }) => loadImage(file, folder)))
      ).filter((image) => image !== null)
      setBusy(false)

      // Dedupe out here rather than inside the updater: an updater runs twice
      // under StrictMode, and this one counts and releases as it goes.
      const seen = new Set(
        latest.current.map((image) => fingerprint(image.file, image.folder)),
      )
      const fresh: GalleryImage[] = []
      let duplicates = 0

      for (const image of loaded) {
        const print = fingerprint(image.file, image.folder)
        if (seen.has(print)) {
          duplicates += 1
          URL.revokeObjectURL(image.url)
          continue
        }
        seen.add(print)
        fresh.push(image)
      }

      // Only the folders that ended up holding something: an archive of nothing
      // but duplicates leaves no folder behind.
      const filled = opened.filter((folder) =>
        fresh.some((image) => image.folder === folder.id),
      )
      if (filled.length > 0) setFolders((current) => [...current, ...filled])
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
    const going = latest.current.find((image) => image.id === id)
    if (!going) return
    URL.revokeObjectURL(going.url)
    setImages((current) => current.filter((image) => image.id !== id))

    // A folder with nothing left in it is not a folder.
    const folder = going.folder
    if (
      folder &&
      !latest.current.some((image) => image.folder === folder && image.id !== id)
    ) {
      setFolders((current) => current.filter((held) => held.id !== folder))
    }
  }, [])

  /** A folder and every page in it. */
  const dropFolder = useCallback((id: string) => {
    for (const image of latest.current) {
      if (image.folder === id) URL.revokeObjectURL(image.url)
    }
    setImages((current) => current.filter((image) => image.folder !== id))
    setFolders((current) => current.filter((folder) => folder.id !== id))
  }, [])

  const clear = useCallback(() => {
    for (const image of latest.current) URL.revokeObjectURL(image.url)
    setImages([])
    setFolders([])
    setNotice(null)
  }, [])

  const dismissNotice = useCallback(() => setNotice(null), [])

  return {
    images,
    folders,
    add,
    remove,
    dropFolder,
    clear,
    busy,
    notice,
    dismissNotice,
  }
}
