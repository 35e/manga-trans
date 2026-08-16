import { useCallback, useEffect, useRef, useState } from 'react'
import type { GalleryFolder, GalleryImage } from '../lib/images'
import { fingerprint, isImage, loadImage, plural, stem } from '../lib/images'
import { expand, isZip } from '../lib/zip'

export type LibraryNotice = { id: string; text: string }

export function useImageLibrary() {
  const [images, setImages] = useState<GalleryImage[]>([])
  const [folders, setFolders] = useState<GalleryFolder[]>([])
  const [notice, setNotice] = useState<LibraryNotice | null>(null)
  const [busy, setBusy] = useState(false)

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

  const makeFolder = useCallback(
    (name: string): string | null => {
      const called = name.trim()
      if (!called) return null
      if (heldFolders.current.some((folder) => folder.name === called)) {
        say(`there is already a folder called ${called}`)
        return null
      }
      const made: GalleryFolder = {
        id: crypto.randomUUID(),
        name: called,
        addedAt: Date.now(),
        manual: true,
      }
      setFolders((current) => [...current, made])
      setNotice(null)
      return made.id
    },
    [say],
  )

  const add = useCallback(
    async (incoming: FileList | File[] | null, into?: string) => {
      const dropped = Array.from(incoming ?? [])
      if (dropped.length === 0) return

      setBusy(true)

      const taken: { file: File; folder?: string }[] = []
      const opened: GalleryFolder[] = []
      const named = new Map<string, string>()
      const unopenable: string[] = []
      const hollow: string[] = []

      const folderFor = (archive: string): GalleryFolder => {
        const name = stem(archive)
        const held =
          heldFolders.current.find((folder) => folder.name === name) ??
          opened.find((folder) => folder.name === name)
        if (held) {
          if (!held.archive) named.set(held.id, archive)
          return held
        }
        const made = { id: crypto.randomUUID(), name, addedAt: Date.now(), archive }
        opened.push(made)
        return made
      }

      for (const file of dropped) {
        if (!isZip(file)) {
          taken.push({ file, folder: into })
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

      const filled = opened.filter((folder) =>
        fresh.some((image) => image.folder === folder.id),
      )
      const naming = new Map(
        [...named].filter(([id]) => fresh.some((image) => image.folder === id)),
      )
      if (filled.length > 0 || naming.size > 0) {
        setFolders((current) =>
          current
            .map((folder) =>
              naming.has(folder.id)
                ? { ...folder, archive: naming.get(folder.id) }
                : folder,
            )
            .concat(filled),
        )
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
    const going = latest.current.find((image) => image.id === id)
    if (!going) return
    URL.revokeObjectURL(going.url)
    setImages((current) => current.filter((image) => image.id !== id))

    const folder = going.folder
    if (
      folder &&
      !heldFolders.current.find((held) => held.id === folder)?.manual &&
      !latest.current.some((image) => image.folder === folder && image.id !== id)
    ) {
      setFolders((current) => current.filter((held) => held.id !== folder))
    }
  }, [])

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
    makeFolder,
    add,
    remove,
    dropFolder,
    clear,
    busy,
    notice,
    dismissNotice,
  }
}
