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
  const heldFolders = useRef(folders)
  const generation = useRef(0)
  const pending = useRef(0)
  const mounted = useRef(true)

  const cancelImports = useCallback(() => {
    generation.current += 1
    pending.current = 0
    if (mounted.current) setBusy(false)
  }, [])

  const restore = useCallback((saved: Omit<GalleryImage, 'url'>[], savedFolders: GalleryFolder[]) => {
    if (!mounted.current) return
    cancelImports()
    const restored: GalleryImage[] = []
    try {
      for (const image of saved) restored.push({ ...image, url: URL.createObjectURL(image.file) })
    } catch (error) {
      for (const image of restored) URL.revokeObjectURL(image.url)
      throw error
    }
    for (const image of latest.current) URL.revokeObjectURL(image.url)
    latest.current = restored
    heldFolders.current = [...savedFolders]
    setImages(restored)
    setFolders(heldFolders.current)
    setNotice(null)
  }, [cancelImports])

  useEffect(() => {
    mounted.current = true
    if (latest.current.length > 0) restore(latest.current, heldFolders.current)
    return () => {
      mounted.current = false
      generation.current += 1
      pending.current = 0
      for (const image of latest.current) URL.revokeObjectURL(image.url)
    }
  }, [restore])

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
      heldFolders.current = [...heldFolders.current, made]
      setFolders(heldFolders.current)
      setNotice(null)
      return made.id
    },
    [say],
  )

  const add = useCallback(
    async (incoming: FileList | File[] | null, into?: string) => {
      const dropped = Array.from(incoming ?? [])
      if (dropped.length === 0 || !mounted.current) return
      const turn = generation.current
      const current = () => mounted.current && turn === generation.current
      pending.current += 1
      setBusy(true)
      try {

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
            if (!current()) return
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
        if (!current()) return

        const rejected = taken.filter(({ file }) => !isImage(file))
        const candidates = taken.filter(({ file }) => isImage(file))

        const loaded = (
          await Promise.all(candidates.map(({ file, folder }) => loadImage(file, folder)))
        ).filter((image) => image !== null)
        if (!current()) {
          for (const image of loaded) URL.revokeObjectURL(image.url)
          return
        }

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
          heldFolders.current = heldFolders.current
            .map((folder) =>
              naming.has(folder.id)
                ? { ...folder, archive: naming.get(folder.id) }
                : folder,
            )
            .concat(filled)
          setFolders(heldFolders.current)
        }
        if (fresh.length > 0) {
          latest.current = [...latest.current, ...fresh]
          setImages(latest.current)
        }

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
      } finally {
        if (current()) {
          pending.current -= 1
          setBusy(pending.current > 0)
        }
      }
    },
    [say],
  )

  const remove = useCallback((id: string) => {
    cancelImports()
    const going = latest.current.find((image) => image.id === id)
    if (!going) return
    URL.revokeObjectURL(going.url)
    latest.current = latest.current.filter((image) => image.id !== id)
    setImages(latest.current)

    const folder = going.folder
    if (
      folder &&
      !heldFolders.current.find((held) => held.id === folder)?.manual &&
      !latest.current.some((image) => image.folder === folder && image.id !== id)
    ) {
      heldFolders.current = heldFolders.current.filter((held) => held.id !== folder)
      setFolders(heldFolders.current)
    }
  }, [cancelImports])

  const dropFolder = useCallback((id: string) => {
    cancelImports()
    for (const image of latest.current) {
      if (image.folder === id) URL.revokeObjectURL(image.url)
    }
    latest.current = latest.current.filter((image) => image.folder !== id)
    heldFolders.current = heldFolders.current.filter((folder) => folder.id !== id)
    setImages(latest.current)
    setFolders(heldFolders.current)
  }, [cancelImports])

  const clear = useCallback(() => restore([], []), [restore])

  const dismissNotice = useCallback(() => setNotice(null), [])

  return {
    images,
    folders,
    makeFolder,
    add,
    remove,
    dropFolder,
    clear,
    restore,
    busy,
    notice,
    dismissNotice,
  }
}
