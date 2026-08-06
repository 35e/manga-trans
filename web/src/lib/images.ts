export type GalleryImage = {
  id: string
  file: File
  /** Object URL for `file`; revoked when the image leaves the library. */
  url: string
  name: string
  size: number
  addedAt: number
  width: number
  height: number
}

/** Same name, size and mtime: the same file dropped twice. */
export function fingerprint(file: File) {
  return `${file.name}:${file.size}:${file.lastModified}`
}

export function isImage(file: File) {
  return file.type.startsWith('image/')
}

/**
 * Wrap a file in a gallery entry, resolving once the browser has decoded it far
 * enough to know its size. A file the browser cannot decode resolves to null and
 * its object URL is released.
 */
export function loadImage(file: File): Promise<GalleryImage | null> {
  const url = URL.createObjectURL(file)

  return new Promise((resolve) => {
    const probe = new Image()

    probe.onload = () =>
      resolve({
        id: crypto.randomUUID(),
        file,
        url,
        name: file.name,
        size: file.size,
        addedAt: Date.now(),
        width: probe.naturalWidth,
        height: probe.naturalHeight,
      })

    probe.onerror = () => {
      URL.revokeObjectURL(url)
      resolve(null)
    }

    probe.src = url
  })
}

export function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  const units = ['KB', 'MB', 'GB']
  let value = bytes / 1024
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit += 1
  }
  return `${value.toFixed(value < 10 ? 1 : 0)} ${units[unit]}`
}

/** "3 pages", "1 page" — plural without the (s). */
export function plural(count: number, noun: string) {
  return `${count} ${noun}${count === 1 ? '' : 's'}`
}
