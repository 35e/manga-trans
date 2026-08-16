export type GalleryFolder = {
  id: string
  name: string
  addedAt: number
  archive?: string
  manual?: true
}

export type GalleryImage = {
  id: string
  file: File
  url: string
  name: string
  size: number
  addedAt: number
  width: number
  height: number
  folder?: string
}

export function fingerprint(file: File, folder = '') {
  return `${folder}:${file.name}:${file.size}:${file.lastModified}`
}

export function stem(name: string) {
  return name.replace(/\.[^.]+$/, '')
}

export function isImage(file: File) {
  return file.type.startsWith('image/')
}

export function loadImage(file: File, folder?: string): Promise<GalleryImage | null> {
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
        folder,
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

export function plural(count: number, noun: string) {
  return `${count} ${noun}${count === 1 ? '' : 's'}`
}
