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

// Keep these aligned with the API's decoded-image limits.
export function validateImageDimensions(width: number, height: number): void {
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width <= 0 || height <= 0 ||
    width > 16384 || height > 16384 || width * height > 24_000_000) {
    throw new Error('images must be at most 24 megapixels and 16384 pixels on either side')
  }
}

export function loadImage(file: File, folder?: string): Promise<GalleryImage | null> {
  if (file.size > 32 * 1024 * 1024) return Promise.reject(new Error('images must be at most 32 MiB'))
  const url = URL.createObjectURL(file)

  return new Promise((resolve, reject) => {
    const probe = new Image()
    const release = () => {
      probe.onload = null
      probe.onerror = null
      probe.src = ''
    }
    probe.onload = () => {
      const width = probe.naturalWidth
      const height = probe.naturalHeight
      try {
        validateImageDimensions(width, height)
        resolve({
          id: crypto.randomUUID(), file, url,
          name: file.name.slice(file.name.lastIndexOf('/') + 1),
          size: file.size, addedAt: Date.now(), width, height, folder,
        })
      } catch (cause) {
        URL.revokeObjectURL(url)
        reject(cause)
      } finally {
        release()
      }
    }
    probe.onerror = () => {
      URL.revokeObjectURL(url)
      release()
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
