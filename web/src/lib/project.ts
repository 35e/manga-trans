import type { Analysis, Fill } from './api'
import type { GalleryFolder, GalleryImage } from './images'
import type { Lines } from './lettering'

export type ProjectData = {
  images: Omit<GalleryImage, 'url'>[]
  folders: GalleryFolder[]
  analyses: Record<string, Analysis>
  lettering: Record<string, Lines>
  cleaned: Record<string, Blob>
  masks: Record<string, Blob>
  touchups: Record<string, Blob>
  activeId: string | null
  openFolder: string | null
  language: string
  model: string
  target: string
  prompt: string | null
  spread: number
  fill: Fill
}

type StoredProject = {
  version: 1
  data: Omit<ProjectData, 'images'> & {
    images: (Omit<GalleryImage, 'url' | 'file'> & { file: Blob; fileName: string; fileLastModified: number })[]
  }
}

const DATABASE = 'manga-trans-project'
let pending = Promise.resolve()

function ordered<T>(work: () => Promise<T>): Promise<T> {
  const result = pending.then(work)
  pending = result.then(() => undefined, () => undefined)
  return result
}

function open(name: string): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(name, 1)
    let blocked = false
    request.onupgradeneeded = () => request.result.createObjectStore('project')
    request.onblocked = () => {
      blocked = true
      reject(new Error('Project storage is blocked by another tab. Close it and retry.'))
    }
    request.onerror = () => reject(request.error ?? new Error('Project storage could not be opened'))
    request.onsuccess = () => {
      if (blocked) request.result.close()
      else resolve(request.result)
    }
  })
}

function object(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function decode(value: unknown): ProjectData {
  if (!object(value) || value.version !== 1 || !object(value.data)) {
    throw new Error('The saved project has an unsupported or damaged format; it has not been overwritten.')
  }
  const data = value.data
  const nullableString = (item: unknown) => item === null || typeof item === 'string'
  if (!Array.isArray(data.images) || !Array.isArray(data.folders) ||
    !object(data.analyses) || !object(data.lettering) ||
    !nullableString(data.activeId) || !nullableString(data.openFolder) || !nullableString(data.prompt) ||
    typeof data.language !== 'string' || typeof data.model !== 'string' || typeof data.target !== 'string' ||
    typeof data.spread !== 'number' || !Number.isFinite(data.spread) ||
    !['art', 'telea', 'white'].includes(data.fill as string)) {
    throw new Error('The saved project is incomplete; it has not been overwritten.')
  }
  for (const key of ['cleaned', 'masks', 'touchups']) {
    const blobs = data[key]
    if (!object(blobs) || Object.values(blobs).some((blob) => !(blob instanceof Blob))) {
      throw new Error(`The saved project's ${key} are damaged; it has not been overwritten.`)
    }
  }
  for (const image of data.images) {
    if (!object(image) || !(image.file instanceof Blob) || typeof image.fileName !== 'string' ||
      typeof image.fileLastModified !== 'number' || !Number.isFinite(image.fileLastModified) ||
      typeof image.id !== 'string' || typeof image.name !== 'string' ||
      typeof image.width !== 'number' || !Number.isFinite(image.width) || image.width <= 0 ||
      typeof image.height !== 'number' || !Number.isFinite(image.height) || image.height <= 0 ||
      typeof image.size !== 'number' || typeof image.addedAt !== 'number' ||
      (image.folder !== undefined && typeof image.folder !== 'string')) {
      throw new Error('A saved source page is damaged; the project has not been overwritten.')
    }
  }
  for (const folder of data.folders) {
    if (!object(folder) || typeof folder.id !== 'string' || typeof folder.name !== 'string' ||
      typeof folder.addedAt !== 'number' || (folder.archive !== undefined && typeof folder.archive !== 'string') ||
      (folder.manual !== undefined && folder.manual !== true)) {
      throw new Error('A saved folder is damaged; the project has not been overwritten.')
    }
  }
  const box = (item: unknown) => Array.isArray(item) && item.length === 4 &&
    item.every((coordinate) => typeof coordinate === 'number' && Number.isFinite(coordinate))
  for (const analysis of Object.values(data.analyses)) {
    if (!object(analysis) || !object(analysis.detection) ||
      typeof analysis.detection.width !== 'number' || typeof analysis.detection.height !== 'number' ||
      !Array.isArray(analysis.detection.regions) ||
      !Array.isArray(analysis.excluded) || analysis.excluded.some((index) => !Number.isInteger(index) || index < 0) ||
      (analysis.texts !== null && (!Array.isArray(analysis.texts) || analysis.texts.some((text) => typeof text !== 'string')))) {
      throw new Error('Saved page analysis is damaged; the project has not been overwritten.')
    }
    for (const region of analysis.detection.regions) {
      if (!object(region) || typeof region.id !== 'string' || !box(region.box) ||
        typeof region.confidence !== 'number' || (region.bubble != null && !box(region.bubble))) {
        throw new Error('A saved text region is damaged; the project has not been overwritten.')
      }
    }
  }
  for (const lettering of Object.values(data.lettering)) {
    if (!Array.isArray(lettering) || lettering.some((line) => line !== null &&
      (!object(line) || typeof line.text !== 'string' || !box(line.box) ||
        typeof line.size !== 'number' || !Number.isFinite(line.size) ||
        typeof line.angle !== 'number' || !Number.isFinite(line.angle)))) {
      throw new Error('Saved lettering is damaged; the project has not been overwritten.')
    }
  }
  const saved = value as StoredProject
  return {
    ...saved.data,
    images: saved.data.images.map(({ file, fileName, fileLastModified, ...image }) => ({
      ...image,
      file: new File([file], fileName, { type: file.type, lastModified: fileLastModified }),
    })),
  }
}

export function loadProject(database = DATABASE): Promise<ProjectData | null> {
  return ordered(async () => {
    const db = await open(database)
    try {
      const stored = await new Promise<unknown>((resolve, reject) => {
        const transaction = db.transaction('project', 'readonly')
        const request = transaction.objectStore('project').get('current')
        transaction.oncomplete = () => resolve(request.result)
        transaction.onabort = () => reject(transaction.error ?? new Error('Project loading was aborted'))
      })
      return stored === undefined ? null : decode(stored)
    } finally {
      db.close()
    }
  })
}

export function saveProject(data: ProjectData, database = DATABASE): Promise<void> {
  return ordered(async () => {
    const stored: StoredProject = {
      version: 1,
      data: {
        ...data,
        images: data.images.map(({ id, file, name, size, addedAt, width, height, folder }) => ({
          id, file, name, size, addedAt, width, height, folder,
          fileName: file.name,
          fileLastModified: file.lastModified,
        })),
      },
    }
    const db = await open(database)
    try {
      await new Promise<void>((resolve, reject) => {
        const transaction = db.transaction('project', 'readwrite')
        transaction.objectStore('project').put(stored, 'current')
        transaction.oncomplete = () => resolve()
        transaction.onabort = () => reject(transaction.error ?? new Error('Project saving was aborted'))
      })
    } finally {
      db.close()
    }
  })
}
