import { Mask } from './mask'
import { loadProject, saveProject } from './project'
import type { ProjectData } from './project'

// Browser console: await (await import('/src/lib/project.test.ts')).checkProjectRoundtrip()
export async function checkProjectRoundtrip() {
  const database = `manga-trans-check-${crypto.randomUUID()}`
  try {
    if (await loadProject(database) !== null) throw new Error('A new project must start absent, not saved empty')
    const mask = new Mask(8, 8)
    mask.boxes([[2, 2, 6, 6]])
    const png = await mask.snapshot()
    if (await mask.snapshot() !== png) throw new Error('Unchanged masks must reuse their PNG')
    const source = new File([png], 'chapter/001.png', { type: 'image/png', lastModified: 123456 })
    const data: ProjectData = {
      images: [{ id: 'page', file: source, name: '001.png', size: source.size, addedAt: 42, width: 8, height: 8, folder: 'folder' }],
      folders: [{ id: 'folder', name: 'Chapter', addedAt: 24, archive: 'chapter.cbz', manual: true }],
      analyses: { page: { detection: { width: 8, height: 8, regions: [{ id: 'region', box: [2, 2, 6, 6], confidence: 1 }] }, texts: ['source'], excluded: [] } },
      lettering: { page: [{ text: 'translated', box: [1, 2, 7, 6], size: 12, angle: 5 }] },
      cleaned: { page: png }, masks: { page: png }, touchups: { page: png },
      activeId: 'page', openFolder: 'folder', language: 'ja', model: 'test-model', target: 'en', prompt: 'test prompt', spread: 3, fill: 'telea',
    }
    const saving = saveProject(data, database)
    const loaded = await loadProject(database)
    await saving
    if (!loaded) throw new Error('A queued load must see the preceding committed save')
    const page = loaded.images[0]
    if (!(page.file instanceof File) || page.file.name !== source.name || page.file.lastModified !== source.lastModified ||
      page.file.type !== source.type || page.id !== 'page' || page.folder !== 'folder' || 'url' in page ||
      (await page.file.arrayBuffer()).byteLength !== source.size) {
      throw new Error('Reload must retain source File metadata and page identity without object URLs')
    }
    for (const key of ['folders', 'analyses', 'lettering', 'activeId', 'openFolder', 'language', 'model', 'target', 'prompt', 'spread', 'fill'] as const) {
      if (JSON.stringify(loaded[key]) !== JSON.stringify(data[key])) throw new Error(`Reload lost ${key}`)
    }
    for (const blobs of [loaded.cleaned, loaded.masks, loaded.touchups]) {
      const bitmap = await createImageBitmap(blobs.page)
      const canvas = document.createElement('canvas')
      canvas.width = canvas.height = 8
      const ctx = canvas.getContext('2d')!
      try {
        ctx.drawImage(bitmap, 0, 0)
        if (ctx.getImageData(0, 0, 1, 1).data[3] !== 0 || ctx.getImageData(3, 3, 1, 1).data[3] !== 255) {
          throw new Error('Project masks must preserve transparent background and opaque brush pixels')
        }
      } finally {
        bitmap.close()
      }
    }
    const restored = await Mask.restore(loaded.masks.page)
    if (restored.empty) throw new Error('Restored marks must remain editable')
    restored.boxes([[3, 3, 4, 4]], true)
    const erased = await createImageBitmap(await restored.snapshot())
    const canvas = document.createElement('canvas')
    canvas.width = canvas.height = 8
    const ctx = canvas.getContext('2d')!
    try {
      ctx.drawImage(erased, 0, 0)
      if (ctx.getImageData(3, 3, 1, 1).data[3] !== 0 || ctx.getImageData(2, 2, 1, 1).data[3] !== 255) {
        throw new Error('Erasing a restored mask must remove only the selected pixels')
      }
    } finally {
      erased.close()
    }
    await saveProject({ ...loaded, images: [], folders: [], analyses: {}, lettering: {}, cleaned: {}, masks: {}, touchups: {}, activeId: null, openFolder: null }, database)
    const deleted = await loadProject(database)
    if (!deleted || deleted.images.length || deleted.folders.length || Object.keys(deleted.masks).length || deleted.activeId !== null) {
      throw new Error('Deleted pages and masks must not return on reload')
    }
    return { file: page.file.name, transparency: 'preserved', masks: 'editable', deletion: 'persisted' }
  } finally {
    // ES2023 target: Promise.withResolvers is not available.
    await new Promise<void>((resolve, reject) => {
      const request = indexedDB.deleteDatabase(database)
      request.onsuccess = () => resolve()
      request.onerror = () => reject(request.error)
      request.onblocked = () => reject(new Error('Temporary project database cleanup was blocked'))
    })
  }
}
