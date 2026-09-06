import { finished } from './chapter'
import { fingerprint, loadImage } from './images'
import { expand, pack } from './zip'

// Browser console has no static imports: await (await import('/src/lib/zip.test.ts')).checkArchiveIdentity()
export async function checkArchiveIdentity() {
  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = 1
  const bytes = new Uint8Array(await (await fetch(canvas.toDataURL())).arrayBuffer())
  const archive = new File([await pack([
    { name: 'a/001.png', bytes },
    { name: 'b/001.png', bytes },
  ])], 'chapter.cbz', { lastModified: 1234 })
  const pages = await expand(archive)
  const identities = pages.map((page) => fingerprint(page, 'chapter'))
  if (pages.length !== 2 || new Set(identities).size !== 2) {
    throw new Error('Equal-sized pages in different archive directories must survive deduplication')
  }
  const repeated = await expand(archive)
  if (repeated.length !== 2 || repeated.some((page, at) => fingerprint(page, 'chapter') !== identities[at])) {
    throw new Error('Importing the same archive again must retain its duplicate identities')
  }

  const images = await Promise.all(pages.map((page) => loadImage(page, 'chapter')))
  try {
    if (images.some((image) => image === null || image.name !== '001.png')) {
      throw new Error('Archive pages must decode and display their basename')
    }
    const packed = await pack(await Promise.all(images.map((image) => finished(image!, undefined, null))))
    const exported = await expand(new File([packed], 'export.cbz'))
    if (exported.length !== 2 || new Set(exported.map((page) => page.name)).size !== 2) {
      throw new Error('Export must retain both pages despite matching display names')
    }
    for (const page of exported) {
      const held = new Uint8Array(await page.arrayBuffer())
      if (held.length !== bytes.length || held.some((byte, at) => byte !== bytes[at])) {
        throw new Error('Export must preserve the original page bytes')
      }
    }
    return { imported: pages.length, exported: exported.map((page) => page.name) }
  } finally {
    for (const image of images) if (image) URL.revokeObjectURL(image.url)
  }
}
