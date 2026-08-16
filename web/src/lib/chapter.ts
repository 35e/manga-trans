import type { Lettering } from './api'
import { compose } from './compose'
import type { GalleryFolder, GalleryImage } from './images'
import { stem } from './images'
import type { Packed } from './zip'

export type Reached = 'lettered' | 'cleaned' | 'original'

async function bytesOf(blob: Blob): Promise<Uint8Array> {
  return new Uint8Array(await blob.arrayBuffer())
}

export async function finished(
  page: GalleryImage,
  lettering: (Lettering | null)[] | undefined,
  cleaned: string | null,
): Promise<Packed & { reached: Reached }> {
  const lettered = lettering?.some((line) => line !== null && line.text.trim())

  if (lettered) {
    const drawn = await compose(
      cleaned ?? page.url,
      page.width,
      page.height,
      lettering ?? [],
    )
    return {
      name: `${stem(page.name)}.png`,
      bytes: await bytesOf(drawn),
      reached: 'lettered',
    }
  }

  if (cleaned) {
    const blob = await fetch(cleaned).then((answer) => answer.blob())
    return {
      name: `${stem(page.name)}.png`,
      bytes: await bytesOf(blob),
      reached: 'cleaned',
    }
  }

  return { name: page.name, bytes: await bytesOf(page.file), reached: 'original' }
}

function slug(said: string): string {
  return (
    said
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '') || 'translated'
  )
}

export function archiveName(
  folder: GalleryFolder,
  target: string,
  anyLettered: boolean,
): string {
  const kind = /\.cbz$/i.test(folder.archive ?? '') ? 'cbz' : 'zip'
  const said = anyLettered ? slug(target) : 'cleaned'
  return `${folder.name}-${said}.${kind}`
}
