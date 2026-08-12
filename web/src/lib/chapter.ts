import type { Lettering } from './api'
import { compose } from './compose'
import type { GalleryFolder, GalleryImage } from './images'
import { stem } from './images'
import type { Packed } from './zip'

/**
 * A finished chapter, put back into the archive it came out of.
 *
 * Every page goes in, at the best state it reached: lettered where it was
 * translated, cleaned where it was only cleaned, and untouched where it was
 * neither. A chapter with a gap in it is not a chapter — a page that fell over
 * is still a page of the story, and leaving it out renumbers everything after
 * it.
 */

/** How a page ended up, which is what decides what goes into the archive. */
export type Reached = 'lettered' | 'cleaned' | 'original'

/** The bytes behind an object URL, which is where a cleaned page is kept. */
async function bytesOf(blob: Blob): Promise<Uint8Array> {
  return new Uint8Array(await blob.arrayBuffer())
}

/**
 * The best version of one page, named for the archive.
 *
 * A page that was drawn on is a PNG, whatever it arrived as, so its name says
 * PNG. A page that was not is passed through exactly as it came — the same bytes
 * under the same name, since re-encoding a JPEG nobody touched would cost
 * quality for nothing.
 */
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

/** A name safe to save under, out of a language a person typed. */
function slug(said: string): string {
  return (
    said
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '') || 'translated'
  )
}

/**
 * What the finished chapter is called: the archive's own name, said to be this
 * rather than the original, and kept as whatever kind of archive it arrived as.
 *
 * `.cbz` and `.zip` are the same format under two names, and which one a reader
 * picks up can depend on the extension — so a chapter that came in as one goes
 * back out as one.
 */
export function archiveName(
  folder: GalleryFolder,
  target: string,
  anyLettered: boolean,
): string {
  const kind = /\.cbz$/i.test(folder.archive ?? '') ? 'cbz' : 'zip'
  const said = anyLettered ? slug(target) : 'cleaned'
  return `${folder.name}-${said}.${kind}`
}
