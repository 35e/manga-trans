import type { Analysis, Lettering } from './api'
import { compose } from './compose'
import type { GalleryFolder, GalleryImage } from './images'
import { stem } from './images'
import type { Packed } from './zip'

export type Reached = 'lettered' | 'cleaned' | 'original'

export type ChapterContext = {
  pages: Pick<GalleryImage, 'id' | 'name'>[]
  readings: Record<string, Analysis>
  translations: Record<string, { source: string; translated: string }[]>
}

type ReferencePage = { at: number; heading: string; lines: string[] }

function referenceSection(
  title: string,
  pages: ReferencePage[],
  total: number,
  budget: number,
): string {
  const omission = '\n[Some pages omitted: unread, empty, excluded, or reference limit.]'
  const truncated = '\n[Further lines/pairs omitted: reference truncated.]'
  const size = pages.reduce(
    (length, page) => length + page.heading.length + 1 +
      page.lines.reduce((sum, line) => sum + line.length + 1, 0),
    title.length + (pages.length < total ? omission.length : 0),
  )
  const overflow = size > budget
  let remaining = budget - title.length -
    (overflow || pages.length < total ? omission.length : 0)
  const kept: { at: number; text: string }[] = []
  for (const page of pages) {
    let text = `${page.heading}\n${page.lines.join('\n')}`
    // On overflow, preserve room for both nearby pages or early and recent examples.
    const room = overflow && kept.length === 0 && pages.length > 1
      ? Math.floor(remaining / 2)
      : remaining
    if (text.length + 1 > room) {
      text = page.heading
      let included = 0
      for (const line of page.lines) {
        if (text.length + line.length + truncated.length + 2 > room) continue
        text += `\n${line}`
        included++
      }
      if (!included) continue
      text += truncated
    }
    kept.push({ at: page.at, text })
    remaining -= text.length + 1
  }
  kept.sort((one, other) => one.at - other.at)
  return (
    title +
    kept.map((page) => `\n${page.text}`).join('') +
    (kept.length < total ? omission : '')
  )
}

export function chapterReference(chapter: ChapterContext, current: string): string {
  const at = chapter.pages.findIndex((page) => page.id === current)
  const sources: ReferencePage[] = []
  const history: ReferencePage[] = []
  for (const [index, page] of chapter.pages.entries()) {
    if (page.id === current) continue
    const reading = chapter.readings[page.id]
    const texts = reading?.texts?.filter(
      (text, line) => text.trim() && !reading.excluded.includes(line),
    )
    if (texts?.length) {
      sources.push({
        at: index,
        heading: `Page ${index + 1} ${JSON.stringify(page.name)}:`,
        lines: texts.map((text) => JSON.stringify(text)),
      })
    }
    const pairs = chapter.translations[page.id]
    if (index < at && pairs?.length) {
      history.push({
        at: index,
        heading: `Page ${index + 1} ${JSON.stringify(page.name)}:`,
        lines: pairs.map((pair) => JSON.stringify(pair)),
      })
    }
  }
  sources.sort(
    (one, other) => Math.abs(one.at - at) - Math.abs(other.at - at) || one.at - other.at,
  )
  const examples: ReferencePage[] = []
  for (let first = 0, last = history.length - 1; first <= last; first++, last--) {
    examples.push(history[first])
    if (first < last) examples.push(history[last])
  }
  // ponytail: 2000 reference characters leave room in local 4k-token models; use retrieval for larger chapters.
  const historyTitle = 'Earlier successful translations:'
  const historySize = historyTitle.length + history.reduce(
    (size, page) => size + page.heading.length + 1 +
      page.lines.reduce((length, line) => length + line.length + 1, 0),
    0,
  )
  const sourceReference = referenceSection(
    'Chapter OCR reference (other pages):',
    sources,
    chapter.pages.length - 1,
    Math.max(999, 1998 - historySize),
  )
  return [
    sourceReference,
    referenceSection(historyTitle, examples, history.length, 1998 - sourceReference.length),
  ].join('\n\n')
}

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
