import type { Lettering } from './api'
import { LINE_HEIGHT, fontFor, ready, wrap } from './fit'

/**
 * The page with the translations set into it, as a PNG.
 *
 * Drawn with the same font, the same sizes and the same wrapping the board
 * previews with, so what was arranged there is what comes out. The base is
 * whatever the board is showing — the cleaned page if there is one, since
 * lettering over words that were never hidden only stacks one on the other.
 */
export async function compose(
  source: string,
  width: number,
  height: number,
  lettering: (Lettering | null)[],
): Promise<Blob> {
  await ready()

  const page = await loadImage(source)
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height

  const context = canvas.getContext('2d')
  if (!context) throw new Error('this browser has no 2D canvas')

  context.drawImage(page, 0, 0, width, height)
  context.fillStyle = '#000'
  context.textAlign = 'center'
  context.textBaseline = 'middle'

  for (const line of lettering) {
    if (line === null || !line.text.trim()) continue
    set(context, line)
  }

  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) =>
        blob ? resolve(blob) : reject(new Error('the page could not be saved')),
      'image/png',
    )
  })
}

/** One translated line, centred in its box the way the board centres it. */
function set(context: CanvasRenderingContext2D, line: Lettering) {
  const [x0, y0, x1, y1] = line.box
  context.font = fontFor(line.size)

  const lines = wrap(context, line.text.trim(), x1 - x0)
  const step = line.size * LINE_HEIGHT
  const middleX = (x0 + x1) / 2
  // The whole block is centred on the box, so the first line starts half a
  // block above the middle — text too tall for its box overruns it evenly,
  // above and below, as it does on the board.
  const top = (y0 + y1) / 2 - ((lines.length - 1) * step) / 2

  lines.forEach((text, index) => {
    context.fillText(text, middleX, top + index * step)
  })
}

function loadImage(source: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = () => reject(new Error('the page could not be read back'))
    image.src = source
  })
}

/** Hand a finished page to the browser to save. */
export function save(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = name
  link.click()
  // Revoked late: pulling the URL out from under the download can cancel it.
  setTimeout(() => URL.revokeObjectURL(url), 60_000)
}
