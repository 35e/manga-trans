import type { Lettering } from './api'
import { LINE_HEIGHT, fontFor, linesFor, ready, strokeFor } from './fit'

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
  context.strokeStyle = '#fff'
  context.textAlign = 'center'
  context.textBaseline = 'middle'
  context.lineJoin = 'round'
  context.miterLimit = 2

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

function set(context: CanvasRenderingContext2D, line: Lettering) {
  const [x0, y0, x1, y1] = line.box
  context.font = fontFor(line.size)
  context.lineWidth = strokeFor(line.size) * 2

  const lines = linesFor(line.text, x1 - x0, line.size)
  const step = line.size * LINE_HEIGHT

  context.save()
  context.translate((x0 + x1) / 2, (y0 + y1) / 2)
  if (line.angle) context.rotate((line.angle * Math.PI) / 180)

  const top = -((lines.length - 1) * step) / 2

  lines.forEach((text, index) => {
    const y = top + index * step
    context.strokeText(text, 0, y)
    context.fillText(text, 0, y)
  })
  context.restore()
}

function loadImage(source: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = () => reject(new Error('the page could not be read back'))
    image.src = source
  })
}

export function save(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = name
  link.click()
  setTimeout(() => URL.revokeObjectURL(url), 60_000)
}
