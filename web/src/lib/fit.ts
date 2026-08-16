
export const FONT_STACK =
  '"Anime Ace", "Comic Sans MS", "Helvetica Neue", Arial, sans-serif'
export const FONT_WEIGHT = 400
export const LINE_HEIGHT = 1.15

export const SIZE_MIN = 6
export const SIZE_MAX = 200

const HYPHEN = '-'

const STROKE_RATIO = 0.12

const LATIN = 1.25

export function strokeFor(size: number): number {
  return Math.max(1, size * STROKE_RATIO)
}

export function fontFor(size: number) {
  return `${FONT_WEIGHT} ${size}px ${FONT_STACK}`
}

let scratch: CanvasRenderingContext2D | null = null

function measurer(): CanvasRenderingContext2D | null {
  if (!scratch) scratch = document.createElement('canvas').getContext('2d')
  return scratch
}

export function ready(): Promise<unknown> {
  if (typeof document === 'undefined' || !document.fonts) return Promise.resolve()
  return document.fonts.load(fontFor(16)).catch(() => undefined)
}

function pieces(
  context: CanvasRenderingContext2D,
  word: string,
  width: number,
): string[] {
  const parts: string[] = []
  let part = ''
  for (const letter of word) {
    const candidate = part + letter
    if (part && context.measureText(candidate + HYPHEN).width > width) {
      parts.push(part + HYPHEN)
      part = letter
    } else {
      part = candidate
    }
  }
  if (part) parts.push(part)
  return parts
}

function wrap(
  context: CanvasRenderingContext2D,
  text: string,
  width: number,
): string[] {
  const lines: string[] = []

  for (const paragraph of text.split('\n')) {
    let line = ''
    for (const word of paragraph.split(/\s+/).filter(Boolean)) {
      if (context.measureText(word).width > width) {
        if (line) {
          lines.push(line)
          line = ''
        }
        const parts = pieces(context, word, width)
        lines.push(...parts.slice(0, -1))
        line = parts[parts.length - 1] ?? ''
        continue
      }

      const candidate = line ? `${line} ${word}` : word
      if (line && context.measureText(candidate).width > width) {
        lines.push(line)
        line = word
      } else {
        line = candidate
      }
    }
    lines.push(line)
  }

  return lines.filter((line, index) => line !== '' || index === 0)
}

export function linesFor(text: string, width: number, size: number): string[] {
  const context = measurer()
  const words = text.trim()
  if (!context || !words) return words ? [words] : []
  context.font = fontFor(size)
  return wrap(context, words, width)
}

function widestWord(context: CanvasRenderingContext2D, text: string): number {
  let widest = 0
  for (const word of text.split(/\s+/).filter(Boolean)) {
    widest = Math.max(widest, context.measureText(word).width)
  }
  return widest
}

function lands(
  context: CanvasRenderingContext2D,
  text: string,
  width: number,
  height: number,
  size: number,
  whole: boolean,
): boolean {
  context.font = fontFor(size)
  if (whole && widestWord(context, text) > width) return false

  const lines = wrap(context, text, width)
  return (
    lines.length * size * LINE_HEIGHT <= height &&
    lines.every((line) => context.measureText(line).width <= width)
  )
}

export function originalSize(
  original: string,
  width: number,
  height: number,
): number {
  const characters = original.replace(/\s+/g, '').length
  if (!characters || width <= 0 || height <= 0) return SIZE_MAX
  return Math.max(
    SIZE_MIN,
    Math.round(LATIN * Math.sqrt((width * height) / characters)),
  )
}

const SAMPLE = 'abcdefghijklmnopqrstuvwxyz '

const PACKING = 0.85

export function roomInCharacters(
  width: number,
  height: number,
  size: number,
): number {
  const context = measurer()
  if (!context || width <= 0 || height <= 0 || size <= 0) return 0
  context.font = fontFor(size)
  const character = context.measureText(SAMPLE).width / SAMPLE.length
  if (character <= 0) return 0
  const rows = Math.max(1, Math.floor(height / (size * LINE_HEIGHT)))
  return Math.max(0, Math.floor((width / character) * rows * PACKING))
}

function largestThatLands(
  context: CanvasRenderingContext2D,
  text: string,
  width: number,
  height: number,
  whole: boolean,
  most: number,
): number | null {
  let best: number | null = null
  let low = SIZE_MIN
  let high = Math.min(most, Math.max(SIZE_MIN, Math.floor(height)))

  while (low <= high) {
    const size = Math.floor((low + high) / 2)
    if (lands(context, text, width, height, size, whole)) {
      best = size
      low = size + 1
    } else {
      high = size - 1
    }
  }
  return best
}

export function fitSize(
  text: string,
  width: number,
  height: number,
  most: number = SIZE_MAX,
): number {
  const context = measurer()
  const words = text.trim()
  if (!context || !words || width <= 0 || height <= 0) return SIZE_MIN

  const ceiling = Math.max(SIZE_MIN, Math.min(SIZE_MAX, Math.round(most)))
  return (
    largestThatLands(context, words, width, height, true, ceiling) ??
    largestThatLands(context, words, width, height, false, ceiling) ??
    SIZE_MIN
  )
}
