/**
 * How big the lettering can be set and still land in its box.
 *
 * Measured with the same font the board draws it in, so what is worked out here
 * and what the browser lays out agree. Both wrap greedily on whitespace.
 */

export const FONT_STACK =
  '"Anime Ace", "Comic Sans MS", "Helvetica Neue", Arial, sans-serif'
export const FONT_WEIGHT = 400
export const LINE_HEIGHT = 1.15

export const SIZE_MIN = 6
export const SIZE_MAX = 200

/**
 * The white the lettering is outlined in, as a share of the type size, so it
 * holds up the same whether the line is set large or small. Black words over
 * black artwork are unreadable without it.
 */
export const STROKE_RATIO = 0.09

export function strokeFor(size: number): number {
  return Math.max(1, size * STROKE_RATIO)
}

let scratch: CanvasRenderingContext2D | null = null

function measurer(): CanvasRenderingContext2D | null {
  if (scratch) return scratch
  const context = document.createElement('canvas').getContext('2d')
  scratch = context
  return context
}

export function fontFor(size: number) {
  return `${FONT_WEIGHT} ${size}px ${FONT_STACK}`
}

/**
 * Wait for the lettering face to be in.
 *
 * A font is only fetched when something needs it, and measuring before it
 * arrives measures the fallback — which would fit the text to the wrong shape
 * and set it at the wrong size. Anything that measures or draws waits on this
 * first.
 */
export function ready(): Promise<unknown> {
  if (typeof document === 'undefined' || !document.fonts) return Promise.resolve()
  return document.fonts.load(fontFor(16)).catch(() => undefined)
}

/** Left at the end of a line to say the word carries on below. */
export const HYPHEN = '-'

/**
 * A word too wide for the box, cut into pieces that fit, each but the last
 * ending in a hyphen so it reads as a word broken rather than a word ending.
 *
 * The hyphen is measured along with the piece it hangs off, or it would be the
 * thing that overruns the box.
 */
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
  // A single letter wider than the box still goes on a line of its own: there
  // is nowhere smaller to put it.
  if (part) parts.push(part)
  return parts
}

/** Greedy wrap on whitespace, and inside a word when the word alone is too wide. */
export function wrap(
  context: CanvasRenderingContext2D,
  text: string,
  width: number,
): string[] {
  const lines: string[] = []

  for (const paragraph of text.split('\n')) {
    let line = ''
    for (const word of paragraph.split(/\s+/).filter(Boolean)) {
      if (context.measureText(word).width > width) {
        // Too wide however it is placed, so break it apart. What was already on
        // the line goes first, and the word's last piece carries on.
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

/**
 * The lines `text` breaks into inside a box `width` wide, set at `size`.
 *
 * The board lays these out one to a line rather than leaving the wrapping to
 * the browser, and the page is drawn from the same call, so what is arranged
 * and what comes out break in the same places — hyphens and all.
 */
export function linesFor(text: string, width: number, size: number): string[] {
  const context = measurer()
  const words = text.trim()
  if (!context || !words) return words ? [words] : []
  context.font = fontFor(size)
  return wrap(context, words, width)
}

/**
 * The largest size that fits `text` inside a box of `width` × `height`, in the
 * same pixels the box is measured in. Never smaller than SIZE_MIN: text too
 * long for its box is set small and left to overrun rather than dropped, which
 * is what the API does when it letters a page too.
 */
export function fitSize(text: string, width: number, height: number): number {
  const context = measurer()
  const words = text.trim()
  if (!context || !words || width <= 0 || height <= 0) return SIZE_MIN

  let best = SIZE_MIN
  let low = SIZE_MIN
  let high = Math.min(SIZE_MAX, Math.max(SIZE_MIN, Math.floor(height)))

  while (low <= high) {
    const size = Math.floor((low + high) / 2)
    context.font = fontFor(size)
    const lines = wrap(context, words, width)
    const fits =
      lines.length * size * LINE_HEIGHT <= height &&
      lines.every((line) => context.measureText(line).width <= width)

    if (fits) {
      best = size
      low = size + 1
    } else {
      high = size - 1
    }
  }
  return best
}
