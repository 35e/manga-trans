/** The back end, which lives on its own port and lets this origin in. */
export const API_BASE: string =
  import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

/** A box is [x0, y0, x1, y1] in image pixels. */
export type Box = [number, number, number, number]

/**
 * One language a page can be lettered in. The list comes from the API rather
 * than being held here: which reader exists for what is its business.
 *
 * `rtl` is which way the page is read across itself, which is the order the
 * blocks come back in and so the order the page is translated in.
 */
export type Language = { code: string; name: string; rtl: boolean }

/** What the API reads a page as when it is told nothing. */
export const LANGUAGE_DEFAULT = 'ja'

export type Region = {
  /**
   * This block, for as long as it exists. Blocks are resized, reordered and
   * added to by hand, so a place in the list is not a name: anything that goes
   * away and comes back finds its block by this.
   */
  id: string
  box: Box
  confidence: number
  /**
   * The room the block was written in: the largest rectangle inside the balloon
   * around it, which is where a translation goes. Null where no balloon could be
   * made out, and then `box` is all there is to go on.
   */
  bubble?: Box | null
  /** Drawn by hand, because the detector missed it. */
  manual?: boolean
}

export type Detection = {
  width: number
  height: number
  regions: Region[]
}

/** What is known about one page. `texts` is null until the reader has been. */
export type Analysis = {
  detection: Detection
  texts: string[] | null
  /**
   * Blocks to leave alone, by their index in `detection.regions`. A detector
   * that boxed something worth keeping is corrected here rather than by
   * re-detecting.
   */
  excluded: number[]
}

/** What the API is busy doing for the page on the board. */
export type Stage =
  | 'detecting'
  | 'reading'
  | 'tracing'
  | 'cleaning'
  | 'translating'
  | 'fitting'

/** What the board is being used for: looking, hiding, or lettering. */
export type BoardMode = 'inspect' | 'mask' | 'translate'

/**
 * What goes where the lettering was. `art` fills it in from the page around it;
 * `white` paints it flat, which is only right where the ground was white.
 */
export type Fill = 'art' | 'white'

/** One translated line, set where the original was. */
export type Lettering = {
  text: string
  box: Box
  size: number
  /**
   * Degrees clockwise about the middle of the box. The box itself stays square
   * to the page, so wrapping and fitting go on measuring what they always did.
   */
  angle: number
}

/**
 * Below this the detector is guessing, and a block it is guessing at starts left
 * alone rather than cleaned: a box over a piece of artwork does more harm hidden
 * than a real one does missed. Putting one back is one click either way.
 */
export const UNSURE = 0.8

/** What went wrong, as a line to show. Everything thrown here is an Error. */
export function said(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause)
}

/** Every error the API raises comes back as {"error": "..."}. */
async function refuse(response: Response): Promise<never> {
  const said = await response
    .json()
    .then((body: { error?: string }) => body.error)
    .catch(() => undefined)
  throw new Error(said ?? `The API answered ${response.status}`)
}

/** A fetch that says where it was trying to go when there is nothing there. */
async function reach(path: string, init?: RequestInit): Promise<Response> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, init)
  } catch (cause) {
    throw new Error(`Could not reach the API at ${API_BASE}`, { cause })
  }
  if (!response.ok) await refuse(response)
  return response
}

/**
 * One page up, one answer back. Anything that is not a Blob or a word goes up as
 * JSON, which is how the API takes boxes.
 */
async function send(
  path: string,
  file: File,
  parts: Record<string, unknown> = {},
): Promise<Response> {
  const body = new FormData()
  body.append('image', file, file.name)
  for (const [name, value] of Object.entries(parts)) {
    if (value instanceof Blob) body.append(name, value, `${name}.png`)
    // A word the API reads as a word would arrive still wearing its quotes.
    else if (typeof value === 'string') body.append(name, value)
    else body.append(name, JSON.stringify(value))
  }
  return reach(path, { method: 'POST', body })
}

/**
 * Every block of lettering the detector finds on one page.
 *
 * Finding it needs no model of the language, but the order the blocks come back
 * in does: Japanese is read right to left across the page and Korean left to
 * right.
 */
export async function detect(file: File, language: string): Promise<Detection> {
  const response = await send('/api/detect', file, { language })
  const found = (await response.json()) as {
    width: number
    height: number
    regions: Omit<Region, 'id'>[]
  }
  // The API knows nothing of ids — it answers the same page the same way twice.
  // Blocks are named here, once, as they arrive.
  return {
    ...found,
    regions: found.regions.map((region) => ({ ...region, id: crypto.randomUUID() })),
  }
}

/**
 * The balloon each box is written in, or null where none could be made out.
 * `detect` already answers with these, so this is for the blocks it did not
 * find: one drawn by hand, one split in two, one pulled off its neighbour.
 */
export async function bubbles(file: File, boxes: Box[]): Promise<(Box | null)[]> {
  if (boxes.length === 0) return []
  const response = await send('/api/bubbles', file, { boxes })
  const answer = (await response.json()) as { regions: { bubble: Box | null }[] }
  return answer.regions.map((region) => region.bubble)
}

/**
 * What each box says, in the order the boxes were given. `language` is what the
 * page is lettered in, and decides which reader the API stands up.
 */
export async function read(
  file: File,
  boxes: Box[],
  language: string,
): Promise<string[]> {
  if (boxes.length === 0) return []
  const response = await send('/api/read', file, { boxes, language })
  const { texts } = (await response.json()) as { texts: string[] }
  return texts
}

/** Every language the API can read a page in. */
export async function languages(): Promise<Language[]> {
  const response = await reach('/api/languages')
  const answer = (await response.json()) as { languages: Language[] }
  return answer.languages
}

/** Every model the API's Ollama has to translate with. */
export async function models(): Promise<string[]> {
  const response = await reach('/api/models')
  const answer = (await response.json()) as { models: string[] }
  return answer.models
}

/**
 * What the API tells the model when it is not told anything else. The API keeps
 * no settings, so a front end that wants its own asks for this, lets it be
 * edited, and sends the edit back each time.
 */
export async function defaultPrompt(): Promise<string> {
  const response = await reach('/api/prompt')
  const answer = (await response.json()) as { prompt: string }
  return answer.prompt
}

/**
 * One translation per text, in order. The whole page goes over at once: a line
 * of manga read on its own often cannot be translated at all.
 *
 * `system` is what the model is told; leave it out for the API's own. `source`
 * is the language the page was lettered in, as a word rather than a code —
 * worth saying, since the same characters are Japanese or Chinese depending on
 * nothing a model can see from one line of dialogue.
 */
export async function translate(
  texts: string[],
  model: string,
  target: string,
  system?: string | null,
  source?: string | null,
): Promise<string[]> {
  if (texts.length === 0) return []

  const body = new FormData()
  body.append('texts', JSON.stringify(texts))
  body.append('model', model)
  body.append('target', target)
  if (system) body.append('system', system)
  if (source) body.append('source', source)

  const response = await reach('/api/translate', { method: 'POST', body })
  const answer = (await response.json()) as { texts: string[] }
  return answer.texts
}

/**
 * The lettering itself, pixel by pixel: a page-sized PNG, opaque white on the
 * ink and clear everywhere else. `grow` is how many pixels to spread it by.
 */
export async function letterMask(file: File, grow?: number): Promise<Blob> {
  const response = await send(
    '/api/letters',
    file,
    grow === undefined ? {} : { grow },
  )
  return response.blob()
}

/**
 * The page with everything the mask marks taken out of it, as a PNG. The mask is
 * page-sized, white where the lettering should go.
 */
export async function clean(file: File, mask: Blob, fill: Fill): Promise<Blob> {
  const response = await send('/api/clean', file, { mask, fill })
  return response.blob()
}
