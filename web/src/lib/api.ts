/**
 * The back end. Keep this `??` and not `||`: under compose the dashboard and the
 * API share an origin and `VITE_API_URL` is the empty string, which `||` would
 * collapse back to the absolute URL.
 */
export const API_BASE: string =
  import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

/** A box is [x0, y0, x1, y1] in image pixels. */
export type Box = [number, number, number, number]

/** One language a page can be lettered in. The list comes from the API. */
export type Language = { code: string; name: string; rtl: boolean }

/** What the API reads a page as when it is told nothing. */
export const LANGUAGE_DEFAULT = 'ja'

/** Lettering inside a balloon, or lettering over the art. */
export type Kind = 'speech' | 'free'

export type Region = {
  /** A place in the list is not a name: anything asynchronous finds it by this. */
  id: string
  box: Box
  confidence: number
  /** Speech or not, where the detector said. Absent on a block drawn by hand. */
  kind?: Kind
  /** The room the block was written in, and where a translation goes. */
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
  /** Blocks to leave alone, by their index in `detection.regions`. */
  excluded: number[]
}

/** What the API is busy doing for the page on the board. */
export type Stage =
  | 'detecting'
  | 'reading'
  | 'tracing'
  | 'cleaning'
  | 'translating'
  | 'surveying'

/** What the board is being used for: looking, hiding, or lettering. */
export type BoardMode = 'inspect' | 'mask' | 'translate'

/** What goes where the lettering was: LaMa, Telea, or flat white. */
export type Fill = 'art' | 'telea' | 'white'

/** One translated line, set where the original was. */
export type Lettering = {
  text: string
  box: Box
  size: number
  /** Degrees clockwise about the middle of the box, which stays square. */
  angle: number
}

/**
 * Below this a block starts left alone rather than cleaned: a box over a piece
 * of artwork does more harm hidden than a real one does missed.
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

/** One page up, one answer back. Anything not a Blob or a word goes up as JSON. */
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

/** Every block of lettering the detector finds on one page, in reading order. */
export async function detect(file: File, language: string): Promise<Detection> {
  const response = await send('/api/detect', file, { language })
  const found = (await response.json()) as {
    width: number
    height: number
    regions: Omit<Region, 'id'>[]
  }
  // The API knows nothing of ids, so blocks are named here as they arrive.
  return {
    ...found,
    regions: found.regions.map((region) => ({ ...region, id: crypto.randomUUID() })),
  }
}

/**
 * The balloon each box is written in, or null where none could be made out.
 * Send **every** box on the page: the answer depends on which others were asked.
 */
export async function bubbles(file: File, boxes: Box[]): Promise<(Box | null)[]> {
  if (boxes.length === 0) return []
  const response = await send('/api/bubbles', file, { boxes })
  const answer = (await response.json()) as { regions: { bubble: Box | null }[] }
  return answer.regions.map((region) => region.bubble)
}

/** What each box says, in the order the boxes were given. */
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

/** What the API tells the model when it is not told anything else. */
export async function defaultPrompt(): Promise<string> {
  const response = await reach('/api/prompt')
  const answer = (await response.json()) as { prompt: string }
  return answer.prompt
}

/** A name or coinage, and the wording settled on for it. */
export type Term = { source: string; target: string; note?: string }

/** `unknown` is a real answer, and the one wanted until a page shows otherwise. */
export type Gender = 'male' | 'female' | 'unknown'

/** The facts about one of the cast that can be held, and so corrected. */
export type Fact = 'gender' | 'note'

export type CastMember = {
  name: string
  gender: Gender
  note?: string
  /**
   * Facts set by hand rather than worked out from a page. The API tells the model
   * those are not its to change, and {@link merged} will not move them either.
   */
  settled?: Fact[]
}

/** Where a chapter has got to: what is going on, and who it is going on between. */
export type Story = { scene: string; cast: CastMember[] }

/**
 * A chapter read whole before a word of it is translated.
 *
 * `beats` is one line per page, indexed by the page's place in its folder, so a
 * folder whose pages changed since has a bible that no longer lines up — see
 * `bible.fits`. It must not be translated against.
 */
export type Bible = {
  synopsis: string
  register: string
  beats: string[]
  cast: CastMember[]
  terms: Term[]
}

/**
 * One line on its way over: what it says, and the two things the model cannot
 * see for itself. Both optional; a line is never refused for running over.
 */
export type Untranslated = { text: string; kind?: Kind; budget?: number }

/** What a page is translated against. `source` is a word, not a code. */
export type Against = {
  system?: string | null
  source?: string | null
  glossary?: Term[] | null
  previously?: Story | null
  /** Sent whole: the API windows the beats around the page itself. */
  chapter?: Bible | null
  /** Which page of the chapter this is, counting from zero, for those beats. */
  page?: number | null
}

export async function translate(
  lines: Untranslated[],
  model: string,
  target: string,
  against: Against = {},
): Promise<{ texts: string[]; terms: Term[]; story: Story }> {
  if (lines.length === 0) {
    return { texts: [], terms: [], story: { scene: '', cast: [] } }
  }
  const { system, source, glossary, previously, chapter, page } = against

  const body = new FormData()
  body.append('texts', JSON.stringify(lines.map((line) => line.text)))
  body.append('model', model)
  body.append('target', target)
  if (system) body.append('system', system)
  if (source) body.append('source', source)
  if (glossary && glossary.length > 0) {
    body.append('glossary', JSON.stringify(glossary))
  }
  // Whole: the cast is what the model is asked to correct, so it sees all of it.
  if (previously && (previously.scene !== '' || previously.cast.length > 0)) {
    body.append('previously', JSON.stringify(previously))
  }
  // Whole, and the page with it, or every beat lands in front of the wrong one.
  if (chapter) {
    body.append('chapter', JSON.stringify(chapter))
    body.append('page', String(page ?? 0))
  }
  // One per line or not at all — the API refuses anything else, since a marker
  // a place out describes the wrong line.
  if (lines.some((line) => line.kind)) {
    body.append('kinds', JSON.stringify(lines.map((line) => line.kind ?? '')))
  }
  if (lines.some((line) => line.budget)) {
    body.append('budgets', JSON.stringify(lines.map((line) => line.budget ?? 0)))
  }

  const response = await reach('/api/translate', { method: 'POST', body })
  const answer = (await response.json()) as {
    texts: string[]
    terms?: Term[]
    story?: Partial<Story>
  }
  return {
    texts: answer.texts,
    terms: answer.terms ?? [],
    story: {
      scene: answer.story?.scene ?? '',
      cast: answer.story?.cast ?? [],
    },
  }
}

/**
 * One window of a chapter's lettering, read before any of it is translated. A
 * page with nothing on it is sent as an empty list rather than left out.
 *
 * No `system` on purpose: the prompt this dashboard holds is a *translation*
 * prompt, and a survey briefed to translate translates its window instead of
 * reading it.
 */
export async function survey(
  pages: string[][],
  model: string,
  target: string,
  against: { source?: string | null; chapter?: Bible | null; first?: number } = {},
): Promise<Bible> {
  const body = new FormData()
  body.append('pages', JSON.stringify(pages))
  body.append('model', model)
  body.append('target', target)
  if (against.source) body.append('source', against.source)
  if (against.chapter) body.append('chapter', JSON.stringify(against.chapter))
  if (against.first) body.append('first', String(against.first))

  const response = await reach('/api/survey', { method: 'POST', body })
  const answer = (await response.json()) as { chapter?: Partial<Bible> }
  return {
    synopsis: answer.chapter?.synopsis ?? '',
    register: answer.chapter?.register ?? '',
    beats: answer.chapter?.beats ?? [],
    cast: answer.chapter?.cast ?? [],
    terms: answer.chapter?.terms ?? [],
  }
}

/**
 * The lettering itself, pixel by pixel: a page-sized PNG, white on clear.
 * `grow` is in the detector's pixels, not the page's.
 */
export async function letterMask(file: File, grow?: number): Promise<Blob> {
  const response = await send(
    '/api/letters',
    file,
    grow === undefined ? {} : { grow },
  )
  return response.blob()
}

/** The page with everything the mask marks taken out of it, as a PNG. */
export async function clean(file: File, mask: Blob, fill: Fill): Promise<Blob> {
  const response = await send('/api/clean', file, { mask, fill })
  return response.blob()
}
