/** The back end, which lives on its own port and lets this origin in. */
export const API_BASE: string =
  import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

/** A box is [x0, y0, x1, y1] in image pixels. */
export type Box = [number, number, number, number]

export type Region = {
  /**
   * This block, for as long as it exists. Blocks are resized, reordered and
   * added to by hand, so a place in the list is not a name: anything that goes
   * away and comes back — a reading, say — finds its block by this.
   */
  id: string
  box: Box
  confidence: number
  /** Drawn by hand, because the detector missed it. Never sure, just certain. */
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
   * Blocks to leave alone: their indices in `detection.regions`. A detector
   * that boxed something worth keeping — a sound effect, a signature, a stray
   * bit of art — is corrected here rather than by re-detecting.
   */
  excluded: number[]
}

/** Detecting comes first, then reading what was detected, then hiding it. */
export type Stage =
  | 'detecting'
  | 'reading'
  | 'tracing'
  | 'cleaning'
  | 'translating'

/** What the board is being used for: looking, hiding, or lettering. */
export type BoardMode = 'inspect' | 'mask' | 'translate'

/**
 * One translated line, set where the original was. The box starts as the block
 * the detector found and the size as whatever fits it, and both are the reader's
 * to change after that: a translation is rarely as long as what it replaces.
 */
export type Lettering = {
  text: string
  box: Box
  size: number
}

/** Below this the detector is guessing; the README says look twice. */
export const UNSURE = 0.6

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
  try {
    return await fetch(`${API_BASE}${path}`, init)
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === 'AbortError') throw cause
    throw new Error(`Could not reach the API at ${API_BASE}`, { cause })
  }
}

/**
 * One page up, one answer back. Anything that is not a Blob goes up as JSON,
 * which is how the API takes boxes.
 */
async function send(
  path: string,
  file: File,
  parts: Record<string, unknown>,
  signal?: AbortSignal,
): Promise<Response> {
  const body = new FormData()
  body.append('image', file, file.name)
  for (const [name, value] of Object.entries(parts)) {
    if (value instanceof Blob) body.append(name, value, `${name}.png`)
    else body.append(name, JSON.stringify(value))
  }

  const response = await reach(path, { method: 'POST', body, signal })
  if (!response.ok) await refuse(response)
  return response
}

/** Every block of lettering the detector finds on one page. */
export async function detect(
  file: File,
  signal?: AbortSignal,
): Promise<Detection> {
  const response = await send('/api/detect', file, {}, signal)
  const found = (await response.json()) as {
    width: number
    height: number
    regions: Omit<Region, 'id'>[]
  }
  // The API knows nothing of ids — it answers the same page the same way twice.
  // Blocks are named here, once, as they arrive.
  return {
    ...found,
    regions: found.regions.map((region) => ({
      ...region,
      id: crypto.randomUUID(),
    })),
  }
}

/**
 * What each box says, read by manga-ocr. One string per box, in the order the
 * boxes were given, so they line up with the regions from `detect`.
 */
export async function read(
  file: File,
  boxes: Box[],
  signal?: AbortSignal,
): Promise<string[]> {
  if (boxes.length === 0) return []
  const response = await send('/api/read', file, { boxes }, signal)
  const { texts } = (await response.json()) as { texts: string[] }
  return texts
}

/** Every model the API's Ollama has to translate with. */
export async function models(signal?: AbortSignal): Promise<string[]> {
  const response = await reach('/api/models', { signal })
  if (!response.ok) await refuse(response)
  const answer = (await response.json()) as { models: string[] }
  return answer.models
}

/**
 * What the API tells the model to do when it is not told anything else. Held
 * nowhere but here: the API keeps no settings, so a front end that wants its
 * own asks for this, lets it be edited, and sends the edit back each time.
 */
export async function defaultPrompt(signal?: AbortSignal): Promise<string> {
  const response = await reach('/api/prompt', { signal })
  if (!response.ok) await refuse(response)
  const answer = (await response.json()) as { prompt: string }
  return answer.prompt
}

/**
 * One translation per text, in the order they were given. The whole page goes
 * over at once: a line of manga read on its own often cannot be translated at
 * all, having no idea who is speaking or about what.
 *
 * `system` is what the model is told; leave it out for the API's own.
 */
export async function translate(
  texts: string[],
  model: string,
  target: string,
  system?: string | null,
  signal?: AbortSignal,
): Promise<string[]> {
  if (texts.length === 0) return []

  const body = new FormData()
  body.append('texts', JSON.stringify(texts))
  body.append('model', model)
  body.append('target', target)
  if (system) body.append('system', system)

  const response = await reach('/api/translate', { method: 'POST', body, signal })
  if (!response.ok) await refuse(response)
  const answer = (await response.json()) as { texts: string[] }
  return answer.texts
}

/**
 * The lettering itself, pixel by pixel: a page-sized PNG, opaque white on the
 * ink and clear everywhere else. `grow` is how many pixels to spread it by, so
 * nothing is left ringing a letter that has been hidden.
 */
export async function letterMask(
  file: File,
  grow?: number,
  signal?: AbortSignal,
): Promise<Blob> {
  const response = await send(
    '/api/letters',
    file,
    grow === undefined ? {} : { grow },
    signal,
  )
  return response.blob()
}

/**
 * The page with everything the mask marks painted out, as a PNG. The mask is a
 * page-sized image, white where the lettering should go.
 */
export async function clean(
  file: File,
  mask: Blob,
  signal?: AbortSignal,
): Promise<Blob> {
  const response = await send('/api/clean', file, { mask }, signal)
  return response.blob()
}
