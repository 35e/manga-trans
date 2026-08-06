/** The back end, which lives on its own port and lets this origin in. */
export const API_BASE: string =
  import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

/** A box is [x0, y0, x1, y1] in image pixels. */
export type Box = [number, number, number, number]

export type Region = {
  box: Box
  confidence: number
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
export type Stage = 'detecting' | 'reading' | 'cleaning'

/** Below this the detector is guessing; the README says look twice. */
export const UNSURE = 0.6

/**
 * One page up, one answer back. Anything that is not a Blob goes up as JSON,
 * which is how the API takes boxes; every error it raises comes back as
 * {"error": "..."}.
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

  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, { method: 'POST', body, signal })
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === 'AbortError') throw cause
    throw new Error(`Could not reach the API at ${API_BASE}`, { cause })
  }

  if (!response.ok) {
    const said = await response
      .json()
      .then((body: { error?: string }) => body.error)
      .catch(() => undefined)
    throw new Error(said ?? `The API answered ${response.status}`)
  }

  return response
}

/** Every block of lettering the detector finds on one page. */
export async function detect(
  file: File,
  signal?: AbortSignal,
): Promise<Detection> {
  const response = await send('/api/detect', file, {}, signal)
  return response.json() as Promise<Detection>
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
