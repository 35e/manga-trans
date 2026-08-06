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
}

/** Detecting comes first, then reading what was detected. */
export type Stage = 'detecting' | 'reading'

/** Below this the detector is guessing; the README says look twice. */
export const UNSURE = 0.6

/** One page up, JSON back. Every error the API raises comes back as {"error"}. */
async function post<T>(
  path: string,
  file: File,
  fields: Record<string, unknown>,
  signal?: AbortSignal,
): Promise<T> {
  const body = new FormData()
  body.append('image', file, file.name)
  for (const [name, value] of Object.entries(fields)) {
    body.append(name, JSON.stringify(value))
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

  return response.json() as Promise<T>
}

/** Every block of lettering the detector finds on one page. */
export function detect(file: File, signal?: AbortSignal): Promise<Detection> {
  return post<Detection>('/api/detect', file, {}, signal)
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
  const { texts } = await post<{ texts: string[] }>(
    '/api/read',
    file,
    { boxes },
    signal,
  )
  return texts
}
