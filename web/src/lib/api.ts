export const API_BASE: string =
  import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export type Box = [number, number, number, number]

export type Language = { code: string; name: string; rtl: boolean }

export const LANGUAGE_DEFAULT = 'ja'

export type Kind = 'speech' | 'free'

export type Region = {
  id: string
  box: Box
  confidence: number
  kind?: Kind
  bubble?: Box | null
  manual?: boolean
}

export type Detection = {
  width: number
  height: number
  regions: Region[]
}

export type Analysis = {
  detection: Detection
  texts: string[] | null
  excluded: number[]
}

export type Stage =
  | 'detecting'
  | 'reading'
  | 'tracing'
  | 'cleaning'
  | 'translating'

export type Tool = 'boxes' | 'mask' | 'text'

export type Fill = 'art' | 'telea' | 'white'

export type Lettering = {
  text: string
  box: Box
  size: number
  angle: number
}

export const UNSURE = 0.8

export function said(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause)
}

async function refuse(response: Response): Promise<never> {
  const said = await response
    .json()
    .then((body: { error?: string }) => body.error)
    .catch(() => undefined)
  throw new Error(said ?? `The API answered ${response.status}`)
}

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

async function send(
  path: string,
  file: File,
  parts: Record<string, unknown> = {},
): Promise<Response> {
  const body = new FormData()
  body.append('image', file, file.name)
  for (const [name, value] of Object.entries(parts)) {
    if (value instanceof Blob) body.append(name, value, `${name}.png`)
    else if (typeof value === 'string') body.append(name, value)
    else body.append(name, JSON.stringify(value))
  }
  return reach(path, { method: 'POST', body })
}

export async function detect(file: File, language: string): Promise<Detection> {
  const response = await send('/api/detect', file, { language })
  const found = (await response.json()) as {
    width: number
    height: number
    regions: Omit<Region, 'id'>[]
  }
  return {
    ...found,
    regions: found.regions.map((region) => ({ ...region, id: crypto.randomUUID() })),
  }
}

export async function bubbles(file: File, boxes: Box[]): Promise<(Box | null)[]> {
  if (boxes.length === 0) return []
  const response = await send('/api/bubbles', file, { boxes })
  const answer = (await response.json()) as { regions: { bubble: Box | null }[] }
  return answer.regions.map((region) => region.bubble)
}

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

export async function languages(): Promise<Language[]> {
  const response = await reach('/api/languages')
  const answer = (await response.json()) as { languages: Language[] }
  return answer.languages
}

export async function models(): Promise<string[]> {
  const response = await reach('/api/models')
  const answer = (await response.json()) as { models: string[] }
  return answer.models
}

export async function defaultPrompt(): Promise<string> {
  const response = await reach('/api/prompt')
  const answer = (await response.json()) as { prompt: string }
  return answer.prompt
}

export type Untranslated = { text: string; kind?: Kind; budget?: number }

export type Against = {
  system?: string | null
  source?: string | null
}

export async function translate(
  lines: Untranslated[],
  model: string,
  target: string,
  against: Against = {},
): Promise<string[]> {
  if (lines.length === 0) return []
  const { system, source } = against

  const body = new FormData()
  body.append('texts', JSON.stringify(lines.map((line) => line.text)))
  body.append('model', model)
  body.append('target', target)
  if (system) body.append('system', system)
  if (source) body.append('source', source)
  if (lines.some((line) => line.kind)) {
    body.append('kinds', JSON.stringify(lines.map((line) => line.kind ?? '')))
  }
  if (lines.some((line) => line.budget)) {
    body.append('budgets', JSON.stringify(lines.map((line) => line.budget ?? 0)))
  }

  const response = await reach('/api/translate', { method: 'POST', body })
  const answer = (await response.json()) as { texts: string[] }
  return answer.texts
}

export async function letterMask(file: File, grow?: number): Promise<Blob> {
  const response = await send(
    '/api/letters',
    file,
    grow === undefined ? {} : { grow },
  )
  return response.blob()
}

export async function clean(file: File, mask: Blob, fill: Fill): Promise<Blob> {
  const response = await send('/api/clean', file, { mask, fill })
  return response.blob()
}
