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
  | 'surveying'

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

export type Term = { source: string; target: string; note?: string }

export type Gender = 'male' | 'female' | 'unknown'

export type Fact = 'gender' | 'note'

export type CastMember = {
  name: string
  gender: Gender
  note?: string
  settled?: Fact[]
}

export type Story = { scene: string; cast: CastMember[] }

export type Bible = {
  synopsis: string
  register: string
  beats: string[]
  cast: CastMember[]
  terms: Term[]
}

export type Untranslated = { text: string; kind?: Kind; budget?: number }

export type Against = {
  system?: string | null
  source?: string | null
  glossary?: Term[] | null
  previously?: Story | null
  chapter?: Bible | null
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
  if (previously && (previously.scene !== '' || previously.cast.length > 0)) {
    body.append('previously', JSON.stringify(previously))
  }
  if (chapter) {
    body.append('chapter', JSON.stringify(chapter))
    body.append('page', String(page ?? 0))
  }
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
