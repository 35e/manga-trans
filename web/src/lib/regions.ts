
import type { Analysis, Box, Region } from './api'
import { insertAt, moveAt, movedIndex } from './order'

function texts(analysis: Analysis): string[] {
  return analysis.texts ?? analysis.detection.regions.map(() => '')
}

function withRegions(analysis: Analysis, regions: Region[]): Analysis {
  return { ...analysis, detection: { ...analysis.detection, regions } }
}

export function toClean(analysis: Analysis): Box[] {
  const skip = new Set(analysis.excluded)
  return analysis.detection.regions
    .filter((_, index) => !skip.has(index))
    .map((region) => region.box)
}

export function inserted(analysis: Analysis, at: number, region: Region): Analysis {
  return {
    ...analysis,
    detection: {
      ...analysis.detection,
      regions: insertAt(analysis.detection.regions, at, region),
    },
    texts: insertAt(texts(analysis), at, ''),
    excluded: analysis.excluded.map((index) => (index >= at ? index + 1 : index)),
  }
}

export function moved(analysis: Analysis, from: number, to: number): Analysis {
  const { regions } = analysis.detection
  if (!regions[from] || !regions[to]) return analysis
  return {
    ...analysis,
    detection: { ...analysis.detection, regions: moveAt(regions, from, to) },
    texts: analysis.texts ? moveAt(analysis.texts, from, to) : null,
    excluded: analysis.excluded.map((index) => movedIndex(index, from, to)),
  }
}

export function withBox(analysis: Analysis, index: number, box: Box): Analysis {
  const region = analysis.detection.regions[index]
  if (!region) return analysis
  const regions = [...analysis.detection.regions]
  regions[index] = { ...region, box, bubble: null }
  return withRegions(analysis, regions)
}

export function toggledExcluded(analysis: Analysis, index: number): Analysis {
  const excluded = new Set(analysis.excluded)
  if (!excluded.delete(index)) excluded.add(index)
  return { ...analysis, excluded: [...excluded] }
}

export function split(
  analysis: Analysis,
  index: number,
  firstBox: Box,
  added: Region,
): Analysis {
  const target = analysis.detection.regions[index]
  if (!target) return analysis

  const regions = [...analysis.detection.regions]
  regions[index] = { ...target, box: firstBox, bubble: null }

  return {
    ...analysis,
    detection: {
      ...analysis.detection,
      regions: insertAt(regions, index + 1, added),
    },
    texts: insertAt(texts(analysis), index + 1, ''),
    excluded: [
      ...analysis.excluded.map((was) => (was >= index + 1 ? was + 1 : was)),
      ...(analysis.excluded.includes(index) ? [index + 1] : []),
    ],
  }
}

export function withReading(
  analysis: Analysis,
  id: string,
  text: string | undefined,
): Analysis {
  if (!analysis.texts) return analysis
  const where = analysis.detection.regions.findIndex((region) => region.id === id)
  if (where === -1) return analysis

  const said = [...analysis.texts]
  said[where] = text ?? ''
  return { ...analysis, texts: said }
}

export function withRooms(
  analysis: Analysis,
  ids: string[],
  rooms: (Box | null)[],
): Analysis {
  const where = new Map(ids.map((id, at) => [id, at]))
  let touched = false
  const regions = analysis.detection.regions.map((region) => {
    const at = where.get(region.id)
    if (at === undefined || at >= rooms.length) return region
    touched = true
    return { ...region, bubble: rooms[at] }
  })
  return touched ? withRegions(analysis, regions) : analysis
}
