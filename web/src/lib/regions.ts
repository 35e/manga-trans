/**
 * Editing the blocks on a page, as transforms on one `Analysis`.
 *
 * `detection.regions`, `texts` and `excluded` are indexed by the same block
 * position, so nothing may insert, move or split a block without carrying all
 * three — which is why every such move lives here. Anything asynchronous must
 * use {@link withReading} instead of an index: the list may have been reordered
 * while the request was in flight.
 */

import type { Analysis, Box, Region } from './api'
import { insertAt, moveAt, movedIndex } from './order'

/** `texts` if the page has been read, else one empty string per block. */
function texts(analysis: Analysis): string[] {
  return analysis.texts ?? analysis.detection.regions.map(() => '')
}

function withRegions(analysis: Analysis, regions: Region[]): Analysis {
  return { ...analysis, detection: { ...analysis.detection, regions } }
}

/** The boxes a clean should hide: every block that was not left alone. */
export function toClean(analysis: Analysis): Box[] {
  const skip = new Set(analysis.excluded)
  return analysis.detection.regions
    .filter((_, index) => !skip.has(index))
    .map((region) => region.box)
}

/** A block put in at `at`, with everything held by position moved along. */
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

/** A block dragged to a different place in the list. */
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

/**
 * A block's box, moved or resized by hand. The balloon goes with it: it was
 * worked out from where the block was, and a stale one letters the translation
 * into the balloon the block came from.
 */
export function withBox(analysis: Analysis, index: number, box: Box): Analysis {
  const region = analysis.detection.regions[index]
  if (!region) return analysis
  const regions = [...analysis.detection.regions]
  regions[index] = { ...region, box, bubble: null }
  return withRegions(analysis, regions)
}

/** Take one block out of what will be cleaned, or put it back. */
export function toggledExcluded(analysis: Analysis, index: number): Analysis {
  const excluded = new Set(analysis.excluded)
  if (!excluded.delete(index)) excluded.add(index)
  return { ...analysis, excluded: [...excluded] }
}

/**
 * One block that turned out to be two, cut in two. The first half keeps the
 * original's place and text until it has been read again; `added` goes in after
 * it. Two bubbles means two balloons, so whichever one the block as a whole was
 * said to be in was wrong for at least one half.
 */
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
    // A block left alone is left alone on both sides of the cut.
    excluded: [
      ...analysis.excluded.map((was) => (was >= index + 1 ? was + 1 : was)),
      ...(analysis.excluded.includes(index) ? [index + 1] : []),
    ],
  }
}

/**
 * What one block says, found again by name because the list may have been added
 * to or reordered while the reader was working.
 */
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

/**
 * The room each of those blocks is in, by name rather than by position.
 *
 * Blocks that go missing while the answer is in the air are skipped, and blocks
 * that were not asked about are left alone. Used where the list may have moved
 * under the request — {@link withBubbles} is the same thing for an answer that
 * is known to still line up.
 */
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

/**
 * A balloon for every block, from one `/api/bubbles` call. Both lists are held
 * by position, so the answer only means anything while the page still has the
 * blocks it was asked about.
 */
export function withBubbles(
  analysis: Analysis,
  balloons: (Box | null)[],
): Analysis | null {
  if (analysis.detection.regions.length !== balloons.length) return null
  return withRegions(
    analysis,
    analysis.detection.regions.map((region, at) => ({
      ...region,
      bubble: balloons[at],
    })),
  )
}
