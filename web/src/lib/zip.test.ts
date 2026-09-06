import { Zip, ZipDeflate, zipSync } from 'fflate'
import { finished } from './chapter'
import { fingerprint, loadImage } from './images'
import { expand, pack } from './zip'

// Browser console has no static imports: await (await import('/src/lib/zip.test.ts')).checkArchiveIdentity()
export async function checkArchiveIdentity() {
  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = 1
  const bytes = new Uint8Array(await (await fetch(canvas.toDataURL())).arrayBuffer())
  const archive = new File([await pack([
    { name: 'a/001.png', bytes },
    { name: 'b/001.png', bytes },
  ])], 'chapter.cbz', { lastModified: 1234 })
  const pages = await expand(archive)
  const identities = pages.map((page) => fingerprint(page, 'chapter'))
  if (pages.length !== 2 || new Set(identities).size !== 2) {
    throw new Error('Equal-sized pages in different archive directories must survive deduplication')
  }
  const repeated = await expand(archive)
  if (repeated.length !== 2 || repeated.some((page, at) => fingerprint(page, 'chapter') !== identities[at])) {
    throw new Error('Importing the same archive again must retain its duplicate identities')
  }

  const images = await Promise.all(pages.map((page) => loadImage(page, 'chapter')))
  try {
    if (images.some((image) => image === null || image.name !== '001.png')) {
      throw new Error('Archive pages must decode and display their basename')
    }
    const packed = await pack(await Promise.all(images.map((image) => finished(image!, undefined, null))))
    const exported = await expand(new File([packed], 'export.cbz'))
    if (exported.length !== 2 || new Set(exported.map((page) => page.name)).size !== 2) {
      throw new Error('Export must retain both pages despite matching display names')
    }
    for (const page of exported) {
      const held = new Uint8Array(await page.arrayBuffer())
      if (held.length !== bytes.length || held.some((byte, at) => byte !== bytes[at])) {
        throw new Error('Export must preserve the original page bytes')
      }
    }
    return { imported: pages.length, exported: exported.map((page) => page.name) }
  } finally {
    for (const image of images) if (image) URL.revokeObjectURL(image.url)
  }
}

async function rejects(work: () => Promise<unknown>, message: string) {
  try {
    await work()
  } catch (error) {
    if (error instanceof Error && error.message.includes(message)) return
    throw error
  }
  throw new Error(`Expected rejection containing: ${message}`)
}

// Keep adversarial fixtures compressed; never allocate a whole expanded bomb.
function bomb(name: string, size: number): Blob {
  const chunks: Blob[] = []
  const zip = new Zip((error, bytes) => {
    if (error) throw error
    chunks.push(new Blob([bytes as BlobPart]))
  })
  const entry = new ZipDeflate(name, { level: 9 })
  zip.add(entry)
  const bytes = new Uint8Array(1024 * 1024)
  for (let sent = 0; sent < size; sent += bytes.length) {
    entry.push(bytes.subarray(0, Math.min(bytes.length, size - sent)), sent + bytes.length >= size)
  }
  zip.end()
  return new Blob(chunks)
}

// Browser: await (await import('/src/lib/zip.test.ts')).checkArchiveLimits()
export async function checkArchiveLimits() {
  const mib = 1024 * 1024
  const small = zipSync({ 'page.png': new Uint8Array([1, 2, 3]) }, { level: 0 })
  const declared = small.slice()
  const view = new DataView(declared.buffer)
  const central = view.getUint32(declared.length - 6, true)
  view.setUint32(central + 24, 32 * mib + 1, true)
  await rejects(() => expand(new File([declared], 'oversize.cbz')), '32 MiB')

  const many = await pack(Array.from({ length: 2001 }, (_, at) => ({
    name: `ignored/${at}.txt`, bytes: new Uint8Array(0),
  })))
  await rejects(() => expand(new File([many], 'entries.zip')), '2000 entry')

  const block = new Blob([new Uint8Array(mib)])
  await rejects(() => expand(new File(new Array(257).fill(block), 'compressed.zip')), '256 MiB')

  const totals = zipSync(Object.fromEntries(Array.from({ length: 17 }, (_, at) => [
    `${at}.txt`, new Uint8Array(0),
  ])), { level: 0 })
  const totalsView = new DataView(totals.buffer)
  let at = totalsView.getUint32(totals.length - 6, true)
  for (let i = 0; i < 17; i++) {
    totalsView.setUint32(at + 24, 32 * mib, true)
    at += 46 + totalsView.getUint16(at + 28, true)
  }
  await rejects(() => expand(new File([totals as BlobPart], 'total.zip')), '512 MiB')

  // Forge a legal central size on a descriptor-based entry. The actual inflate
  // counter, not the declared header, must reject before retaining byte 32 MiB+1.
  const inflated = new Uint8Array(await bomb('page.png', 32 * mib + 1).arrayBuffer())
  const inflatedView = new DataView(inflated.buffer)
  const inflatedCentral = inflatedView.getUint32(inflated.length - 6, true)
  inflatedView.setUint32(inflatedCentral + 24, 32 * mib, true)
  await rejects(() => expand(new File([inflated], 'bomb.zip')), '32 MiB')

  const lying = small.slice()
  const lyingView = new DataView(lying.buffer)
  lyingView.setUint32(22, 2, true)
  lyingView.setUint32(central + 24, 2, true)
  await rejects(() => expand(new File([lying], 'lying.zip')), 'declared size')
  await rejects(() => expand(new File([small.slice(0, -22)], 'truncated.zip')), 'end record')

  // The exact per-image boundary still works, including streaming descriptors.
  const boundary = await expand(new File([bomb('page.png', 32 * mib)], 'boundary.cbz'))
  if (boundary.length !== 1 || boundary[0].size !== 32 * mib) throw new Error('Exact image byte limit must be accepted')
  return { declared: true, actual: true, entries: true, compressed: true, total: true, boundary: true }
}

// Separate heavier check: ignored files must not bypass the actual total limit.
export async function checkArchiveExpandedLimit() {
  const limit = 512 * 1024 * 1024
  const bytes = new Uint8Array(await bomb('ignored.txt', limit + 1).arrayBuffer())
  const view = new DataView(bytes.buffer)
  view.setUint32(view.getUint32(bytes.length - 6, true) + 24, limit, true)
  await rejects(() => expand(new File([bytes], 'expanded-bomb.zip')), '512 MiB')
  return { ignoredEntryActualLimit: true }
}

// Browser: await (await import('/src/lib/zip.test.ts')).checkArchiveStreaming()
export async function checkArchiveStreaming() {
  const reused = new Uint8Array([1, 2, 3])
  async function* pages() {
    yield { name: 'page.png', bytes: reused }
    reused.fill(4)
    yield { name: 'page.png', bytes: reused }
    reused.fill(9)
  }
  const packed = await pack(pages())
  const bytes = new Uint8Array(await packed.arrayBuffer())
  const source = new File([packed], 'stream.cbz')
  let position = 0
  const input = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (position === bytes.length) controller.close()
      else {
        const end = Math.min(position + 7, bytes.length)
        controller.enqueue(bytes.subarray(position, end))
        position = end
      }
    },
  })
  Object.defineProperty(source, 'stream', { value: () => input })
  const unpacked = await expand(source)
  const original = unpacked.find((page) => page.name === 'page.png')
  const duplicate = unpacked.find((page) => page.name === 'page (2).png')
  if (!original || !duplicate ||
    new Uint8Array(await original.arrayBuffer()).join() !== '1,2,3' ||
    new Uint8Array(await duplicate.arrayBuffer()).join() !== '4,4,4' || input.locked) {
    throw new Error('Streaming pack must snapshot each page before requesting the next and retain duplicate names')
  }

  let closed = false
  async function* failing() {
    try {
      yield { name: 'page.png', bytes: reused }
      throw new Error('generator failed')
    } finally { closed = true }
  }
  await rejects(() => pack(failing()), 'generator failed')
  if (!closed) throw new Error('Failed packing must close its input iterator')

  closed = false
  async function* invalid() {
    try {
      yield { name: 'x'.repeat(65536), bytes: reused }
      throw new Error('Input must not advance after ZIP error')
    } finally { closed = true }
  }
  await rejects(() => pack(invalid()), 'filename too long')
  if (!closed) throw new Error('ZIP errors must close their input iterator')

  const broken = new ReadableStream<Uint8Array>({
    start(controller) { controller.error(new Error('read failed')) },
  })
  const unreadable = new File([packed], 'unreadable.cbz')
  Object.defineProperty(unreadable, 'stream', { value: () => broken })
  await rejects(() => expand(unreadable), 'read failed')
  if (broken.locked) throw new Error('A failed archive read must release its reader')

  const controller = new AbortController()
  let canceled = false
  const aborted = new ReadableStream<Uint8Array>({
    pull(stream) {
      stream.enqueue(bytes.subarray(0, 7))
      controller.abort(new Error('import canceled'))
    },
    cancel() { canceled = true },
  }, { highWaterMark: 0 })
  const cancelable = new File([packed], 'cancel.cbz')
  Object.defineProperty(cancelable, 'stream', { value: () => aborted })
  await rejects(() => expand(cancelable, controller.signal), 'import canceled')
  if (!canceled || aborted.locked) throw new Error('Aborted archive reads must cancel and unlock their stream')
  return { chunked: true, snapshots: true, generatorFailure: true, zipFailure: true, readFailure: true, cancellation: true }
}

export async function checkArchiveInflateCancellation() {
  const bytes = new Uint8Array(await bomb('ignored.txt', 64 * 1024 * 1024).arrayBuffer())
  const controller = new AbortController()
  let sent = false
  let canceled = false
  let timer: number | undefined
  const input = new ReadableStream<Uint8Array>({
    pull(stream) {
      if (sent) stream.close()
      else {
        sent = true
        stream.enqueue(bytes)
        // Integration check of real browser task yielding; fake timers cannot
        // prove an abort task can interrupt synchronous inflation.
        timer = window.setTimeout(() => controller.abort(new Error('inflate canceled')), 0)
      }
    },
    cancel() { canceled = true },
  }, { highWaterMark: 0 })
  const source = new File([bytes], 'cancel-inflate.cbz')
  Object.defineProperty(source, 'stream', { value: () => input })
  try {
    await rejects(() => expand(source, controller.signal), 'inflate canceled')
    if (!canceled || input.locked) throw new Error('Mid-inflate cancellation must cancel and unlock the reader')
  } finally { window.clearTimeout(timer) }
  return { midInflateCancellation: true }
}

function forceZip64(): Uint8Array<ArrayBuffer> {
  const source = zipSync({ 'nested/page.png': new Uint8Array([1, 2, 3]) }, { level: 0 })
  const view = new DataView(source.buffer)
  const oldCentral = view.getUint32(source.length - 6, true)
  const nameLength = view.getUint16(26, true)
  const headerLength = 30 + nameLength
  const centralLength = 46 + nameLength + 32
  const centralOffset = oldCentral + 20
  const recordOffset = centralOffset + centralLength
  const bytes = new Uint8Array(recordOffset + 56 + 20 + 22)
  const result = new DataView(bytes.buffer)
  bytes.set(source.subarray(0, headerLength))
  result.setUint16(4, 45, true)
  result.setUint32(18, 0xffffffff, true)
  result.setUint32(22, 0xffffffff, true)
  result.setUint16(28, 20, true)
  result.setUint16(headerLength, 1, true)
  result.setUint16(headerLength + 2, 16, true)
  result.setBigUint64(headerLength + 4, 3n, true)
  result.setBigUint64(headerLength + 12, 3n, true)
  bytes.set(source.subarray(headerLength, oldCentral), headerLength + 20)
  bytes.set(source.subarray(oldCentral, oldCentral + 46 + nameLength), centralOffset)
  result.setUint16(centralOffset + 6, 45, true)
  result.setUint32(centralOffset + 20, 0xffffffff, true)
  result.setUint32(centralOffset + 24, 0xffffffff, true)
  result.setUint16(centralOffset + 30, 32, true)
  result.setUint16(centralOffset + 34, 0xffff, true)
  result.setUint32(centralOffset + 42, 0xffffffff, true)
  const extra = centralOffset + 46 + nameLength
  result.setUint16(extra, 1, true)
  result.setUint16(extra + 2, 28, true)
  result.setBigUint64(extra + 4, 3n, true)
  result.setBigUint64(extra + 12, 3n, true)
  result.setBigUint64(extra + 20, 0n, true)
  result.setUint32(extra + 28, 0, true)
  result.setUint32(recordOffset, 0x06064b50, true)
  result.setBigUint64(recordOffset + 4, 44n, true)
  result.setUint16(recordOffset + 12, 45, true)
  result.setUint16(recordOffset + 14, 45, true)
  result.setBigUint64(recordOffset + 24, 1n, true)
  result.setBigUint64(recordOffset + 32, 1n, true)
  result.setBigUint64(recordOffset + 40, BigInt(centralLength), true)
  result.setBigUint64(recordOffset + 48, BigInt(centralOffset), true)
  result.setUint32(recordOffset + 56, 0x07064b50, true)
  result.setBigUint64(recordOffset + 64, BigInt(recordOffset), true)
  result.setUint32(recordOffset + 72, 1, true)
  const end = recordOffset + 76
  result.setUint32(end, 0x06054b50, true)
  result.setUint16(end + 8, 0xffff, true)
  result.setUint16(end + 10, 0xffff, true)
  result.setUint32(end + 12, 0xffffffff, true)
  result.setUint32(end + 16, 0xffffffff, true)
  return bytes
}

export async function checkArchiveZip64() {
  const bytes = forceZip64()
  const pages = await expand(new File([bytes], 'zip64.cbz'))
  if (pages.length !== 1 || pages[0].name !== 'nested/page.png' ||
    new Uint8Array(await pages[0].arrayBuffer()).join() !== '1,2,3') {
    throw new Error('Small ZIP64 archives must preserve paths and page bytes')
  }
  async function* output() {
    for (const page of pages) yield { name: page.name, bytes: new Uint8Array(await page.arrayBuffer()) }
  }
  const restored = await expand(new File([await pack(output())], 'roundtrip.cbz'))
  if (restored.length !== 1 || restored[0].name !== pages[0].name ||
    new Uint8Array(await restored[0].arrayBuffer()).join() !== '1,2,3') {
    throw new Error('ZIP64 pages must roundtrip through the streaming exporter')
  }
  const record = bytes.length - 98
  const view = new DataView(bytes.buffer)
  const central = Number(view.getBigUint64(record + 48, true))
  const sizeAt = central + 46 + view.getUint16(central + 28, true) + 4
  view.setBigUint64(sizeAt, 32n * 1024n * 1024n + 1n, true)
  await rejects(() => expand(new File([bytes], 'zip64-oversize.cbz')), '32 MiB')
  view.setBigUint64(sizeAt, BigInt(Number.MAX_SAFE_INTEGER) + 1n, true)
  await rejects(() => expand(new File([bytes], 'zip64-unsafe.cbz')), 'safe integer')
  view.setBigUint64(sizeAt, 3n, true)
  view.setBigUint64(record + 24, 2001n, true)
  view.setBigUint64(record + 32, 2001n, true)
  await rejects(() => expand(new File([bytes], 'zip64-entries.cbz')), '2000 entry')
  return { zip64Roundtrip: true, zip64Bounds: true }
}
