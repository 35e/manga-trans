import { strFromU8, Unzip, UnzipInflate, Zip, ZipPassThrough } from 'fflate'

const TYPES: Record<string, string> = {
  png: 'image/png',
  jpg: 'image/jpeg',
  jpeg: 'image/jpeg',
  gif: 'image/gif',
  webp: 'image/webp',
  bmp: 'image/bmp',
  avif: 'image/avif',
  tif: 'image/tiff',
  tiff: 'image/tiff',
}

const MAX_ARCHIVE = 256 * 1024 * 1024
const MAX_EXPANDED = 512 * 1024 * 1024
const MAX_IMAGE = 32 * 1024 * 1024
const MAX_ENTRIES = 2000
// DEFLATE can expand roughly 1,032:1; small pushes bound transient inflate buffers.
const CHUNK = 4096

export function isZip(file: File): boolean {
  return (
    file.type === 'application/zip' ||
    file.type === 'application/x-zip-compressed' ||
    /\.(zip|cbz)$/i.test(file.name)
  )
}

function typeOf(name: string): string | undefined {
  const dot = name.lastIndexOf('.')
  return dot === -1 ? undefined : TYPES[name.slice(dot + 1).toLowerCase()]
}

function worthKeeping(path: string): boolean {
  const name = path.slice(path.lastIndexOf('/') + 1)
  return (
    !path.endsWith('/') &&
    !path.startsWith('__MACOSX/') &&
    !path.includes('/__MACOSX/') &&
    !name.startsWith('.') &&
    typeOf(name) !== undefined
  )
}

const order = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' })

export type Packed = { name: string; bytes: Uint8Array }

export async function pack(files: Iterable<Packed> | AsyncIterable<Packed>): Promise<Blob> {
  const chunks: Blob[] = []
  const names = new Set<string>()
  const zip = new Zip((error, chunk) => {
    if (error) throw error
    // Snapshot each output now: neither the ZIP nor this array retains input pages.
    chunks.push(new Blob([chunk as BlobPart]))
  })
  try {
    for await (const file of files) {
      let name = file.name
      for (let again = 2; names.has(name); again++) {
        const dot = file.name.lastIndexOf('.')
        const stem = dot === -1 ? file.name : file.name.slice(0, dot)
        const rest = dot === -1 ? '' : file.name.slice(dot)
        name = `${stem} (${again})${rest}`
      }
      names.add(name)
      const entry = new ZipPassThrough(name)
      zip.add(entry)
      entry.push(file.bytes, true)
    }
    zip.end()
    return new Blob(chunks, { type: 'application/zip' })
  } catch (error) {
    throw new Error(`the chapter could not be packed: ${error instanceof Error ? error.message : error}`)
  } finally {
    zip.terminate()
  }
}

function uint64(view: DataView, offset: number): number {
  const value = view.getBigUint64(offset, true)
  if (value > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error('ZIP64 value exceeds safe integer limit')
  return Number(value)
}

// Unzip's streaming parser does not validate the central directory or require an
// end record. Preflight bounded headers so truncated archives never import partly.
async function directory(file: File, signal?: AbortSignal) {
  const read = async (offset: number, size: number) => {
    signal?.throwIfAborted()
    if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(size) ||
      offset < 0 || size < 0 || offset + size > file.size) throw new Error('invalid ZIP header bounds')
    const buffer = await file.slice(offset, offset + size).arrayBuffer()
    signal?.throwIfAborted()
    if (buffer.byteLength !== size) throw new Error('truncated ZIP headers')
    return new DataView(buffer)
  }
  if (file.size > MAX_ARCHIVE) throw new Error('archive exceeds 256 MiB compressed limit')
  if (file.size < 22) throw new Error('invalid ZIP end record')
  const tailStart = Math.max(0, file.size - 65557)
  const tail = await read(tailStart, file.size - tailStart)
  let end = tail.byteLength - 22
  while (end >= 0 && (tail.getUint32(end, true) !== 0x06054b50 ||
    end + 22 + tail.getUint16(end + 20, true) !== tail.byteLength)) end--
  if (end < 0) throw new Error('missing ZIP end record')
  if (tail.getUint16(end + 4, true) || tail.getUint16(end + 6, true)) {
    throw new Error('multi-volume ZIP archives are not supported')
  }
  let count = tail.getUint16(end + 10, true)
  let size = tail.getUint32(end + 12, true)
  let start = tail.getUint32(end + 16, true)
  let directoryEnd = tailStart + end
  const diskCount = tail.getUint16(end + 8, true)
  const locator = directoryEnd >= 20 ? await read(directoryEnd - 20, 20) : undefined
  if (locator?.getUint32(0, true) === 0x07064b50) {
    if (locator.getUint32(4, true) || locator.getUint32(16, true) !== 1) {
      throw new Error('multi-volume ZIP archives are not supported')
    }
    const recordOffset = uint64(locator, 8)
    const record = await read(recordOffset, 56)
    if (record.getUint32(0, true) !== 0x06064b50 ||
      uint64(record, 4) < 44 || recordOffset + 12 + uint64(record, 4) !== directoryEnd - 20) {
      throw new Error('invalid ZIP64 end record')
    }
    if (record.getUint32(16, true) || record.getUint32(20, true)) {
      throw new Error('multi-volume ZIP archives are not supported')
    }
    const total = uint64(record, 32)
    const bytes = uint64(record, 40)
    const offset = uint64(record, 48)
    if (uint64(record, 24) !== total ||
      (count !== 0xffff && count !== total) || (diskCount !== 0xffff && diskCount !== total) ||
      (size !== 0xffffffff && size !== bytes) || (start !== 0xffffffff && start !== offset)) {
      throw new Error('inconsistent ZIP64 directory')
    }
    count = total
    size = bytes
    start = offset
    directoryEnd = recordOffset
  } else if (count === 0xffff || diskCount === 0xffff || size === 0xffffffff || start === 0xffffffff) {
    throw new Error('missing ZIP64 end record')
  } else if (diskCount !== count) {
    throw new Error('invalid ZIP directory entry count')
  }
  if (count > MAX_ENTRIES) throw new Error('archive exceeds 2000 entry limit')
  if (start + size !== directoryEnd) throw new Error('invalid ZIP directory')
  const entries: { name: string; size: number; compressed: number; offset: number; method: number }[] = []
  let at = start
  let expanded = 0
  for (let i = 0; i < count; i++) {
    if (at + 46 > start + size) throw new Error('truncated ZIP directory')
    const header = await read(at, 46)
    if (header.getUint32(0, true) !== 0x02014b50) throw new Error('invalid ZIP directory entry')
    const flags = header.getUint16(8, true)
    const method = header.getUint16(10, true)
    let compressed = header.getUint32(20, true)
    let original = header.getUint32(24, true)
    const length = header.getUint16(28, true)
    let offset = header.getUint32(42, true)
    let disk = header.getUint16(34, true)
    const extraLength = header.getUint16(30, true)
    const next = at + 46 + length + extraLength + header.getUint16(32, true)
    if (flags & 1) throw new Error('encrypted ZIP archives are not supported')
    if (method !== 0 && method !== 8) throw new Error(`unsupported ZIP compression method ${method}`)
    if (next > start + size) throw new Error('invalid ZIP entry bounds')
    if (original === 0xffffffff || compressed === 0xffffffff || offset === 0xffffffff || disk === 0xffff) {
      const extra = await read(at + 46 + length, extraLength)
      let found = false
      for (let field = 0; field + 4 <= extra.byteLength;) {
        const id = extra.getUint16(field, true)
        const end = field + 4 + extra.getUint16(field + 2, true)
        if (end > extra.byteLength) throw new Error('truncated ZIP extra field')
        if (id === 1) {
          let value = field + 4
          const take = () => {
            if (value + 8 > end) throw new Error('truncated ZIP64 size')
            const size = uint64(extra, value)
            value += 8
            return size
          }
          if (original === 0xffffffff) original = take()
          if (compressed === 0xffffffff) compressed = take()
          if (offset === 0xffffffff) offset = take()
          if (disk === 0xffff) {
            if (value + 4 > end) throw new Error('truncated ZIP64 disk number')
            disk = extra.getUint32(value, true)
          }
          found = true
          break
        }
        field = end
      }
      if (!found) throw new Error('missing ZIP64 extra field')
    }
    if (disk || offset + 30 + compressed > start) {
      throw new Error('invalid ZIP entry bounds')
    }
    const name = strFromU8(new Uint8Array((await read(at + 46, length)).buffer), !(flags & 2048))
    if (typeOf(name) && original > MAX_IMAGE) throw new Error(`${name} exceeds 32 MiB image limit`)
    expanded += original
    if (expanded > MAX_EXPANDED) throw new Error('archive exceeds 512 MiB expanded limit')
    entries.push({ name, size: original, compressed, offset, method })
    at = next
  }
  if (at !== start + size) throw new Error('invalid ZIP directory size')
  entries.sort((one, other) => one.offset - other.offset)
  return { entries, start }
}

export async function expand(file: File, signal?: AbortSignal): Promise<File[]> {
  let reader: ReadableStreamDefaultReader<Uint8Array> | undefined
  let active: { terminate(): void } | undefined
  const cancel = () => { void reader?.cancel().catch(() => {}) }
  try {
    const { entries, start } = await directory(file, signal)
    signal?.throwIfAborted()
    const pages: File[] = []
    let count = 0
    let total = 0
    let completed = 0
    const unzip = new Unzip((entry) => {
      if (++count > MAX_ENTRIES) throw new Error('archive exceeds 2000 entry limit')
      const declared = entries[count - 1]
      if (!declared || entry.name !== declared.name || entry.compression !== declared.method ||
        (entry.size !== undefined && entry.size !== declared.compressed) ||
        (entry.originalSize !== undefined && entry.originalSize !== declared.size)) {
        throw new Error('ZIP local headers do not match its directory')
      }
      const keep = worthKeeping(entry.name)
      const chunks: Blob[] = []
      let size = 0
      active = entry
      entry.ondata = (error, bytes, final) => {
        if (error) throw error
        size += bytes.byteLength
        total += bytes.byteLength
        if (typeOf(entry.name) && size > MAX_IMAGE) throw new Error(`${entry.name} exceeds 32 MiB image limit`)
        if (total > MAX_EXPANDED) throw new Error('archive exceeds 512 MiB expanded limit')
        if (size > declared.size) throw new Error(`${entry.name} expands beyond its declared size`)
        if (keep && bytes.byteLength) chunks.push(new Blob([bytes as BlobPart]))
        if (final) {
          if (size !== declared.size) throw new Error(`${entry.name} has an invalid expanded size`)
          completed++
          if (keep) pages.push(new File(chunks, entry.name, {
            type: typeOf(entry.name), lastModified: file.lastModified,
          }))
          active = undefined
        }
      }
      // Inflate even ignored entries: their actual bytes count toward the bomb limit.
      entry.start()
    })
    unzip.register(UnzipInflate)
    reader = file.stream().getReader()
    signal?.addEventListener('abort', cancel, { once: true })
    let compressed = 0
    let yieldedAt = performance.now()
    while (true) {
      signal?.throwIfAborted()
      const { done, value } = await reader.read()
      signal?.throwIfAborted()
      if (done) break
      const offset = compressed
      compressed += value.byteLength
      if (compressed > MAX_ARCHIVE) throw new Error('archive exceeds 256 MiB compressed limit')
      // Do not feed directory bytes to Unzip: it scans them for local signatures.
      const length = Math.min(value.byteLength, Math.max(0, start - offset))
      for (let at = 0; at < length; at += CHUNK) {
        signal?.throwIfAborted()
        unzip.push(value.subarray(at, Math.min(at + CHUNK, length)))
        if (performance.now() - yieldedAt >= 8) {
          await new Promise<void>((resolve) => setTimeout(resolve, 0))
          signal?.throwIfAborted()
          yieldedAt = performance.now()
        }
      }
    }
    if (compressed !== file.size) throw new Error('truncated ZIP input')
    unzip.push(new Uint8Array(0), true)
    if (count !== entries.length || completed !== count) throw new Error('incomplete ZIP entries')
    return pages.sort((one, other) => order.compare(one.name, other.name))
  } catch (error) {
    if (signal?.aborted) throw signal.reason
    throw new Error(`${file.name} could not be opened: ${error instanceof Error ? error.message : error}`)
  } finally {
    signal?.removeEventListener('abort', cancel)
    active?.terminate()
    if (reader) {
      try { await reader.cancel() } catch { /* Preserve the original read/inflate failure. */ }
      reader.releaseLock()
    }
  }
}
