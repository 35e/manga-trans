import { unzip, zip } from 'fflate'

/**
 * Pulling the pages out of an archive, and putting them back into one.
 *
 * A chapter arrives as a zip far more often than as fifty loose files, and the
 * order the pages are named in is the order they are read in — so they come out
 * sorted the way a person would sort them: page 2 before page 10. It leaves the
 * same way: a finished chapter is an archive again, not fifty saved files.
 */

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

export function isZip(file: File): boolean {
  return (
    file.type === 'application/zip' ||
    file.type === 'application/x-zip-compressed' ||
    /\.(zip|cbz)$/i.test(file.name)
  )
}

/** The image type a name implies, or nothing if it does not imply one. */
function typeOf(name: string): string | undefined {
  const dot = name.lastIndexOf('.')
  return dot === -1 ? undefined : TYPES[name.slice(dot + 1).toLowerCase()]
}

/** Junk an archive carries that was never a page: folders, metadata, dotfiles. */
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

/**
 * Every image inside a zip, in the order their names put them. Rejects only when
 * the archive itself cannot be read: one holding no images comes back empty,
 * which is a different thing and is reported differently.
 */
/** One file on its way into an archive. */
export type Packed = { name: string; bytes: Uint8Array }

/**
 * Those files as one archive.
 *
 * Stored rather than deflated. Every one of these is a PNG or a JPEG and so is
 * compressed already: deflating it a second time takes seconds over a chapter
 * and gives back a percent or so, which is the wrong way round for something a
 * person is waiting on. Asynchronous for the same reason — fflate does the work
 * off the main thread, and the dashboard stays usable while a chapter packs.
 */
export function pack(files: Packed[]): Promise<Blob> {
  const held: Record<string, [Uint8Array, { level: 0 }]> = {}
  for (const file of files) {
    // An archive is keyed by name, so two pages called the same thing would
    // leave one page in it. Two can be: pages are told apart by name *and* size
    // *and* date, so one chapter holding `001.png` twice at different sizes is
    // two pages. Numbering the second beats losing it silently.
    let name = file.name
    for (let again = 2; name in held; again++) {
      const dot = file.name.lastIndexOf('.')
      const stem = dot === -1 ? file.name : file.name.slice(0, dot)
      const rest = dot === -1 ? '' : file.name.slice(dot)
      name = `${stem} (${again})${rest}`
    }
    held[name] = [file.bytes, { level: 0 }]
  }

  return new Promise((resolve, reject) => {
    zip(held, { level: 0 }, (error, packed) => {
      if (error) reject(new Error(`the chapter could not be packed: ${error.message}`))
      else resolve(new Blob([packed as BlobPart], { type: 'application/zip' }))
    })
  })
}

export function expand(file: File): Promise<File[]> {
  return new Promise((resolve, reject) => {
    file
      .arrayBuffer()
      .then((buffer) => {
        unzip(
          new Uint8Array(buffer),
          // Unpacking every page of a chapter to keep three would be wasteful,
          // so the filter runs before anything is decompressed.
          { filter: (entry) => worthKeeping(entry.name) },
          (error, unpacked) => {
            if (error) {
              reject(new Error(`${file.name} could not be opened: ${error.message}`))
              return
            }
            resolve(
              Object.entries(unpacked)
                .sort(([one], [other]) => order.compare(one, other))
                .map(([path, bytes]) => {
                  const name = path.slice(path.lastIndexOf('/') + 1)
                  return new File([bytes as BlobPart], name, {
                    type: typeOf(name),
                    lastModified: file.lastModified,
                  })
                }),
            )
          },
        )
      })
      .catch(reject)
  })
}
