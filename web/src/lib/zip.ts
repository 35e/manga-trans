import { unzip } from 'fflate'

/**
 * Pulling the pages out of an archive.
 *
 * A chapter arrives as a zip far more often than as fifty loose files, and the
 * order the pages are named in is the order they are meant to be read in — so
 * they come out sorted the way a person would sort them, not the way a computer
 * would: page 2 before page 10.
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
 * Every image inside a zip, in the order their names put them.
 *
 * Rejects only when the archive itself cannot be read; an archive holding
 * nothing that is an image comes back empty, which is a different thing and is
 * reported differently.
 */
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
