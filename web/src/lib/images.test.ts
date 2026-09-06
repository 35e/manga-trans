import { loadImage, validateImageDimensions } from './images'

// Browser console: await (await import('/src/lib/images.test.ts')).checkImageLimits()
export async function checkImageLimits() {
  validateImageDimensions(4000, 6000)
  validateImageDimensions(16384, 1)
  for (const [width, height] of [[4001, 6000], [16385, 1], [0, 1]]) {
    let rejected = false
    try { validateImageDimensions(width, height) } catch { rejected = true }
    if (!rejected) throw new Error(`Unsafe canvas dimensions accepted: ${width} × ${height}`)
  }

  const tooWide = new File([
    '<svg xmlns="http://www.w3.org/2000/svg" width="16385" height="1"><rect width="100%" height="100%"/></svg>',
  ], 'too-wide.svg', { type: 'image/svg+xml' })
  let rejected = false
  try { await loadImage(tooWide) } catch { rejected = true }
  if (!rejected) throw new Error('The actual image probe must reject excessive decoded dimensions')

  const block = new Blob([new Uint8Array(1024)])
  const tooLarge = new File(Array(32769).fill(block), 'too-large.png', { type: 'image/png' })
  rejected = false
  try { await loadImage(tooLarge) } catch { rejected = true }
  if (!rejected) throw new Error('Oversized source bytes must fail before normal image decoding')

  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = 8
  const png = await (await fetch(canvas.toDataURL())).blob()
  canvas.width = canvas.height = 0
  const page = await loadImage(new File([png], 'chapter/page.png', { type: 'image/png' }))
  if (!page) throw new Error('Ordinary PNG import failed')
  try {
    if (page.width !== 8 || page.height !== 8 || page.name !== 'page.png') throw new Error('Image metadata changed')
    const decoded = await createImageBitmap(await (await fetch(page.url)).blob())
    decoded.close()
  } finally {
    URL.revokeObjectURL(page.url)
  }
  return 'Image dimensions and source bytes are bounded; ordinary PNGs remain usable'
}
