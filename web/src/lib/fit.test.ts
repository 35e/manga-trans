import { compose } from './compose'
import { fitSize, fontFor, LINE_HEIGHT, linesFor, ready, strokeFor } from './fit'

// Browser console has no static imports: (await import('/src/lib/fit.test.ts')).checkFitting()
export async function checkFitting() {
  await ready()
  const canvas = document.createElement('canvas')
  canvas.width = 160
  canvas.height = 140
  const context = canvas.getContext('2d')!
  context.fillStyle = '#a0a0a0'
  context.fillRect(0, 0, canvas.width, canvas.height)
  const background = canvas.toDataURL()
  context.font = fontFor(20)
  const twoParts = linesFor('MMMMMM', context.measureText('MMMM').width + strokeFor(20) * 2, 20)
  if (twoParts.length !== 2) throw new Error('A final fragment that fits must not split again')
  const dotsWidth = Math.max(context.measureText('HELLO').width, context.measureText('...').width) + strokeFor(20) * 2
  const dotted = linesFor('HELLO......', dotsWidth, 20)
  if (dotted.join('|') !== 'HELLO|...') {
    throw new Error('Repeated dots must be capped at three and separated from a word')
  }
  const dashesWidth = Math.max(context.measureText('WAIT').width, context.measureText('---').width) + strokeFor(20) * 2
  const dashed = linesFor('WAIT------', dashesWidth, 20)
  if (dashed.join('|') !== 'WAIT|---') {
    throw new Error('Repeated punctuation must be capped at three and separated from a word')
  }
  const shortWidth = context.measureText('HELL').width + strokeFor(20) * 2
  const shortSize = fitSize('HELLO', shortWidth, 50, 20)
  const shortRows = linesFor('HELLO', shortWidth, shortSize)
  if (shortSize >= 20 || shortRows.length !== 1 || shortRows[0] !== 'HELLO') {
    throw new Error('Words with five letters or fewer must shrink instead of splitting')
  }
  const narrowSize = fitSize('incomprehensible', 40, 50, 32)
  const narrowParts = linesFor('incomprehensible', 40, narrowSize)
  if (narrowParts.filter((part) => part.endsWith('-')).length > 1) {
    throw new Error('Prefer a smaller font over splitting one word into more than two parts')
  }
  const cases = [
    { text: 'This is incomprehensible!', broken: true },
    { text: 'Good morning!', broken: false },
    { text: 'HELLO!', broken: false },
    { text: 'WAIT!', broken: false },
  ]
  const results = []
  for (const { text, broken } of cases) {
    const size = fitSize(text, 120, 100, 28)
    const rows = linesFor(text, 120, size)
    if (size > 28) throw new Error(`${text}: font exceeds the source-size ceiling`)
    if (rows.some((row) => row.endsWith('-')) !== broken) throw new Error(`${text}: wrong word breaking`)
    if (rows.filter((row) => row.endsWith('-')).length > 1) {
      throw new Error(`${text}: prefer a smaller font over splitting a word into three lines`)
    }
    if (rows.join('').replaceAll('-', '').replaceAll(' ', '') !== text.replaceAll(' ', '')) {
      throw new Error(`${text}: wrapping lost text`)
    }
    context.font = fontFor(size)
    if (rows.some((row) => context.measureText(row).width + strokeFor(size) * 2 > 120)) {
      throw new Error(`${text}: outlined text exceeds box width`)
    }
    if (rows.length * size * LINE_HEIGHT + strokeFor(size) * 2 > 100) {
      throw new Error(`${text}: outlined text exceeds box height`)
    }
    const output = await compose(background, 160, 140, [
      { text, size, box: [20, 20, 140, 120], angle: 0 },
    ])
    const bitmap = await createImageBitmap(output)
    context.drawImage(bitmap, 0, 0)
    bitmap.close()
    const pixels = context.getImageData(0, 0, 160, 140).data
    for (let y = 0; y < 140; y++) {
      for (let x = 0; x < 160; x++) {
        if (x >= 20 && x < 140 && y >= 20 && y < 120) continue
        const at = (y * 160 + x) * 4
        // Allow minor canvas readback noise, not visible lettering outside the box.
        if (
          Math.abs(pixels[at] - 160) > 2 ||
          Math.abs(pixels[at + 1] - 160) > 2 ||
          Math.abs(pixels[at + 2] - 160) > 2
        ) {
          throw new Error(`${text}: exported lettering escaped its box at ${x},${y}`)
        }
      }
    }
    results.push({ text, size, rows })
  }
  return results
}
