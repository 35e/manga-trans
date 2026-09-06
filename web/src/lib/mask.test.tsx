import { createRoot } from 'react-dom/client'
import { flushSync } from 'react-dom'
import { useMasks } from '../hooks/useMasks'
import type { Masks } from '../hooks/useMasks'
import { MaskCanvas } from '../components/MaskCanvas'
import type { GalleryImage } from './images'
import { Mask } from './mask'

async function alpha(blob: Blob, x: number, y: number) {
  const bitmap = await createImageBitmap(blob)
  const canvas = document.createElement('canvas')
  canvas.width = bitmap.width
  canvas.height = bitmap.height
  try {
    const ctx = canvas.getContext('2d')!
    ctx.drawImage(bitmap, 0, 0)
    return ctx.getImageData(x, y, 1, 1).data[3]
  } finally {
    bitmap.close()
    canvas.width = canvas.height = 0
  }
}

// Browser console: await (await import('/src/lib/mask.test.tsx')).checkMaskMemory()
export async function checkMaskMemory() {
  const host = document.createElement('div')
  const root = createRoot(host)
  let api!: Masks
  let changes = 0
  function Harness() {
    api = useMasks(() => { changes += 1 })
    return null
  }
  const source = new Mask(16, 16)
  source.boxes([[2, 2, 8, 8]])
  const png = await source.snapshot()
  source.dispose()
  const page: GalleryImage = {
    id: 'page', width: 16, height: 16, file: new File([png], 'page.png'),
    url: '', name: 'page.png', size: png.size, addedAt: 0,
  }
  const restore = Mask.restore
  let decodes = 0
  Mask.restore = async (...args) => { decodes += 1; return restore(...args) }
  try {
    flushSync(() => root.render(<Harness />))
    if (api.get(page) !== null) throw new Error('Reading an absent mask must not allocate it')
    await api.restore({ page: png, idle: png })
    const saved = await api.snapshot()
    if (decodes || saved.page !== png || saved.idle !== png) throw new Error('Restore and snapshot must leave idle masks compressed')
    const [mask, same] = await Promise.all([api.load(page), api.load(page)])
    if (!mask || mask !== same || Number(decodes) !== 1) throw new Error('Concurrent loads must share one decoded mask')
    mask.boxes([[3, 3, 4, 4]], true)
    await api.release(page.id)
    if (api.get(page)) throw new Error('Released masks must not retain a drawable canvas')
    const edited = await api.load(page)
    if (!edited || await alpha(await edited.snapshot(), 3, 3) !== 0 || await alpha(await edited.snapshot(), 2, 2) !== 255) {
      throw new Error('Release and reload must preserve erasing and untouched alpha')
    }
    const cached = await api.snapshot()
    edited.snapshot = () => { throw new Error('An unchanged release must reuse its compressed PNG') }
    await api.release(page.id)
    if ((await api.snapshot()).page !== cached.page) throw new Error('Clean release must retain saved pixels')

    const raced = (await api.load(page))!
    raced.boxes([[10, 10, 12, 12]])
    const snapshot = raced.snapshot.bind(raced)
    // ES2023 target: Promise.withResolvers is not available.
    let finish!: (blob: Blob) => void
    raced.snapshot = () => new Promise(resolve => { finish = resolve })
    const releasing = api.release(page.id)
    const old = await snapshot()
    raced.boxes([[12, 12, 14, 14]])
    finish(old)
    await releasing
    if (api.get(page) !== raced) throw new Error('An edit during serialization must keep its canvas alive')
    raced.snapshot = snapshot
    await api.release(page.id)
    const reloaded = (await api.load(page))!
    if (await alpha(await reloaded.snapshot(), 12, 12) !== 255) throw new Error('Serialization races must not lose later edits')

    reloaded.boxes([[0, 0, 1, 1]])
    const latest = reloaded.snapshot.bind(reloaded)
    reloaded.snapshot = () => new Promise(resolve => { finish = resolve })
    const leaving = api.release(page.id)
    if (await api.load(page) !== reloaded) throw new Error('Reactivation should retain the loaded mask')
    finish(await latest())
    await leaving
    reloaded.snapshot = latest
    if (api.get(page) !== reloaded || await alpha(await reloaded.snapshot(), 0, 0) !== 255) {
      throw new Error('A late release must not dispose a reactivated page')
    }

    for (const invalidate of ['drop', 'clear', 'restore'] as const) {
      await api.restore({ page: png })
      const held = (await api.load(page))!
      held.boxes([[1, 1, 2, 2]])
      const pixels = await held.snapshot()
      held.snapshot = () => new Promise(resolve => { finish = resolve })
      const pending = api.release(page.id)
      if (invalidate === 'drop') api.drop(page.id)
      else if (invalidate === 'clear') api.clear()
      else await api.restore({ page: png })
      finish(pixels)
      await pending
      if (api.get(page) || (invalidate !== 'restore' && Object.keys(await api.snapshot()).length)) {
        throw new Error(`${invalidate} must not let a late release resurrect deleted pixels`)
      }
    }

    for (const invalidate of ['drop', 'clear', 'restore', 'unmount'] as const) {
      await api.restore({ page: png })
      let proceed!: () => void
      const gate = new Promise<void>(resolve => { proceed = resolve })
      Mask.restore = async (...args) => { await gate; return restore(...args) }
      const pending = api.load(page)
      if (invalidate === 'drop') api.drop(page.id)
      else if (invalidate === 'clear') api.clear()
      else if (invalidate === 'restore') await api.restore({ page: png })
      else flushSync(() => root.unmount())
      proceed()
      if (await pending !== null || api.get(page) !== null) throw new Error(`${invalidate} must invalidate late mask decoding`)
      if (invalidate !== 'unmount' && invalidate !== 'restore' && Object.keys(await api.snapshot()).length) {
        throw new Error(`${invalidate} must not let deleted mask pixels return to persistence`)
      }
    }
    if (!changes) throw new Error('Mask edits must notify project persistence')
    return { pixels: 'preserved', idle: 'compressed', races: 'invalidated' }
  } finally {
    Mask.restore = restore
    flushSync(() => root.unmount())
    host.remove()
  }
}

// Browser console: await (await import('/src/lib/mask.test.tsx')).checkMaskFrames()
export async function checkMaskFrames() {
  const host = document.createElement('div')
  host.style.cssText = 'position:fixed;left:0;top:0;width:64px;height:64px'
  document.body.append(host)
  const root = createRoot(host)
  const mask = new Mask(64, 64)
  const next = new Mask(64, 64)
  let paints = 0
  let strokes = 0
  const show = mask.showOn.bind(mask)
  mask.showOn = canvas => { paints += 1; show(canvas) }
  const request = window.requestAnimationFrame
  const cancel = window.cancelAnimationFrame
  const frames = new Map<number, FrameRequestCallback>()
  let id = 0
  window.requestAnimationFrame = callback => { frames.set(++id, callback); return id }
  window.cancelAnimationFrame = key => { frames.delete(key) }
  const tick = () => {
    const pending = [...frames.values()]
    frames.clear()
    for (const callback of pending) callback(0)
  }
  const render = (value: Mask) => flushSync(() => root.render(
    <MaskCanvas page={{ width: 64, height: 64 }} mask={value} brush={{ radius: 2, erase: false }} panning={false} onStroke={() => { strokes += 1 }} />,
  ))
  try {
    render(mask)
    tick()
    paints = 0
    const canvas = host.querySelector('canvas')!
    // Synthetic pointer events cannot acquire native capture; exercise the same handlers.
    canvas.setPointerCapture = () => {}
    canvas.hasPointerCapture = () => false
    const rect = canvas.getBoundingClientRect()
    const pointer = (type: string, x: number, y: number, samples?: [number, number][]) => {
      const event = new PointerEvent(type, { bubbles: true, pointerId: 1, button: 0, clientX: rect.left + x, clientY: rect.top + y })
      if (samples) Object.defineProperty(event, 'getCoalescedEvents', { value: () => samples.map(([sx, sy]) => new PointerEvent(type, { clientX: rect.left + sx, clientY: rect.top + sy })) })
      canvas.dispatchEvent(event)
    }
    pointer('pointerdown', 4, 4)
    pointer('pointermove', 48, 32, [[4, 32], [32, 32], [32, 4], [48, 4], [48, 32]])
    pointer('pointermove', 56, 32)
    pointer('pointerup', 56, 48)
    if (paints || frames.size !== 1 || strokes !== 1) throw new Error('Pointer bursts must schedule one repaint and one final stroke update')
    tick()
    if (Number(paints) !== 1 || canvas.getContext('2d')!.getImageData(56, 46, 1, 1).data[3] === 0) {
      throw new Error('The final frame must visibly include the pointer-up endpoint')
    }
    const png = await mask.snapshot()
    for (const [x, y] of [[4, 20], [20, 32], [32, 16], [40, 4], [48, 20], [56, 46]]) {
      if (await alpha(png, x, y) !== 255) throw new Error('Frame coalescing must retain every underlying stroke sample')
    }
    pointer('pointerdown', 8, 8)
    render(next)
    tick()
    if (Number(paints) !== 1 || canvas.getContext('2d')!.getImageData(8, 8, 1, 1).data[3] !== 0) {
      throw new Error('Switching masks must cancel old frames and display only the new pixels')
    }
    pointer('pointermove', 30, 30)
    if (!next.empty) throw new Error('Switching masks must cancel the previous pointer stroke')
    pointer('pointerdown', 10, 10)
    flushSync(() => root.unmount())
    if (frames.size) throw new Error('Unmount must cancel scheduled overlay work')
    return { repaint: 'one per frame', samples: 'preserved', cleanup: 'cancelled' }
  } finally {
    flushSync(() => root.unmount())
    window.requestAnimationFrame = request
    window.cancelAnimationFrame = cancel
    mask.dispose()
    next.dispose()
    host.remove()
  }
}
