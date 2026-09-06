import type { Box } from './api'
import { validateImageDimensions } from './images'

export type Point = { x: number; y: number }
export type Brush = { radius: number; erase: boolean }

export function mark(mask: Mask, boxes: Box[], letters: ImageBitmap | null) {
  if (boxes.length === 0) return
  if (letters) mask.letters(letters, boxes)
  else mask.boxes(boxes)
}

const TINT = 'rgba(79, 70, 229, 0.45)'

export class Mask {
  readonly width: number
  readonly height: number
  private readonly canvas: HTMLCanvasElement
  private readonly ctx: CanvasRenderingContext2D
  private marked = false
  private disposed = false
  private saved: Promise<Blob> | null = null
  private readonly onChange?: () => void

  constructor(width: number, height: number, onChange?: () => void) {
    validateImageDimensions(width, height)
    this.onChange = onChange
    this.width = width
    this.height = height
    this.canvas = document.createElement('canvas')
    this.canvas.width = width
    this.canvas.height = height

    const ctx = this.canvas.getContext('2d')
    if (!ctx) throw new Error('this browser has no 2D canvas')
    this.ctx = ctx
    ctx.fillStyle = '#fff'
    ctx.strokeStyle = '#fff'
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
  }

  get empty(): boolean {
    return !this.marked
  }

  private changed() {
    this.saved = null
    this.onChange?.()
  }

  private into(erase: boolean) {
    if (this.disposed) throw new DOMException('Mask was released', 'AbortError')
    this.ctx.globalCompositeOperation = erase ? 'destination-out' : 'source-over'
    if (!erase) this.marked = true
  }

  boxes(boxes: Box[], erase = false) {
    if (boxes.length === 0) return
    this.into(erase)
    for (const [x0, y0, x1, y1] of boxes) {
      this.ctx.fillRect(x0, y0, x1 - x0, y1 - y0)
    }
    this.changed()
  }

  letters(source: CanvasImageSource, boxes: Box[]) {
    const drawable = boxes.filter(([x0, y0, x1, y1]) => x1 > x0 && y1 > y0)
    if (drawable.length === 0) return

    this.into(false)
    for (const [x0, y0, x1, y1] of drawable) {
      const width = x1 - x0
      const height = y1 - y0
      this.ctx.drawImage(source, x0, y0, width, height, x0, y0, width, height)
    }
    this.changed()
  }

  dot(at: Point, brush: Brush) {
    this.into(brush.erase)
    this.ctx.beginPath()
    this.ctx.arc(at.x, at.y, brush.radius, 0, Math.PI * 2)
    this.ctx.fill()
    this.changed()
  }

  stroke(from: Point, to: Point, brush: Brush) {
    this.into(brush.erase)
    this.ctx.lineWidth = brush.radius * 2
    this.ctx.beginPath()
    this.ctx.moveTo(from.x, from.y)
    this.ctx.lineTo(to.x, to.y)
    this.ctx.stroke()
    this.changed()
  }

  clear() {
    if (this.disposed) throw new DOMException('Mask was released', 'AbortError')
    if (!this.marked) return
    this.ctx.globalCompositeOperation = 'source-over'
    this.ctx.clearRect(0, 0, this.width, this.height)
    this.marked = false
    this.changed()
  }

  showOn(target: HTMLCanvasElement) {
    const ctx = target.getContext('2d')
    if (!ctx) return
    ctx.globalCompositeOperation = 'source-over'
    ctx.clearRect(0, 0, this.width, this.height)
    if (this.disposed) return
    ctx.drawImage(this.canvas, 0, 0)
    ctx.globalCompositeOperation = 'source-in'
    ctx.fillStyle = TINT
    ctx.fillRect(0, 0, this.width, this.height)
    ctx.globalCompositeOperation = 'source-over'
  }

  snapshot(): Promise<Blob> {
    if (this.disposed) return Promise.reject(new DOMException('Mask was released', 'AbortError'))
    if (!this.saved) {
      const saving = new Promise<Blob>((resolve, reject) => {
        this.canvas.toBlob(
          (blob) => blob ? resolve(blob) : reject(new Error('the mask could not be saved')),
          'image/png',
        )
      })
      this.saved = saving
      void saving.catch(() => {
        if (this.saved === saving) this.saved = null
      })
    }
    return this.saved
  }

  dispose() {
    this.disposed = true
    this.canvas.width = this.canvas.height = 0
    this.saved = null
  }

  static async restore(blob: Blob, onChange?: () => void): Promise<Mask> {
    const image = await createImageBitmap(blob)
    let mask: Mask | undefined
    try {
      mask = new Mask(image.width, image.height, onChange)
      mask.ctx.drawImage(image, 0, 0)
      for (let y = 0; y < image.height && !mask.marked; y += 32) {
        const pixels = mask.ctx.getImageData(0, y, image.width, Math.min(32, image.height - y)).data
        for (let at = 3; at < pixels.length; at += 4) {
          if (pixels[at] !== 0) {
            mask.marked = true
            break
          }
        }
      }
      mask.saved = Promise.resolve(blob)
      return mask
    } catch (error) {
      mask?.dispose()
      throw error
    } finally {
      image.close()
    }
  }

  toBlob(): Promise<Blob> {
    if (this.disposed) return Promise.reject(new DOMException('Mask was released', 'AbortError'))
    const out = document.createElement('canvas')
    out.width = this.width
    out.height = this.height

    const ctx = out.getContext('2d')
    if (!ctx) return Promise.reject(new Error('this browser has no 2D canvas'))
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, this.width, this.height)
    ctx.drawImage(this.canvas, 0, 0)

    return new Promise((resolve, reject) => {
      out.toBlob(
        (blob) => {
          out.width = out.height = 0
          if (blob) resolve(blob)
          else reject(new Error('the mask could not be saved'))
        },
        'image/png',
      )
    })
  }
}
