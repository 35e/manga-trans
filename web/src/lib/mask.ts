import type { Box } from './api'

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

  constructor(width: number, height: number) {
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

  private into(erase: boolean) {
    this.ctx.globalCompositeOperation = erase ? 'destination-out' : 'source-over'
    if (!erase) this.marked = true
  }

  boxes(boxes: Box[], erase = false) {
    this.into(erase)
    for (const [x0, y0, x1, y1] of boxes) {
      this.ctx.fillRect(x0, y0, x1 - x0, y1 - y0)
    }
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
  }

  dot(at: Point, brush: Brush) {
    this.into(brush.erase)
    this.ctx.beginPath()
    this.ctx.arc(at.x, at.y, brush.radius, 0, Math.PI * 2)
    this.ctx.fill()
  }

  stroke(from: Point, to: Point, brush: Brush) {
    this.into(brush.erase)
    this.ctx.lineWidth = brush.radius * 2
    this.ctx.beginPath()
    this.ctx.moveTo(from.x, from.y)
    this.ctx.lineTo(to.x, to.y)
    this.ctx.stroke()
  }

  clear() {
    this.ctx.globalCompositeOperation = 'source-over'
    this.ctx.clearRect(0, 0, this.width, this.height)
    this.marked = false
  }

  showOn(target: HTMLCanvasElement) {
    const ctx = target.getContext('2d')
    if (!ctx) return
    ctx.globalCompositeOperation = 'source-over'
    ctx.clearRect(0, 0, this.width, this.height)
    ctx.drawImage(this.canvas, 0, 0)
    ctx.globalCompositeOperation = 'source-in'
    ctx.fillStyle = TINT
    ctx.fillRect(0, 0, this.width, this.height)
    ctx.globalCompositeOperation = 'source-over'
  }

  toBlob(): Promise<Blob> {
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
        (blob) =>
          blob ? resolve(blob) : reject(new Error('the mask could not be saved')),
        'image/png',
      )
    })
  }
}
