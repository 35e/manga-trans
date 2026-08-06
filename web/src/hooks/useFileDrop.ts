import { useEffect, useRef, useState } from 'react'

type FileHandler = (files: FileList | File[] | null) => void

function carriesFiles(event: DragEvent) {
  return Array.from(event.dataTransfer?.types ?? []).includes('Files')
}

/**
 * Files dropped anywhere on the page, and files pasted into it. Listening at the
 * window keeps a stray drop from navigating the browser away from the app, and
 * gives the whole page one answer to "is something being dragged in right now".
 */
export function useFileDrop(onFiles: FileHandler) {
  const [dragging, setDragging] = useState(false)

  // dragenter/dragleave fire for every element the pointer crosses, so the depth
  // counter is what tells a move between children from a real exit.
  const depth = useRef(0)
  const handler = useRef(onFiles)
  handler.current = onFiles

  useEffect(() => {
    const stop = () => {
      depth.current = 0
      setDragging(false)
    }

    const onEnter = (event: DragEvent) => {
      if (!carriesFiles(event)) return
      depth.current += 1
      setDragging(true)
    }

    const onLeave = (event: DragEvent) => {
      if (!carriesFiles(event)) return
      depth.current -= 1
      if (depth.current <= 0) stop()
    }

    const onOver = (event: DragEvent) => {
      if (!carriesFiles(event)) return
      event.preventDefault()
      if (event.dataTransfer) event.dataTransfer.dropEffect = 'copy'
    }

    const onDrop = (event: DragEvent) => {
      if (!carriesFiles(event)) return
      event.preventDefault()
      stop()
      handler.current(event.dataTransfer?.files ?? [])
    }

    const onPaste = (event: ClipboardEvent) => {
      const files = Array.from(event.clipboardData?.files ?? [])
      if (files.length > 0) handler.current(files)
    }

    window.addEventListener('dragenter', onEnter)
    window.addEventListener('dragleave', onLeave)
    window.addEventListener('dragover', onOver)
    window.addEventListener('drop', onDrop)
    window.addEventListener('dragend', stop)
    window.addEventListener('paste', onPaste)

    return () => {
      window.removeEventListener('dragenter', onEnter)
      window.removeEventListener('dragleave', onLeave)
      window.removeEventListener('dragover', onOver)
      window.removeEventListener('drop', onDrop)
      window.removeEventListener('dragend', stop)
      window.removeEventListener('paste', onPaste)
    }
  }, [])

  return dragging
}
