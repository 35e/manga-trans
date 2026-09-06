import { useCallback, useEffect, useMemo, useRef } from 'react'

export type PageOperation = {
  pageId: string
  signal: AbortSignal
  check: () => void
  finish: () => void
}

export type PageOperations = {
  begin: (pageId: string, externalSignal?: AbortSignal) => PageOperation
  cancel: (pageId?: string) => void
}

export function usePageOperation(): PageOperations {
  const active = useRef<{ operation: PageOperation; abort: () => void } | null>(null)
  const mounted = useRef(true)

  const cancel = useCallback((pageId?: string) => {
    const current = active.current
    if (current && (pageId === undefined || current.operation.pageId === pageId)) {
      current.abort()
    }
  }, [])

  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      cancel()
    }
  }, [cancel])

  const begin = useCallback((pageId: string, externalSignal?: AbortSignal): PageOperation => {
    cancel()
    const controller = new AbortController()
    let finished = false
    const operation: PageOperation = {
      pageId,
      signal: controller.signal,
      check() {
        controller.signal.throwIfAborted()
        if (finished) throw new DOMException('The page operation has finished', 'AbortError')
      },
      finish() {
        finished = true
        externalSignal?.removeEventListener('abort', abort)
        if (active.current?.operation === operation) active.current = null
      },
    }
    const abort = () => {
      controller.abort(externalSignal?.reason)
      operation.finish()
    }
    active.current = { operation, abort }
    if (!mounted.current || externalSignal?.aborted) abort()
    else externalSignal?.addEventListener('abort', abort, { once: true })
    return operation
  }, [cancel])

  return useMemo(() => ({ begin, cancel }), [begin, cancel])
}
