import { createElement, useLayoutEffect } from 'react'
import { flushSync } from 'react-dom'
import { createRoot } from 'react-dom/client'
import { usePageOperation } from '../hooks/usePageOperation'
import type { PageOperations } from '../hooks/usePageOperation'

// Browser console has no static imports: load this module and call checkPageOperations().
export function checkPageOperations() {
  const host = document.createElement('div')
  const root = createRoot(host)
  let operations!: PageOperations
  function Probe() {
    const current = usePageOperation()
    useLayoutEffect(() => { operations = current })
    return null
  }

  let unmounted = false
  try {
    flushSync(() => root.render(createElement(Probe)))
    const original = operations
    const superseded = operations.begin('one')
    const newer = operations.begin('two')
    superseded.finish()
    operations.cancel('one')
    newer.check()
    operations.cancel('two')
    if (!superseded.signal.aborted || !newer.signal.aborted) {
      throw new Error('Superseding and matching cancellation must abort their requests')
    }

    const batch = new AbortController()
    const linked = operations.begin('three', batch.signal)
    batch.abort()
    if (!linked.signal.aborted) throw new Error('Batch cancellation must abort the page request')
    linked.finish()

    const completedBatch = new AbortController()
    const completed = operations.begin('four', completedBatch.signal)
    completed.finish()
    completedBatch.abort()
    if (completed.signal.aborted) throw new Error('Finishing must detach the external signal')

    const alreadyAborted = operations.begin('five', batch.signal)
    if (!alreadyAborted.signal.aborted) throw new Error('An already cancelled batch must not start work')
    const live = operations.begin('six')
    completed.finish()
    linked.finish()
    live.check()
    flushSync(() => root.render(createElement(Probe)))
    if (operations !== original) throw new Error('The operation controls must remain stable across renders')
    live.check()
    flushSync(() => root.unmount())
    unmounted = true
    if (!live.signal.aborted) throw new Error('Unmounting must abort the active page request')

    for (const stale of [superseded, newer, linked, completed, alreadyAborted, live]) {
      let rejected = false
      try {
        stale.check()
      } catch (cause) {
        if (!(cause instanceof Error) || cause.name !== 'AbortError') throw cause
        rejected = true
      }
      if (!rejected) throw new Error(`Stale operation ${stale.pageId} may still commit`)
    }
    return 'Page operations reject superseded, cancelled, completed, and unmounted work'
  } finally {
    if (!unmounted) flushSync(() => root.unmount())
  }
}
