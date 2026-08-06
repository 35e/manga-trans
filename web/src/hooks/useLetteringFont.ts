import { useEffect, useState } from 'react'
import { ready } from '../lib/fit'

/**
 * Flips once the lettering face has arrived.
 *
 * Where the lines break is worked out by measuring, and measuring before the
 * face is in measures the fallback. Anything that depends on this re-measures
 * when it turns true.
 */
export function useLetteringFont(): boolean {
  const [loaded, setLoaded] = useState(false)

  useEffect(() => {
    let live = true
    void ready().then(() => {
      if (live) setLoaded(true)
    })
    return () => {
      live = false
    }
  }, [])

  return loaded
}
