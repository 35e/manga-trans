import { useEffect, useState } from 'react'
import { ready } from '../lib/fit'

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
