import { useEffect, useState } from 'react'
import { models as listModels } from '../lib/api'

export function useLlamaCpp() {
  const [models, setModels] = useState<string[]>([])
  const [model, setModel] = useState('')
  const [target, setTarget] = useState('English')
  const [problem, setProblem] = useState<string | null>(null)

  useEffect(() => {
    let dropped = false
    listModels().then(
      (found) => {
        if (dropped) return
        setModels(found)
        setModel((chosen) => chosen || found[0] || '')
        setProblem(found.length === 0 ? 'llama.cpp has no models loaded' : null)
      },
      (cause: unknown) => {
        if (dropped) return
        setProblem(cause instanceof Error ? cause.message : String(cause))
      },
    )
    return () => {
      dropped = true
    }
  }, [])

  return { models, model, setModel, target, setTarget, problem }
}
