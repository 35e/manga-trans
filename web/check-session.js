// With Vite open: playwright-cli run-code --filename=check-session.js
// Uses an isolated browser context; never reads or replaces the user's project.
globalThis.checkSession = async page => {
  const context = await page.context().browser().newContext()
  const check = await context.newPage()
  try {
    await check.addInitScript(() => {
      const state = window.sessionCheck = { calls: [], pending: [], hold: null, failTranslation: null, failSave: false }
      state.canvases = []
      state.decoding = state.maxDecoding = 0
      const create = document.createElement.bind(document)
      document.createElement = (...args) => {
        const element = create(...args)
        if (args[0] === 'canvas') state.canvases.push(new WeakRef(element))
        return element
      }
      const Image = window.Image
      window.Image = class extends Image {
        constructor(...args) {
          super(...args)
          state.maxDecoding = Math.max(state.maxDecoding, ++state.decoding)
          let done = false
          const settled = () => {
            if (!done) state.decoding--
            done = true
          }
          this.addEventListener('load', settled, { once: true })
          this.addEventListener('error', settled, { once: true })
        }
      }
      const fetch = window.fetch.bind(window)
      window.fetch = async (input, init) => {
        const path = new URL(String(input), location.href).pathname
        if (!path.startsWith('/api/')) return fetch(input, init)
        const file = init?.body?.get('image')
        const texts = JSON.parse(init?.body?.get('texts') ?? '[]')
        const name = file?.name ?? texts[0]?.split(':')[0] ?? ''
        state.calls.push({ path, name })
        const answer = () => {
          if (path === '/api/letters' || path === '/api/clean') {
            return new Response(state.png, { headers: { 'Content-Type': 'image/png' } })
          }
          if (path === '/api/translate' && name === state.failTranslation) {
            return Response.json({ error: 'Translation temporarily unavailable' }, { status: 503 })
          }
          const answers = {
            '/api/models': { models: ['test-model'] },
            '/api/languages': { languages: [{ code: 'ja', name: 'Japanese', rtl: true }] },
            '/api/prompt': { prompt: '' },
            '/api/detect': { width: 128, height: 192, regions: [{ box: [10, 10, 118, 96], confidence: 1 }] },
            '/api/read': { texts: [`${name}: source`] },
            '/api/translate': { texts: texts.map(text => `Translated ${text}`) },
          }
          return Response.json(answers[path] ?? {})
        }
        // Deliberately ignore abort to prove late results cannot mutate state.
        if (state.hold === path) return new Promise(resolve => {
          state.pending.push({ signal: init.signal, release: () => resolve(answer()) })
        })
        return answer()
      }
      const put = IDBObjectStore.prototype.put
      IDBObjectStore.prototype.put = function (...args) {
        if (state.failSave) throw new DOMException('Simulated full storage', 'QuotaExceededError')
        return put.apply(this, args)
      }
      const open = indexedDB.open.bind(indexedDB)
      indexedDB.open = (...args) => {
        if (localStorage.getItem('check-fail-load')) throw new DOMException('Storage temporarily unavailable', 'UnknownError')
        return open(...args)
      }
    })
    await check.goto(page.url())
    await check.getByText('Project saved', { exact: true }).waitFor()
    await check.evaluate(async () => {
      const { pack } = await import('/src/lib/zip.ts')
      const canvas = document.createElement('canvas')
      canvas.width = 128
      canvas.height = 192
      canvas.getContext('2d').fillRect(0, 0, 128, 192)
      const bytes = new Uint8Array(await (await fetch(canvas.toDataURL())).arrayBuffer())
      window.sessionCheck.png = bytes
      window.sessionCheck.failTranslation = 'b.png'
      const archive = await pack([{ name: 'a.png', bytes }, { name: 'b.png', bytes }])
      const files = new DataTransfer()
      files.items.add(new File([archive], 'session-check.cbz', { lastModified: 1234 }))
      const input = document.querySelector('input[type=file]')
      input.files = files.files
      input.dispatchEvent(new Event('change', { bubbles: true }))
    })
    await check.getByRole('button', { name: 'Open folder session-check, 2 pages' }).click()
    await check.getByRole('button', { name: 'Translate folder', exact: true }).click()
    await check.getByRole('button', { name: 'Close the review' }).waitFor()
    await check.getByRole('button', { name: /^Edit / }).first().click()
    await check.locator('textarea').fill('Human edited translation')
    await check.getByRole('button', { name: 'Clean up', exact: true }).click()
    await check.locator('canvas').click({ position: { x: 15, y: 15 } })
    await check.waitForFunction(async () => {
      const data = await (await import('/src/lib/project.ts')).loadProject()
      return data && Object.keys(data.touchups).length > 0 &&
        Object.values(data.lettering).some(lines => lines[0]?.text === 'Human edited translation')
    })
    await check.getByText('Project saved', { exact: true }).waitFor()
    const inspect = async () => check.evaluate(async () => {
      const data = await (await import('/src/lib/project.ts')).loadProject()
      const first = data.images.find(image => image.name === 'a.png')
      const hash = async blob => blob ? Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', await blob.arrayBuffer()))).join(',') : null
      return {
        ids: data.images.map(image => image.id), file: first.file.name, mtime: first.file.lastModified,
        line: data.lettering[first.id]?.[0], mask: await hash(data.masks[first.id]),
        touchup: await hash(data.touchups[first.id]), cleaned: await hash(data.cleaned[first.id]),
        active: data.activeId, folder: data.openFolder, source: data.language, target: data.target,
      }
    })
    const before = await inspect()
    await check.reload()
    await check.getByText('Project saved', { exact: true }).waitFor()
    if (JSON.stringify(await inspect()) !== JSON.stringify(before) ||
      await check.locator('textarea').inputValue() !== 'Human edited translation') {
      throw new Error('Reload must restore files, IDs, edits, masks, cleanup, selection, and settings')
    }
    await check.evaluate(async () => {
      const data = await (await import('/src/lib/project.ts')).loadProject()
      window.sessionCheck.png = new Uint8Array(await data.images[0].file.arrayBuffer())
      window.sessionCheck.calls = []
    })
    await check.getByRole('button', { name: 'Resume folder', exact: true }).click()
    await check.getByRole('button', { name: 'Close the review' }).waitFor()
    await check.getByText('Project saved', { exact: true }).waitFor()
    const resumed = await check.evaluate(() => window.sessionCheck.calls)
    if (resumed.some(call => call.path === '/api/detect' || call.path === '/api/read') ||
      resumed.filter(call => call.path === '/api/translate').length !== 1 ||
      resumed.filter(call => call.path === '/api/clean').length !== 1 ||
      resumed.some(call => call.name && call.name !== 'b.png')) {
      throw new Error('Resume must process only the failed page stages, not completed OCR or edited translations')
    }
    if (JSON.stringify(await inspect()) !== JSON.stringify(before)) {
      throw new Error('Resuming another page must preserve the edited page and its touch-up')
    }

    check.once('dialog', dialog => dialog.dismiss())
    await check.getByRole('button', { name: 'Reprocess all pages' }).click()
    if (JSON.stringify(await inspect()) !== JSON.stringify(before)) throw new Error('Declining reprocess must preserve edits')
    await check.evaluate(() => { window.sessionCheck.calls = [] })
    check.once('dialog', dialog => dialog.accept())
    await check.getByRole('button', { name: 'Reprocess all pages' }).click()
    await check.getByRole('button', { name: 'Close the review' }).waitFor()
    await check.getByRole('button', { name: 'Stop', exact: true }).waitFor({ state: 'hidden' })
    await check.getByText('Project saved', { exact: true }).waitFor()
    await check.waitForFunction(async () => {
      const data = await (await import('/src/lib/project.ts')).loadProject()
      return data && Object.keys(data.cleaned).length === 2 &&
        Object.values(data.lettering).every(lines => lines[0]?.text.startsWith('Translated'))
    })
    const reprocessed = await check.evaluate(() => window.sessionCheck.calls)
    for (const path of ['/api/detect', '/api/read', '/api/translate', '/api/letters', '/api/clean']) {
      if (reprocessed.filter(call => call.path === path).length !== 2) throw new Error(`Reprocess must invalidate ${path}`)
    }
    if ((await inspect()).touchup === before.touchup) throw new Error('Reprocess must discard obsolete touch-up masks')

    await check.getByRole('button', { name: /^Edit / }).first().click()
    await check.evaluate(() => { window.sessionCheck.hold = '/api/translate' })
    await check.getByRole('button', { name: 'Translate again', exact: true }).click()
    await check.getByRole('button', { name: 'Replace edited translations?', exact: true }).click()
    await check.waitForFunction(() => window.sessionCheck.pending.length === 1)
    await check.locator('textarea').fill('Latest manual edit')
    if (!await check.evaluate(() => window.sessionCheck.pending[0].signal.aborted)) {
      throw new Error('Editing must abort the obsolete translation request')
    }
    await check.evaluate(() => {
      window.sessionCheck.pending.shift().release()
      window.sessionCheck.hold = null
    })
    await check.evaluate(() => new Promise(requestAnimationFrame))
    if (await check.locator('textarea').inputValue() !== 'Latest manual edit') throw new Error('Late translation overwrote a newer edit')

    await check.evaluate(() => { window.sessionCheck.failSave = true })
    await check.locator('textarea').fill('Unsaved edit survives retry')
    await check.getByRole('button', { name: 'Retry saving project' }).waitFor()
    if (!await check.evaluate(() => {
      const event = new Event('beforeunload', { cancelable: true })
      window.dispatchEvent(event)
      return event.defaultPrevented
    })) throw new Error('Unsaved work must warn before leaving')
    await check.evaluate(() => { window.sessionCheck.failSave = false })
    await check.getByRole('button', { name: 'Retry saving project' }).click()
    await check.getByText('Project saved', { exact: true }).waitFor()
    if ((await inspect()).line.text !== 'Unsaved edit survives retry') throw new Error('Save retry lost the newest edit')

    await check.evaluate(() => { window.sessionCheck.hold = '/api/clean' })
    check.once('dialog', dialog => dialog.accept())
    await check.getByRole('button', { name: 'Reprocess all pages' }).click()
    await check.waitForFunction(() => window.sessionCheck.pending.length === 1)
    await check.getByRole('button', { name: 'Delete a.png', exact: true }).click()
    if (!await check.evaluate(() => window.sessionCheck.pending[0].signal.aborted)) throw new Error('Deletion must abort cleanup')
    await check.evaluate(() => {
      window.sessionCheck.pending.shift().release()
      window.sessionCheck.hold = null
    })
    await check.getByRole('button', { name: 'Resume folder', exact: true }).waitFor({ state: 'visible' })
    await check.waitForFunction(async () => {
      const data = await (await import('/src/lib/project.ts')).loadProject()
      return data?.images.length === 1
    })
    const deleted = await check.evaluate(async () => {
      const data = await (await import('/src/lib/project.ts')).loadProject()
      const valid = new Set(data.images.map(image => image.id))
      return ['analyses', 'lettering', 'cleaned', 'masks', 'touchups'].every(key => Object.keys(data[key]).every(id => valid.has(id)))
    })
    if (!deleted) throw new Error('Late cleanup resurrected deleted page resources')
    await check.getByText('Project saved', { exact: true }).waitFor()
    await check.reload()
    await check.getByText('Project saved', { exact: true }).waitFor()
    if (await check.getByRole('button', { name: /Open page .*a.png/ }).count()) throw new Error('Deleted page returned after reload')

    await check.evaluate(() => localStorage.setItem('check-fail-load', '1'))
    await check.reload()
    await check.getByRole('button', { name: 'Retry loading project' }).waitFor()
    await check.evaluate(() => localStorage.removeItem('check-fail-load'))
    await check.getByRole('button', { name: 'Retry loading project' }).click()
    await check.getByRole('button', { name: /Open page .*b.png/ }).waitFor()
    await check.evaluate(async () => {
      const data = await (await import('/src/lib/project.ts')).loadProject()
      window.sessionCheck.png = new Uint8Array(await data.images[0].file.arrayBuffer())
      window.sessionCheck.hold = '/api/clean'
    })
    await check.getByRole('button', { name: 'Resume folder', exact: true }).click()
    await check.waitForFunction(() => window.sessionCheck.pending.length === 1)
    await check.getByRole('button', { name: 'Stop', exact: true }).click()
    if (!await check.evaluate(() => window.sessionCheck.pending[0].signal.aborted)) throw new Error('Stop must abort the active request immediately')
    await check.evaluate(() => window.sessionCheck.pending.shift().release())
    await check.getByRole('button', { name: 'Stopping…', exact: true }).waitFor({ state: 'hidden' })
    await check.getByText('Project saved', { exact: true }).waitFor()
    if (!await check.evaluate(async () => {
      const data = await (await import('/src/lib/project.ts')).loadProject()
      return Object.keys(data.cleaned).length === 0
    })) throw new Error('Stopped cleanup committed its late response')

    await check.evaluate(() => {
      window.sessionCheck.hold = null
      window.sessionCheck.maxDecoding = 0
      const transfer = new DataTransfer()
      for (let index = 1; index <= 10; index++) {
        transfer.items.add(new File([window.sessionCheck.png], `c${index}.png`, { type: 'image/png' }))
      }
      const input = document.querySelector('input[type=file]')
      input.files = transfer.files
      input.dispatchEvent(new Event('change', { bubbles: true }))
    })
    await check.waitForFunction(async () => {
      const data = await (await import('/src/lib/project.ts')).loadProject()
      return data?.images.length === 11
    })
    const decoders = await check.evaluate(() => window.sessionCheck.maxDecoding)
    if (decoders > 2 || decoders === 0) throw new Error(`Image import used ${decoders} simultaneous decoders`)
    await check.getByRole('button', { name: 'Resume folder', exact: true }).click()
    await check.getByRole('button', { name: 'Stop', exact: true }).waitFor({ state: 'hidden' })
    await check.getByText('Project saved', { exact: true }).waitFor()
    await check.getByRole('button', { name: 'Close the review' }).click()
    let canvases = 0
    for (let index = 1; index <= 10; index++) {
      await check.getByRole('button', { name: new RegExp(`Open page .*c${index}[.]png$`) }).click()
      await check.getByText('Loading editable masks…', { exact: true }).waitFor({ state: 'hidden' })
      canvases = await check.evaluate(() => window.sessionCheck.canvases
        .map(reference => reference.deref())
        .filter(canvas => canvas?.width === 128 && canvas.height === 192).length)
      if (canvases > 2) throw new Error(`Inactive pages retained ${canvases} full-size canvases`)
    }
    return { restored: 'editable project and masks', resumed: resumed.map(call => call.path), reprocessed: 'all stages', staleTranslation: 'discarded', quota: 'retry recovered newest edit', loadFailure: 'retry restored saved work', stop: 'request aborted, late result discarded', deletion: 'persisted without late resources', memory: { pages: 11, canvases, decoders } }
  } catch (cause) {
    const calls = await check.evaluate(() => window.sessionCheck?.calls)
    throw new Error(`${cause.message}\n${await check.locator('body').innerText()}\nCalls: ${JSON.stringify(calls)}`)
  } finally {
    await context.close()
  }
}
