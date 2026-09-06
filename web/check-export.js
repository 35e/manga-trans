// With Vite open in playwright-cli: playwright-cli run-code --filename=check-export.js
globalThis.checkExport = async page => {
  const context = await page.context().browser().newContext()
  const check = await context.newPage()
  try {
    await check.addInitScript(() => {
      const state = window.exportCheck = { fail: 'none', cleaned: [], downloads: [] }
      const fetch = window.fetch.bind(window)
      const create = URL.createObjectURL.bind(URL)
      URL.createObjectURL = blob => {
        const url = create(blob)
        if (blob.type === 'image/png' && !(blob instanceof File)) state.cleaned.push(url)
        return url
      }
      window.fetch = (input, init) => {
        const url = String(input)
        const path = new URL(url, location.href).pathname
        if (path.startsWith('/api/')) {
          const answers = {
            '/api/models': { models: [] },
            '/api/languages': { languages: [{ code: 'ja', name: 'Japanese', rtl: true }] },
            '/api/prompt': { prompt: '' },
            '/api/detect': { width: 32, height: 32, regions: [{ box: [4, 4, 28, 28], confidence: 1 }] },
            '/api/read': { texts: ['source text'] },
          }
          if (path === '/api/letters' || path === '/api/clean') {
            return Promise.resolve(new Response(state.png, { headers: { 'Content-Type': 'image/png' } }))
          }
          return Promise.resolve(Response.json(answers[path] ?? {}))
        }
        if (state.cleaned.includes(url) &&
          (state.fail === 'all' || (state.fail === 'one' && url === state.cleaned[0]))) {
          return Promise.reject(new Error('Cleaned page unavailable'))
        }
        return fetch(input, init)
      }
      // Inspect the actual archive handed to the browser download sink.
      const click = HTMLAnchorElement.prototype.click
      HTMLAnchorElement.prototype.click = function () {
        if (!this.download) return click.call(this)
        const name = this.download
        void fetch(this.href).then(answer => answer.arrayBuffer()).then(bytes => {
          state.downloads.push({ name, bytes })
        })
      }
    })
    await check.goto(page.url())
    await check.getByText('Project saved', { exact: true }).waitFor()
    await check.evaluate(async () => {
      const { pack } = await import('/src/lib/zip.ts')
      const canvas = document.createElement('canvas')
      canvas.width = canvas.height = 32
      canvas.getContext('2d').fillRect(0, 0, 32, 32)
      const data = canvas.toDataURL()
      const bytes = new Uint8Array(await (await fetch(data)).arrayBuffer())
      window.exportCheck.png = bytes
      const archive = await pack([{ name: 'bad.png', bytes }, { name: 'good.png', bytes }])
      const files = new DataTransfer()
      files.items.add(new File([archive], 'export-check.cbz'))
      const input = document.querySelector('input[type=file]')
      input.files = files.files
      input.dispatchEvent(new Event('change', { bubbles: true }))
    })
    await check.getByRole('button', { name: 'Open folder export-check, 2 pages' }).click()
    await check.getByRole('button', { name: 'Read & clean folder', exact: true }).click()
    await check.getByRole('button', { name: 'Close the review' }).waitFor()
    await check.waitForFunction(() => window.exportCheck.cleaned.length === 2)

    await check.evaluate(() => { window.exportCheck.fail = 'one' })
    check.once('dialog', dialog => dialog.dismiss())
    await check.getByRole('button', { name: 'Download chapter', exact: true }).last().click()
    const retry = check.getByRole('button', { name: 'Retry export', exact: true })
    await retry.waitFor()
    if (!(await check.getByRole('alert').innerText()).includes('bad.png') ||
      await check.evaluate(() => window.exportCheck.downloads.length) !== 0) {
      throw new Error('Declining a partial export must report the failed page and download nothing')
    }

    check.once('dialog', dialog => dialog.accept())
    await retry.click()
    await check.waitForFunction(() => window.exportCheck.downloads.length === 1)
    const partial = await check.evaluate(async () => {
      const { expand } = await import('/src/lib/zip.ts')
      const saved = window.exportCheck.downloads[0]
      return { name: saved.name, pages: (await expand(new File([saved.bytes], saved.name))).map(file => file.name) }
    })
    if (partial.name !== 'export-check-cleaned-partial.cbz' ||
      partial.pages.join() !== 'good.png') {
      throw new Error('Explicit partial export must contain only the successful page, never an original substitute')
    }

    await check.evaluate(() => { window.exportCheck.fail = 'all' })
    await retry.click()
    await check.waitForFunction(() => document.querySelector('[role=alert]')?.textContent.includes('good.png'))
    if (await check.evaluate(() => window.exportCheck.downloads.length) !== 1) {
      throw new Error('An entirely failed export must not download an empty archive')
    }

    await check.evaluate(() => { window.exportCheck.fail = 'none' })
    await retry.click()
    await check.waitForFunction(() => window.exportCheck.downloads.length === 2)
    const recovered = await check.evaluate(async () => {
      const { expand } = await import('/src/lib/zip.ts')
      const saved = window.exportCheck.downloads[1]
      return { name: saved.name, pages: (await expand(new File([saved.bytes], saved.name))).map(file => file.name) }
    })
    if (recovered.name !== 'export-check-cleaned.cbz' ||
      recovered.pages.join() !== 'bad.png,good.png' || await check.getByRole('alert').count()) {
      throw new Error('Retry after recovery must export all pages and clear the error')
    }
    return { cancelled: 'no download', partial, allFailed: 'no download', recovered }
  } finally {
    await context.close()
  }
}
