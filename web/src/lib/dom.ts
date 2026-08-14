/** Whether a key was pressed into something being typed in. */
export function typingInto(target: EventTarget | null): boolean {
  const element = target as HTMLElement | null
  return Boolean(
    element &&
      (element.isContentEditable ||
        ['INPUT', 'TEXTAREA', 'SELECT'].includes(element.tagName)),
  )
}

/** The same question, of whatever has the focus right now. */
export function typingNow(): boolean {
  return typingInto(document.activeElement)
}

/**
 * What this browser was told to remember, the API keeping no settings. A browser
 * that will not remember is still one to work in, so neither of these throws.
 */
export function held(key: string): string | null {
  try {
    return window.localStorage.getItem(key)
  } catch {
    return null
  }
}

export function keep(key: string, value: string | null) {
  try {
    if (value === null) window.localStorage.removeItem(key)
    else window.localStorage.setItem(key, value)
  } catch {
    /* nothing to be done, and nothing worth stopping for */
  }
}
