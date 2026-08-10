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
 * What this browser was told to remember. The API keeps no settings, so
 * everything chosen once and meant thereafter — the prompt, the language — is
 * kept here. A browser that will not remember anything is still a browser to
 * work in, so neither of these ever throws.
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
