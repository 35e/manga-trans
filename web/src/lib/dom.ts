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
