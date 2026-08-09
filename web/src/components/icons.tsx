/**
 * The line icons, one place rather than spelled out wherever they are used.
 *
 * All drawn on the same 24 grid with the same stroke, so they sit together, and
 * all `currentColor`, so they take the colour of whatever they are put in.
 */

type Props = { className?: string }

function Line({
  className = 'size-4',
  width = 1.8,
  children,
}: Props & { width?: number; children: React.ReactNode }) {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={width}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      {children}
    </svg>
  )
}

export function GearIcon(props: Props) {
  return (
    <Line {...props} width={1.7}>
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.6 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.6a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </Line>
  )
}

export function CloseIcon(props: Props) {
  return (
    <Line {...props}>
      <path d="M6 6l12 12M18 6 6 18" />
    </Line>
  )
}

export function DownloadIcon(props: Props) {
  return (
    <Line {...props}>
      <path d="M12 4v11m0 0 4-4m-4 4-4-4" />
      <path d="M5 19h14" />
    </Line>
  )
}

export function UploadIcon(props: Props) {
  return (
    <Line {...props} width={1.5}>
      <path d="M12 16V4m0 0L8 8m4-4 4 4" />
      <path d="M3 15v3a3 3 0 0 0 3 3h12a3 3 0 0 0 3-3v-3" />
    </Line>
  )
}

export function ZoomInIcon(props: Props) {
  return (
    <Line {...props} width={2}>
      <path d="M12 6v12M6 12h12" />
    </Line>
  )
}

export function ZoomOutIcon(props: Props) {
  return (
    <Line {...props} width={2}>
      <path d="M6 12h12" />
    </Line>
  )
}

export function CheckIcon(props: Props) {
  return (
    <Line {...props} width={3.5}>
      <path d="m5 13 5 5L20 7" />
    </Line>
  )
}

/** Put this block back into the clean, after it was left alone. */
export function RestoreIcon(props: Props) {
  return (
    <Line {...props} width={1.9}>
      <path d="M4 12a8 8 0 1 0 2.3-5.6M4 4v4h4" />
    </Line>
  )
}

export function TrashIcon(props: Props) {
  return (
    <Line {...props}>
      <path d="M4 7h16M10 11v6M14 11v6M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
    </Line>
  )
}

export function PageIcon(props: Props) {
  return (
    <Line {...props} width={1.5}>
      <rect x="4" y="3" width="16" height="18" rx="2" />
      <path d="M8 8h5M8 12h8M8 16h6" />
    </Line>
  )
}

/** The dots on a row that can be dragged to reorder it. */
export function GripIcon({ className = 'size-3.5' }: Props) {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" fill="currentColor" className={className}>
      <circle cx="9" cy="6" r="1.6" />
      <circle cx="15" cy="6" r="1.6" />
      <circle cx="9" cy="12" r="1.6" />
      <circle cx="15" cy="12" r="1.6" />
      <circle cx="9" cy="18" r="1.6" />
      <circle cx="15" cy="18" r="1.6" />
    </svg>
  )
}
