import type { ReactNode } from 'react'

/**
 * The handful of controls the whole app is built from.
 *
 * They are here so that a button in the mask toolbar and a button in the header
 * are the same button — the interface has one of each thing, not one per place
 * it happens to appear.
 */

export const FOCUS =
  'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent'

type Variant = 'primary' | 'outline' | 'ghost' | 'danger'
type Size = 'sm' | 'md'

const VARIANTS: Record<Variant, string> = {
  primary: 'bg-accent text-white hover:bg-accent-lit',
  outline: 'border border-line text-muted hover:border-line hover:bg-raised hover:text-ink',
  ghost: 'text-muted hover:bg-raised hover:text-ink',
  danger: 'text-muted hover:bg-danger/15 hover:text-danger',
}

const SIZES: Record<Size, string> = {
  sm: 'gap-1.5 rounded-lg px-2.5 py-1.5 text-xs',
  md: 'gap-2 rounded-lg px-3 py-2 text-sm',
}

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: Variant
  size?: Size
}

export function Button({
  variant = 'outline',
  size = 'sm',
  className = '',
  ...rest
}: ButtonProps) {
  return (
    <button
      type="button"
      className={`inline-flex shrink-0 items-center justify-center font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${SIZES[size]} ${VARIANTS[variant]} ${FOCUS} ${className}`}
      {...rest}
    />
  )
}

/** A square button holding nothing but an icon. */
export function IconButton({
  label,
  variant = 'ghost',
  className = '',
  children,
  ...rest
}: ButtonProps & { label: string }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={rest.title ?? label}
      className={`inline-grid size-7 shrink-0 place-items-center rounded-lg transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${VARIANTS[variant]} ${FOCUS} ${className}`}
      {...rest}
    >
      {children}
    </button>
  )
}

export type Choice<T extends string> = {
  value: T
  label: ReactNode
  title?: string
}

/**
 * One of a few, shown as a row of pills. Used for every either-or in the app:
 * draw or erase, art or white, the page as it came in or as it was cleaned.
 */
export function Segmented<T extends string>({
  value,
  onChange,
  options,
  label,
  className = '',
}: {
  value: T
  onChange: (value: T) => void
  options: Choice<T>[]
  label?: string
  className?: string
}) {
  return (
    <div
      role="group"
      aria-label={label}
      className={`flex shrink-0 items-center gap-0.5 rounded-lg border border-line bg-canvas p-0.5 ${className}`}
    >
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          onClick={() => onChange(option.value)}
          aria-pressed={value === option.value}
          title={option.title}
          className={`rounded-md px-2.5 py-1 text-xs font-medium whitespace-nowrap transition-colors ${FOCUS} ${
            value === option.value
              ? 'bg-accent text-white'
              : 'text-muted hover:bg-raised hover:text-ink'
          }`}
        >
          {option.label}
        </button>
      ))}
    </div>
  )
}

/** A control that is either on or off, shown as one pill. */
export function Toggle({
  on,
  onChange,
  children,
  title,
}: {
  on: boolean
  onChange: (on: boolean) => void
  children: ReactNode
  title?: string
}) {
  return (
    <button
      type="button"
      onClick={() => onChange(!on)}
      aria-pressed={on}
      title={title}
      className={`shrink-0 rounded-lg border px-2.5 py-1.5 text-xs font-medium transition-colors ${FOCUS} ${
        on
          ? 'border-accent bg-accent/15 text-accent-lit'
          : 'border-line text-muted hover:bg-raised hover:text-ink'
      }`}
    >
      {children}
    </button>
  )
}

/** A label with its control beside it, which is most of every toolbar. */
export function Field({
  label,
  title,
  children,
}: {
  label: ReactNode
  title?: string
  children: ReactNode
}) {
  return (
    <label
      title={title}
      className="flex shrink-0 items-center gap-2 text-xs font-medium text-muted"
    >
      {label}
      {children}
    </label>
  )
}

export function Select({
  className = '',
  ...rest
}: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      className={`rounded-lg border border-line bg-raised px-2 py-1 text-xs text-ink transition-colors hover:border-faint disabled:opacity-40 ${FOCUS} ${className}`}
      {...rest}
    />
  )
}

export function TextInput({
  className = '',
  ...rest
}: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={`rounded-lg border border-line bg-raised px-2 py-1 text-xs text-ink transition-colors hover:border-faint ${FOCUS} ${className}`}
      {...rest}
    />
  )
}

/** The row every mode's tools sit in. */
export function Toolbar({ children }: { children: ReactNode }) {
  return (
    <div className="flex shrink-0 flex-wrap items-center gap-x-2.5 gap-y-2 border-b border-line bg-surface px-4 py-2">
      {children}
    </div>
  )
}

/** A hairline between groups in a toolbar. */
export function Divider() {
  return <span aria-hidden="true" className="h-5 w-px shrink-0 bg-line" />
}

/** Something worth saying about the state of things, but not an error. */
export function Note({ children }: { children: ReactNode }) {
  return <span className="text-xs text-warn">{children}</span>
}

/** A quiet aside: how to work the thing it sits beside. */
export function Hint({ children }: { children: ReactNode }) {
  return <span className="text-xs text-faint">{children}</span>
}

export function Spinner({ className = 'size-4' }: { className?: string }) {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 24 24"
      className={`animate-spin ${className}`}
      fill="none"
    >
      <circle
        cx="12"
        cy="12"
        r="9"
        stroke="currentColor"
        strokeOpacity="0.3"
        strokeWidth="3"
      />
      <path
        d="M21 12a9 9 0 0 0-9-9"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="round"
      />
    </svg>
  )
}
