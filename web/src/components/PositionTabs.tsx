const POSITIONS = ['ALL', 'QB', 'RB', 'WR', 'TE', 'FLEX', 'K', 'DST'] as const

interface PositionTabsProps {
  value: string
  onChange: (v: string) => void
}

export default function PositionTabs({ value, onChange }: PositionTabsProps) {
  return (
    <div className="position-tabs">
      {POSITIONS.map((pos) => (
        <button
          key={pos}
          type="button"
          className={pos === value ? 'active' : undefined}
          onClick={(e) => {
            onChange(pos)
            // Keep board keyboard nav (↑/↓/Enter/D) live immediately after
            // a mouse click on a tab, instead of this button holding focus.
            // Pointer-only (`detail` is 0 for a keyboard-synthesized click)
            // so Tab order is preserved for keyboard users -- they keep
            // focus here and are covered by App's Enter/D activeElement
            // guard instead.
            if (e.detail > 0) e.currentTarget.blur()
          }}
        >
          {pos}
        </button>
      ))}
    </div>
  )
}
