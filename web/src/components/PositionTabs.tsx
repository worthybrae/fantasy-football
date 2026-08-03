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
            // clicking a tab, instead of this button holding focus.
            e.currentTarget.blur()
          }}
        >
          {pos}
        </button>
      ))}
    </div>
  )
}
