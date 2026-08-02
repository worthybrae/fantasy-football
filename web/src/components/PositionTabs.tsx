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
          onClick={() => onChange(pos)}
        >
          {pos}
        </button>
      ))}
    </div>
  )
}
