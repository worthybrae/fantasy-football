interface FactorBarsProps {
  factors: {
    production: number
    durability: number
    role: number
    environment: number
    schedule: number
  }
}

const FACTOR_LABELS: { key: keyof FactorBarsProps['factors']; label: string }[] = [
  { key: 'production', label: 'Production' },
  { key: 'durability', label: 'Durability' },
  { key: 'role', label: 'Role' },
  { key: 'environment', label: 'Environment' },
  { key: 'schedule', label: 'Schedule' },
]

export default function FactorBars({ factors }: FactorBarsProps) {
  return (
    <div className="factor-bars">
      {FACTOR_LABELS.map(({ key, label }) => {
        const value = Math.max(0, Math.min(100, factors[key]))
        return (
          <div className="factor-bar" key={key}>
            <span className="factor-bar-label">{label}</span>
            <div className="factor-bar-track">
              <div className="factor-bar-fill" style={{ width: `${value}%` }} />
            </div>
            <span className="factor-bar-value mono">{value.toFixed(1)}</span>
          </div>
        )
      })}
    </div>
  )
}
