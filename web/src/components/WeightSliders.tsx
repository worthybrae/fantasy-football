import type { Weights } from '../api'

interface WeightSlidersProps {
  weights: Weights
  onChange: (w: Weights) => void
}

const LABELS: Record<keyof Weights, string> = {
  production: 'Production',
  role: 'Role',
  environment: 'Environment',
  schedule: 'Schedule',
  durability: 'Durability',
}

export default function WeightSliders({ weights, onChange }: WeightSlidersProps) {
  function handleChange(key: keyof Weights, value: number) {
    onChange({ ...weights, [key]: value })
  }

  return (
    <div className="weight-sliders">
      <h2>Weights</h2>
      {(Object.keys(LABELS) as (keyof Weights)[]).map((key) => (
        <div className="weight-slider" key={key}>
          <label htmlFor={`weight-${key}`}>
            <span>{LABELS[key]}</span>
            <span className="mono">{weights[key].toFixed(2)}</span>
          </label>
          <input
            id={`weight-${key}`}
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={weights[key]}
            onChange={(e) => handleChange(key, Number(e.target.value))}
          />
        </div>
      ))}
    </div>
  )
}
