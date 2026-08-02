import { useEffect, useState } from 'react'
import { fetchMeta } from '../api'

interface SourceMeta {
  source: string
  ok: boolean
  rows: number
  refreshed_at: string | null
}

// adp and schedules feed player ranking directly; a stale/failed pull for
// either one means the board itself is untrustworthy, so call it out.
const CRITICAL_SOURCES = new Set(['adp', 'schedules'])

export default function FreshnessBadge() {
  const [sources, setSources] = useState<SourceMeta[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchMeta()
      .then((data) => setSources(data.sources))
      .catch((e) => setError(e instanceof Error ? e.message : 'Failed to load source freshness'))
  }, [])

  const criticalFailures = sources.filter((s) => CRITICAL_SOURCES.has(s.source) && !s.ok)

  return (
    <div className="freshness">
      <div className="freshness-chips">
        {sources.map((s) => (
          <span
            key={s.source}
            className={`freshness-chip ${s.ok ? 'ok' : 'fail'}`}
            title={`${s.rows} rows · refreshed ${s.refreshed_at ?? 'never'}`}
          >
            {s.source}
          </span>
        ))}
      </div>
      {error && <div className="freshness-warning">{error}</div>}
      {criticalFailures.length > 0 && (
        <div className="freshness-warning">
          Warning: {criticalFailures.map((s) => s.source).join(', ')} failed to refresh — board
          data may be stale.
        </div>
      )}
    </div>
  )
}
