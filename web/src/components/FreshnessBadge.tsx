import { useEffect, useState } from 'react'
import { fetchMeta } from '../api'

interface SourceMeta {
  source: string
  ok: boolean
  rows: number
  refreshed_at: string | null
}

// Kept small and deliberate -- not "any source failing is critical":
//   - schedules feeds environment/strength-of-schedule scoring directly, so
//     a failed pull skews every player's composite/VOR, not just ranking.
//   - adp (FFC) is how rookies and K/DST enter the board at all (see
//     scoring/board.py: _add_adp_only_players) -- a failed pull doesn't just
//     drop one of three market-consensus inputs, it can make whole players
//     vanish from the board.
// espn_adp and fp_ecr, by contrast, are now just 2-of-3 market-consensus
// inputs each; either one failing degrades the Mkt/edge columns but leaves
// the board itself (roster + VOR ranking) intact, so neither is in this set.
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
            title={`${s.source}: ${s.rows} rows · refreshed ${s.refreshed_at ?? 'never'}`}
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
