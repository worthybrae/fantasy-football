import { useEffect, useState } from 'react'
import { ageLabel, fetchLandingStatus, type LandingStatus } from '../api'

// Draft night is the one night this has to work, and every way it can fail is
// something that happened (or didn't) days earlier: a refresh that never ran,
// a league never imported, models never fitted. This strip is where you find
// that out while there is still time to fix it -- so every "not ready" row
// names the command that fixes it rather than just reporting the gap.

interface Row {
  label: string
  value: string
  /** false renders the hollow dot: not broken, just not done yet. */
  ready: boolean
  /** Shown only when not ready. The fix, not an apology. */
  fix?: string
}

/** Most recent successful refresh across all sources -- the strip reports the
 *  board's age as a whole, and the board is only as fresh as its last pull. */
function freshest(sources: LandingStatus['sources']): string | null {
  const stamps = sources
    .filter((s) => s.refreshed_at)
    .map((s) => s.refreshed_at as string)
    .sort()
  return stamps.length ? stamps[stamps.length - 1] : null
}

function rowsFor(status: LandingStatus): Row[] {
  const failed = status.sources.filter((s) => !s.ok).map((s) => s.source)
  const last = freshest(status.sources)
  const { history, managers, league, sim } = status

  return [
    {
      label: 'data',
      value: last
        ? `refreshed ${ageLabel(last)}${failed.length ? ` · ${failed.join(', ')} failed` : ''}`
        : 'never refreshed',
      ready: last !== null && failed.length === 0,
      fix: 'make refresh',
    },
    {
      label: 'league',
      value: league.derived
        ? `${league.teams} teams · ${league.rounds} rounds · from ESPN`
        : `${league.teams} teams · ${league.rounds} rounds · default shape`,
      ready: league.derived,
      fix: 'make espn-import LEAGUE=<url>',
    },
    {
      label: 'history',
      value: history.picks
        ? `${history.picks} picks · ${history.seasons.length} seasons · ${history.teams} managers`
        : 'nothing imported',
      ready: history.picks > 0,
      fix: 'make espn-import LEAGUE=<url>',
    },
    {
      label: 'managers',
      value: managers.fitted
        ? `${managers.fitted} fitted · ${managers.personal} on personal models`
        : 'not fitted — the sim will use the league-average model',
      ready: managers.fitted > 0,
      fix: 'make fit-managers',
    },
    {
      label: 'sim',
      value: sim
        ? `slot ${sim.my_slot ?? '—'} · ${ageLabel(sim.created_at)}`
        : 'never run',
      ready: sim !== null,
      fix: 'make sim SLOT=<n>',
    },
  ]
}

export default function ReadinessStrip() {
  const [status, setStatus] = useState<LandingStatus | null>(null)

  useEffect(() => {
    let cancelled = false
    fetchLandingStatus()
      .then((s) => { if (!cancelled) setStatus(s) })
      // No error state: a landing page reached without the helper running is
      // a page whose job is still to hand over the bookmarklet. A red banner
      // about a status endpoint would be answering a question nobody asked.
      .catch(() => undefined)
    return () => { cancelled = true }
  }, [])

  if (!status) return null

  const rows = rowsFor(status)
  const blocked = rows.filter((r) => !r.ready).length

  return (
    <section className="readiness" aria-labelledby="readiness-heading">
      <h2 className="readiness-heading" id="readiness-heading">
        Ready for draft night
        <span className="readiness-count mono">
          {blocked === 0 ? 'all set' : `${blocked} to do`}
        </span>
      </h2>
      <dl className="readiness-rows">
        {rows.map((row) => (
          <div className={row.ready ? 'readiness-row is-ready' : 'readiness-row'} key={row.label}>
            <dt className="readiness-label mono">
              <span className="readiness-dot" aria-hidden="true" />
              {row.label}
            </dt>
            <dd className="readiness-value">
              {row.value}
              {!row.ready && row.fix && <code className="readiness-fix">{row.fix}</code>}
            </dd>
          </div>
        ))}
      </dl>
    </section>
  )
}
