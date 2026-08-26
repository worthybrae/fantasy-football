import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { fetchManagerProfile, type ManagerProfile } from '../api'
import { Logo } from '../components/Logo'
import PlayerOverlay, { type OverlayTarget } from '../components/draft/PlayerOverlay'
import { useDocumentMeta } from '../lib/documentMeta'
import '../landing.css'
import '../market.css'
import './league.css'

// ONE MANAGER, OVER EVERY SEASON. /league/:leagueId/manager/:memberId.
//
// The sections mirror the facts document (scoring/manager_profile.py):
// finishes, playoffs, head-to-head, luck, draft, waivers, trades, lineups.
// Each carries the number of observations it rests on, and a figure that
// rests on fewer than five is shown with that number and no adjective --
// "63% waiver win rate (n=3)" is a fact; "wins the wire" off three claims
// is not.
//
// Player names open the same profile popup the draft room and the archive
// use (PlayerOverlay -> PlayerProfile); this page describes no player
// itself.
const SMALL = 5

function pct(v: number | null): string { return v === null ? '—' : `${Math.round(v * 100)}%` }
function num(v: number | null, digits = 1): string { return v === null ? '—' : v.toFixed(digits) }
// `digits` defaults to 2 because the fields this formats -- career and
// per-season luck chief among them -- are all-play win fractions straight
// off division (games * (wins / opponents)), not numbers anyone rounded for
// storage; printed raw they read "+0.3333333333333335". Call sites that
// carry their own rounding (trades' dollar-ish balances, an already-integer
// margin) pass a `digits` that matches what the value actually is.
function signed(v: number | null, digits = 2): string {
  if (v === null) return '—'
  const s = v.toFixed(digits)
  return v > 0 ? `+${s}` : s
}

function Fig({ label, value, n }: { label: string; value: string; n?: number }) {
  return (
    <div className="lg-fig">
      <span className="lg-fig-value mono">{value}</span>
      <span className="lg-fig-label">{label}{n !== undefined && n < SMALL && <span className="lg-n"> n={n}</span>}</span>
    </div>
  )
}

export default function ManagerPage() {
  const { leagueId = '', memberId = '' } = useParams()
  const [p, setP] = useState<ManagerProfile | null | undefined>(undefined)
  const [error, setError] = useState<string | null>(null)
  const [target, setTarget] = useState<OverlayTarget | null>(null)

  useDocumentMeta({ title: `${p?.display_name ?? 'Manager'} – ESPN Draft Assist`, noindex: true })

  useEffect(() => {
    let cancelled = false
    fetchManagerProfile(leagueId, memberId)
      .then((body) => { if (!cancelled) setP(body) })
      .catch((e) => { if (!cancelled) { setError(e instanceof Error ? e.message : String(e)); setP(null) } })
    return () => { cancelled = true }
  }, [leagueId, memberId])

  const openPlayer = (id: number) => setTarget({ playerId: String(id), seed: null })
  // `p.draft` is a loose bag (the league-report branch's tables, or absent
  // entirely) rather than a typed shape, so the count backing its three
  // rates -- mean value, steal rate, reach rate -- is read out once here,
  // the same defensive way each rate itself is, rather than three times
  // inline below.
  const gradedPicks = p?.draft && typeof p.draft.graded_picks === 'number' ? p.draft.graded_picks : undefined

  return (
    <div className="mk-page">
      <header className="draft-topbar">
        <Link to="/" className="draft-topbar-title"><Logo /> ESPN Draft Assist</Link>
        <span className="draft-topbar-sep" aria-hidden="true" />
        <nav className="draft-topbar-tabs" aria-label="Views">
          <Link className="draft-tab is-active" to="/" aria-current="page">Home</Link>
          <Link className="draft-tab" to="/archive">Data</Link>
          <Link className="draft-tab" to="/live">Live</Link>
        </nav>
        <span className="draft-topbar-sep" aria-hidden="true" />
        <span className="draft-topbar-league">{p ? `${p.seasons.length} seasons` : 'Manager'}</span>
        <span className="draft-topbar-spacer" />
        <Link className="draft-topbar-link" to={`/league/${encodeURIComponent(leagueId)}`}>← Back to the league</Link>
      </header>

      <main className="mk lg">
        {p === undefined ? (
          <p className="mk-loading">Reading the manager…</p>
        ) : p === null ? (
          <section className="mk-panel mk-gate">
            <h2 className="mk-h2">No profile here.</h2>
            <p className="mk-gate-copy">{error ?? 'That manager is not in this league.'}</p>
          </section>
        ) : (
          <>
            <header className="mk-mast">
              <div className="mk-mast-say">
                <h1 className="mk-title">{p.display_name}</h1>
                <p className="mk-lede">
                  {p.seasons.length} {p.seasons.length === 1 ? 'season' : 'seasons'} · {p.seasons[0]}–{p.seasons[p.seasons.length - 1]}
                  {p.finishes.titles.length > 0 && ` · ${p.finishes.titles.length} ${p.finishes.titles.length === 1 ? 'title' : 'titles'} (${p.finishes.titles.join(', ')})`}
                </p>
              </div>
              <div className="lg-figs">
                <Fig label="win rate" value={pct(p.finishes.win_pct)} n={p.finishes.n} />
                <Fig label="avg finish" value={num(p.finishes.avg_finish, 1)} n={p.finishes.n} />
                {/* Draft-day rank minus final rank, summed across seasons -- always a
                    whole number, so it gets no decimal places of its own. */}
                <Fig label="vs draft-day projection" value={signed(p.finishes.outperformance, 0)} n={p.finishes.n} />
              </div>
            </header>

            <section className="lg-sec">
              <h2 className="mk-h2">Finishes</h2>
              <table className="lg-table mono">
                <thead><tr><th>Season</th><th>Team</th><th>Record</th><th>Points</th><th>Seed</th><th>Finish</th><th>Projected</th></tr></thead>
                <tbody>
                  {p.finishes.seasons.map((s) => (
                    <tr key={s.season} className={s.final_rank === 1 ? 'is-title' : ''}>
                      <td>{s.season}</td><td className="lg-td-text">{s.team_name}</td>
                      <td>{s.wins ?? '—'}-{s.losses ?? '—'}{s.ties ? `-${s.ties}` : ''}</td>
                      <td>{num(s.points_for, 0)}</td><td>{s.playoff_seed ?? '—'}</td>
                      <td>{s.final_rank ?? '—'}</td><td>{s.draft_day_rank ?? '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>

            <section className="lg-sec">
              <h2 className="mk-h2">Playoffs</h2>
              <div className="lg-figs">
                <Fig label="appearances" value={String(p.playoffs.appearances.length)} />
                <Fig label="bracket record" value={`${p.playoffs.bracket_wins}-${p.playoffs.bracket_losses}`} n={p.playoffs.n} />
                <Fig label="titles" value={String(p.playoffs.titles.length)} />
                <Fig label="last place" value={p.playoffs.last_place.length ? p.playoffs.last_place.join(', ') : 'never'} />
              </div>
            </section>

            <section className="lg-sec">
              <h2 className="mk-h2">Head-to-head</h2>
              <table className="lg-table mono">
                <thead><tr><th>Opponent</th><th>Games</th><th>Record</th><th>Margin</th><th>Playoff</th></tr></thead>
                <tbody>
                  {p.head_to_head.map((h) => (
                    <tr key={h.opponent.member_id ?? ''}>
                      <td className="lg-td-text">
                        <Link to={`/league/${encodeURIComponent(leagueId)}/manager/${encodeURIComponent(h.opponent.member_id ?? '')}`}>{h.opponent.display_name}</Link>
                      </td>
                      <td>{h.games}</td><td>{h.wins}-{h.losses}{h.ties ? `-${h.ties}` : ''}</td>
                      {/* Already an integer by the time it reaches here (Math.round below),
                          so it is formatted with no decimal places. */}
                      <td className={h.margin > 0 ? 'is-up' : h.margin < 0 ? 'is-down' : ''}>{signed(Math.round(h.margin), 0)}</td>
                      <td>{h.playoff_games || '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>

            <section className="lg-sec">
              <h2 className="mk-h2">Luck</h2>
              <p className="mk-sub">Actual wins against the all-play expectation: your score against every team's, every week.</p>
              <div className="lg-figs">
                <Fig label="career luck (wins)" value={signed(p.luck.luck)} n={p.luck.n} />
                {p.luck.seasons.map((s) => (
                  <Fig key={s.season} label={`${s.season}: ${s.actual_wins} won, ${num(s.expected_wins, 2)} expected`} value={signed(s.luck)} />
                ))}
              </div>
            </section>

            <section className="lg-sec">
              <h2 className="mk-h2">Draft</h2>
              <div className="lg-figs">
                <Fig label="autodraft rate" value={pct(p.draft_flags.autodraft_rate)} n={p.draft_flags.n} />
                {p.draft && typeof p.draft.mean_value === 'number' && (
                  <Fig label="value per pick vs ADP" value={signed(p.draft.mean_value as number)} n={gradedPicks} />
                )}
                {p.draft && typeof p.draft.steal_rate === 'number' && (
                  <Fig label="steal rate" value={pct(p.draft.steal_rate as number)} n={gradedPicks} />
                )}
                {p.draft && typeof p.draft.reach_rate === 'number' && (
                  <Fig label="reach rate" value={pct(p.draft.reach_rate as number)} n={gradedPicks} />
                )}
              </div>
            </section>

            <section className="lg-sec">
              <h2 className="mk-h2">Waivers</h2>
              <div className="lg-figs">
                <Fig label="claims" value={String(p.waivers.claims)} />
                <Fig label="won" value={pct(p.waivers.win_rate)} n={p.waivers.won + p.waivers.lost} />
                <Fig label="free-agent adds" value={String(p.waivers.free_agent_adds)} />
                <Fig label="drops" value={String(p.waivers.drops)} />
                {p.waivers.bids && <Fig label="avg bid" value={`$${num(p.waivers.bids.mean, 1)}`} n={p.waivers.bids.n} />}
                {p.waivers.busiest_week && (
                  <Fig label="busiest week" value={`${p.waivers.busiest_week.season} wk ${p.waivers.busiest_week.week}`} />
                )}
              </div>
              <p className="mk-sub mono">
                {Object.entries(p.waivers.adds_by_weekday).map(([d, n]) => `${d} ${n}`).join(' · ')}
              </p>
            </section>

            <section className="lg-sec">
              <h2 className="mk-h2">Trades</h2>
              <div className="lg-figs">
                <Fig label="proposed" value={String(p.trades.proposed)} />
                <Fig label="received" value={String(p.trades.received)} />
                <Fig label="done" value={String(p.trades.accepted)} />
                <Fig label="declined (by me / by them)" value={`${p.trades.declined_by_me} / ${p.trades.declined_by_them}`} />
                <Fig label="rest-of-season balance" value={signed(p.trades.balance, 1)} n={p.trades.n} />
              </div>
              {p.trades.ledger.length > 0 && (
                <ul className="lg-ledger">
                  {p.trades.ledger.map((t, i) => (
                    <li key={i} className="lg-trade">
                      <span className="mono lg-trade-when">{t.season} wk {t.week}</span>
                      <span className="lg-trade-body">
                        sent {t.sent.map((x) => <button key={x.player_id} type="button" className="lg-player" onClick={() => openPlayer(x.player_id)}>{x.name}</button>)}
                        {' '}to{' '}
                        {t.with ? (
                          <Link to={`/league/${encodeURIComponent(leagueId)}/manager/${encodeURIComponent(t.with)}`}>{t.with_name}</Link>
                        ) : (
                          'another team'
                        )}
                        {' '}for {t.received.map((x) => <button key={x.player_id} type="button" className="lg-player" onClick={() => openPlayer(x.player_id)}>{x.name}</button>)}
                      </span>
                      <span className={`mono lg-trade-balance${t.balance > 0 ? ' is-up' : t.balance < 0 ? ' is-down' : ''}`}>{signed(t.balance, 1)}</span>
                    </li>
                  ))}
                </ul>
              )}
            </section>

            <section className="lg-sec">
              <h2 className="mk-h2">Lineups</h2>
              <div className="lg-figs">
                <Fig label="bench points left / week" value={num(p.lineups.bench_points_left)} n={p.lineups.n} />
                <Fig label="optimal lineup rate" value={pct(p.lineups.hit_rate)} n={p.lineups.n} />
                <Fig label="started an OUT player" value={String(p.lineups.started_out)} />
                <Fig label="lineup moves / week" value={num(p.lineups.moves_per_week)} />
                {p.lineups.worst_week && (
                  <Fig label="worst week" value={`${p.lineups.worst_week.season} wk ${p.lineups.worst_week.week}: ${num(p.lineups.worst_week.left, 1)} left`} />
                )}
              </div>
            </section>
          </>
        )}
      </main>
      {target && <PlayerOverlay target={target} onClose={() => setTarget(null)} onSelectPlayer={() => {}} />}
    </div>
  )
}
