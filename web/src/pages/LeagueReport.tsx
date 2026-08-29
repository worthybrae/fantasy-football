import { useEffect, useState } from 'react'
import { Link, useLocation, useParams } from 'react-router-dom'
import {
  fetchLeagueReport, type LeagueReport, type ReportCard, type ReportPick, type TeamProfile,
} from '../api'
import { useDocumentMeta } from '../lib/documentMeta'
import { Logo } from '../components/Logo'
import { ordinal } from '../components/profile/payload'
import '../league.css'

// A report is built once and stored; this page only reads it. While the
// owner's build is in flight the GET answers 404, so a page opened with
// `#building` (the dashboard sends it there right after the POST) polls
// for two minutes before giving up.
const POLL_MS = 3000
const POLL_LIMIT = 40

const VERDICT_WORD: Record<NonNullable<ReportPick['verdict']>, string> = {
  steal: 'Steal', value: 'Value', market: 'On the market', early: 'Early', reach: 'Reach',
}

function tone(verdict: ReportPick['verdict']): string {
  if (verdict === 'steal' || verdict === 'value') return 'is-steal'
  if (verdict === 'reach' || verdict === 'early') return 'is-reach'
  return 'is-flat'
}

function gap(pick: ReportPick): string {
  if (pick.value === null) return 'no ADP'
  const n = Math.round(Math.abs(pick.value))
  return pick.value > 0 ? `${n} past ADP` : pick.value < 0 ? `${n} before ADP` : 'at ADP'
}

function PickLine({ label, pick }: { label: string; pick: ReportPick | null }) {
  if (!pick) return null
  return (
    <p className="lr-pick">
      <span className="lr-pick-label">{label}</span>
      <span className="lr-pick-name">{pick.player_name}</span>
      {pick.position && <span className={`lr-pos lr-pos-${pick.position.toLowerCase()}`}>{pick.position}</span>}
      <span className="mono lr-pick-where">R{pick.round ?? '?'} · #{pick.overall_pick}</span>
      <span className={`lr-pick-verdict ${tone(pick.verdict)}`}>
        {pick.verdict ? `${VERDICT_WORD[pick.verdict]} · ${gap(pick)}` : 'No ADP'}
      </span>
    </p>
  )
}

function Shape({ card }: { card: ReportCard }) {
  const order = ['QB', 'RB', 'WR', 'TE', 'K', 'DST']
  const entries = order.filter((p) => card.shape.positions[p]).map((p) => [p, card.shape.positions[p]] as const)
  if (entries.length === 0) return null
  return (
    <p className="lr-shape">
      {entries.map(([pos, n]) => (
        <span key={pos} className={`lr-shape-chip lr-pos-${pos.toLowerCase()}`}>{n} {pos}</span>
      ))}
    </p>
  )
}

function Card({ card }: { card: ReportCard }) {
  return (
    <li className={`lr-card lr-grade-${card.grade.toLowerCase()}`}>
      <div className="lr-card-head">
        <span className="lr-grade" aria-label={`Grade ${card.grade}`}>{card.grade}</span>
        <div className="lr-card-who">
          <p className="lr-card-team">{card.team_name}</p>
          <p className="lr-card-manager">
            {card.manager}
            {card.nickname && <span className="lr-nickname"> · “{card.nickname}”</span>}
          </p>
        </div>
        <p className="mono lr-card-stat">
          <span className="is-steal">{card.steals} steal{card.steals === 1 ? '' : 's'}</span>
          <span className="lr-sep">·</span>
          <span className="is-reach">{card.reaches} reach{card.reaches === 1 ? '' : 'es'}</span>
        </p>
      </div>
      {card.blurb && <p className="lr-blurb">{card.blurb}</p>}
      <PickLine label="Best" pick={card.best_pick} />
      <PickLine label="Worst" pick={card.worst_pick} />
      <Shape card={card} />
    </li>
  )
}

function record(p: TeamProfile['seasons'][number]): string {
  if (p.wins === null || p.losses === null) return '—'
  const base = `${p.wins}-${p.losses}${p.ties ? `-${p.ties}` : ''}`
  return p.final_rank ? `${base} · ${ordinal(p.final_rank)}` : base
}

function habits(p: TeamProfile): string {
  const bits: string[] = []
  const opens = Object.entries(p.first_pick_positions).sort((a, b) => b[1] - a[1])[0]
  if (opens) bits.push(`opens ${opens[0]}`)
  if (p.steal_rate !== null && p.steal_rate >= 0.2) bits.push('finds steals')
  if (p.reach_rate !== null && p.reach_rate >= 0.2) bits.push('reaches')
  if (p.career_best) bits.push(`best pick ever: ${p.career_best.player_name} (R${p.career_best.round ?? '?'})`)
  return bits.join(' · ')
}

function History({ profile }: { profile: TeamProfile }) {
  return (
    <li className="lr-history">
      <div className="lr-history-head">
        <p className="lr-history-team">{profile.team_name}</p>
        <p className="lr-history-manager">{profile.manager}</p>
        <p className="mono lr-history-pct">
          {profile.win_pct === null ? 'first season' : `${Math.round(profile.win_pct * 100)}% career`}
        </p>
      </div>
      {profile.seasons.length > 0 && (
        <ul className="lr-seasons">
          {profile.seasons.map((s) => (
            <li key={s.season} className={`lr-season${s.final_rank === 1 ? ' is-title' : ''}`}>
              <span className="mono lr-season-year">{s.season}</span>
              <span className="mono lr-season-rec">{record(s)}</span>
              {s.final_rank === 1 && <span className="lr-crown" aria-label="champion">♛</span>}
            </li>
          ))}
        </ul>
      )}
      {habits(profile) && <p className="lr-habits">{habits(profile)}</p>}
    </li>
  )
}

export default function LeagueReport() {
  const { leagueId = '', season = '' } = useParams()
  const location = useLocation()
  const [report, setReport] = useState<LeagueReport | null | undefined>(undefined)
  const [error, setError] = useState<string | null>(null)
  const [polls, setPolls] = useState(0)
  const building = report === null && location.hash === '#building' && polls < POLL_LIMIT

  useDocumentMeta({
    title: report ? `${report.league_name} ${report.season} draft report card` : 'Draft report card',
    description: 'Power rankings, a grade for every team and how each manager has drafted and finished over the years.',
  })

  useEffect(() => {
    let cancelled = false
    fetchLeagueReport(leagueId, season)
      .then((body) => { if (!cancelled) { setReport(body); setError(null) } })
      .catch((err) => { if (!cancelled) setError(String(err.message || err)) })
    return () => { cancelled = true }
  }, [leagueId, season, polls])

  useEffect(() => {
    if (!building) return
    const id = window.setTimeout(() => setPolls((n) => n + 1), POLL_MS)
    return () => window.clearTimeout(id)
  }, [building, polls])

  return (
    <div className="lr-page">
      <header className="lr-bar">
        <Link to="/" className="lr-logo" aria-label="Home"><Logo /></Link>
      </header>
      {error && <p className="lr-note is-fail">{error}</p>}
      {report === undefined && !error && (
        <p className="lr-note is-waiting">Looking for this season's report…</p>
      )}
      {report === null && building && (
        <p className="lr-note">Writing the report card… this takes a few seconds.</p>
      )}
      {report === null && !building && (
        <p className="lr-note">No report for this season.</p>
      )}
      {report && report.status === 'failed' && (
        <p className="lr-note is-fail">This report could not be built: {report.reason}</p>
      )}
      {report && report.status !== 'failed' && (
        <main className="lr-main">
          <section className="lr-hero">
            <p className="lr-kicker">{report.league_name} · {report.season}</p>
            <h1 className="lr-title">Draft report card</h1>
            {report.intro && <p className="lr-intro">{report.intro}</p>}
          </section>

          <section className="lr-section">
            <h2 className="lr-h2">Power rankings</h2>
            <ol className="lr-ranks">
              {report.power_rankings.map((r) => (
                <li key={r.manager} className="lr-rank">
                  <span className="mono lr-rank-n">{r.rank}</span>
                  <div className="lr-rank-who">
                    <p className="lr-rank-team">{r.team_name}</p>
                    <p className="lr-rank-manager">{r.manager}{r.first_year && <span className="lr-first"> · first draft</span>}</p>
                    {r.line && <p className="lr-rank-line">{r.line}</p>}
                  </div>
                </li>
              ))}
            </ol>
          </section>

          <section className="lr-section">
            <h2 className="lr-h2">Report cards</h2>
            <ul className="lr-cards">
              {report.report_cards.map((c) => <Card key={c.manager} card={c} />)}
            </ul>
          </section>

          <section className="lr-section">
            <h2 className="lr-h2">The teams over the years</h2>
            <ul className="lr-histories">
              {report.profiles.map((p) => <History key={p.manager} profile={p} />)}
            </ul>
          </section>

          <footer className="lr-foot mono">
            {report.generated_at && <span>Generated {report.generated_at.slice(0, 10)}</span>}
            {report.status === 'numbers_only' && <span> · numbers only, no write-up</span>}
          </footer>
        </main>
      )}
    </div>
  )
}
