import { useCallback, useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import {
  fetchLeagueHistory, fetchLeagueHistoryProgress, fetchUpcomingDrafts, mintDraftToken, startLeagueHistory,
  type HistoryProgress, type LeagueHistory, type UpcomingDraft,
} from '../api'
import { Logo } from '../components/Logo'
import { calendarLabel, countdownTo, secondsUntil } from '../lib/countdown'
import { useDocumentMeta } from '../lib/documentMeta'
import '../landing.css'
import '../market.css'
import './league.css'

// ONE LEAGUE, AS A PLACE. /league/:leagueId.
//
// The Home page lists your leagues as cards with a locked door on each --
// ESPN only opens a draft room when the draft starts. This is where the
// name on that card goes: the league itself, which is open at any time.
// Today it holds what the account's league list already knows -- the
// draft's date and shape, your team, the way in once the room opens. It is
// built to grow: seasons past, draft report cards, player profiles, and
// whatever comes after them land on this page, not on the Home grid.
//
// Read from the same `/api/espn/drafts` answer the dashboard draws, so a
// league here is a league there and nothing is fetched twice for the
// crossing. The join is the exact door the waiting room uses: mint, then
// `/` with the token in the hash, which Landing already knows how to open.

function useNow(fast: boolean): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), fast ? 1000 : 20_000)
    return () => clearInterval(id)
  }, [fast])
  return now
}

export default function LeaguePage() {
  const navigate = useNavigate()
  const { leagueId } = useParams()
  const [league, setLeague] = useState<UpcomingDraft | null | undefined>(undefined)
  const [connected, setConnected] = useState<boolean | null>(null)
  const [joining, setJoining] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useDocumentMeta({
    title: `${league?.name ?? 'League'} – ESPN Draft Assist`,
    noindex: true,
  })

  useEffect(() => {
    let cancelled = false
    fetchUpcomingDrafts().then((body) => {
      if (cancelled) return
      setConnected(body.connected)
      setLeague(body.leagues.find((l) => l.league_id === leagueId) ?? null)
    }).catch((e) => {
      if (cancelled) return
      setError(e instanceof Error ? e.message : String(e))
      setLeague(null)
    })
    return () => { cancelled = true }
  }, [leagueId])

  const seconds = league ? secondsUntil(league.draft_at, Date.now()) : null
  const now = useNow(seconds !== null && seconds < 3600)
  const count = league ? countdownTo(secondsUntil(league.draft_at, now)) : null

  const enter = useCallback(async () => {
    if (!league?.team_id) return
    setJoining(true)
    setError(null)
    try {
      const params = await mintDraftToken(league.league_id, league.team_id, league.season)
      const q = new URLSearchParams({
        leagueId: params.leagueId, teamId: params.teamId,
        swid: params.swid, token: params.token,
      })
      if (params.season) q.set('season', params.season)
      navigate({ pathname: '/', hash: q.toString() })
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setJoining(false)
    }
  }, [league, navigate])

  const shape = league
    ? [league.teams ? `${league.teams} teams` : null, league.draft_type?.toLowerCase()]
        .filter(Boolean).join(' · ')
    : ''

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
        <span className="draft-topbar-league">
          {league === undefined ? 'Reading the league…' : league === null ? 'League' : shape}
        </span>
        <span className="draft-topbar-spacer" />
        {league?.live && (
          <span className="draft-status-pill draft-status-pill-ok">
            <span className="draft-status-dot" aria-hidden="true" />
            DRAFTING
          </span>
        )}
      </header>

      <main className="mk lg">
        {league === undefined ? (
          <p className="mk-loading">Reading the league…</p>
        ) : league === null ? (
          <section className="mk-panel mk-gate">
            <h2 className="mk-h2">
              {connected === false
                ? 'This page needs a connected ESPN account.'
                : 'That league is not on this account.'}
            </h2>
            <p className="mk-gate-copy">
              {error ?? 'A league shows here once it is on the ESPN account this browser has connected.'}
            </p>
            <Link className="lg-back" to="/">← Back to Home</Link>
          </section>
        ) : (
          <>
            <header className="mk-mast">
              <div className="mk-mast-say">
                <h1 className="mk-title">{league.name ?? `League ${league.league_id}`}</h1>
                <p className="mk-lede">
                  {league.team_name ?? 'Your team'}
                  {league.team_name && shape && ' · '}
                  {shape}
                </p>
              </div>
              <div className="lg-when">
                {league.live ? (
                  <>
                    <span className="lg-when-label"><span className="db-dot" aria-hidden="true" />Drafting now</span>
                    <button
                      type="button"
                      className="db-go db-league-go"
                      onClick={enter}
                      disabled={joining || !league.team_id}
                    >
                      {joining ? 'Joining…' : 'Enter the room'}
                      <span className="db-league-go-arrow" aria-hidden="true">→</span>
                    </button>
                  </>
                ) : (
                  <>
                    <span className={`mono lg-when-count${count?.imminent ? ' is-soon' : ''}`}>
                      {count?.text ?? 'No date'}
                    </span>
                    <span className="lg-when-label">
                      {league.draft_at ? `Draft ${calendarLabel(league.draft_at)}` : 'No draft date set'}
                    </span>
                  </>
                )}
              </div>
            </header>

            {error !== null && <p className="db-error">{error}</p>}

            {/* `me` is the visiting member's SWID; the league list this page
                is built from does not carry it yet, so nobody is marked as
                "you" among the managers below until /api/espn/drafts grows a
                member_id -- a follow-up, not a gap in this section. */}
            <LeagueHistorySection leagueId={league.league_id} me={null} />
          </>
        )}
      </main>
    </div>
  )
}

// THE LEAGUE'S HISTORY, IMPORTED ON THE FIRST VISIT. The page asks for the
// overview; a 404 means nobody has read this league yet, so it starts the
// import with the visitor's own session and draws the stage list while
// ESPN is read -- one row per season, the current one ticking week by
// week -- then the overview the moment the job says done.
const POLL_MS = 2000

function LeagueHistorySection({ leagueId, me }: { leagueId: string; me: string | null }) {
  const [history, setHistory] = useState<LeagueHistory | null | undefined>(undefined)
  const [progress, setProgress] = useState<HistoryProgress | null>(null)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      const body = await fetchLeagueHistory(leagueId)
      setHistory(body)
      return body
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setHistory(null)
      return null
    }
  }, [leagueId])

  const start = useCallback(async () => {
    setError(null)
    try {
      const status = await startLeagueHistory(leagueId)
      if (status === 'fresh') { await load(); return }
      setProgress({ phase: 'running', stages: [] })
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [leagueId, load])

  useEffect(() => {
    let cancelled = false
    load().then((body) => { if (!cancelled && body === null) start() })
    return () => { cancelled = true }
  }, [load, start])

  // Poll while running; on done, fetch the overview and stop.
  useEffect(() => {
    if (progress?.phase !== 'running') return
    let cancelled = false
    const id = setInterval(async () => {
      try {
        const next = await fetchLeagueHistoryProgress(leagueId)
        if (cancelled) return
        setProgress(next)
        if (next.phase === 'done') { clearInterval(id); await load() }
        if (next.phase === 'failed') clearInterval(id)
      } catch { /* the next tick asks again */ }
    }, POLL_MS)
    return () => { cancelled = true; clearInterval(id) }
  }, [progress?.phase, leagueId, load])

  if (history === undefined && progress === null && error === null) {
    return <p className="mk-loading">Reading the league…</p>
  }
  if (history) return <History history={history} leagueId={leagueId} me={me} />
  if (progress?.phase === 'failed' || error) {
    return (
      <section className="lg-import">
        <h2 className="mk-h2">The history could not be read.</h2>
        <p className="mk-sub">{progress?.error ?? error}</p>
        <button type="button" className="db-join" onClick={start}>Try again</button>
      </section>
    )
  }
  return (
    <section className="lg-import" aria-live="polite">
      <h2 className="mk-h2">Reading this league's history from ESPN…</h2>
      <p className="mk-sub">Every season, every week: standings, matchups, waivers, trades, lineups. A minute or two.</p>
      <ul className="lg-stages mono">
        {(progress?.stages ?? []).map((s) => (
          <li key={s.season} className={`lg-stage${s.done ? ' is-done' : s.week > 0 ? ' is-live' : ''}`}>
            <span className="lg-stage-dot" aria-hidden="true" />{s.label}
          </li>
        ))}
      </ul>
    </section>
  )
}

function History({ history, leagueId, me }: { history: LeagueHistory; leagueId: string; me: string | null }) {
  return (
    <>
      <section className="lg-seasons">
        <h2 className="mk-h2">Seasons</h2>
        <ol className="lg-strip">
          {history.seasons.map((s) => (
            <li className={`lg-season${s.complete ? '' : ' is-open'}`} key={s.season}>
              <span className="mono lg-season-year">{s.season}</span>
              <Row label="Champion" who={s.champion} leagueId={leagueId} crown />
              <Row label="Last" who={s.last} leagueId={leagueId} />
              <Row label="Top scorer" who={s.top_scorer} leagueId={leagueId} />
            </li>
          ))}
        </ol>
      </section>
      <section className="lg-managers">
        <h2 className="mk-h2">Managers</h2>
        <ul className="lg-grid">
          {history.members.map((m) => (
            <li className={`lg-manager${m.member_id === me ? ' is-me' : ''}`} key={m.member_id}>
              <Link to={`/league/${encodeURIComponent(leagueId)}/manager/${encodeURIComponent(m.member_id)}`} className="lg-manager-name">
                {m.display_name}
              </Link>
              <p className="lg-manager-line">{m.defining_line}</p>
              <p className="mono lg-manager-figs">
                {m.seasons} {m.seasons === 1 ? 'season' : 'seasons'}
                {m.titles > 0 && ` · ${m.titles} ${m.titles === 1 ? 'title' : 'titles'}`}
                {m.avg_finish !== null && ` · avg finish ${m.avg_finish}`}
                {m.win_pct !== null && ` · ${Math.round(m.win_pct * 100)}% wins`}
              </p>
            </li>
          ))}
        </ul>
      </section>
    </>
  )
}

function Row({ label, who, leagueId, crown = false }: {
  label: string; who: { member_id: string | null; display_name: string | null } | null
  leagueId: string; crown?: boolean
}) {
  if (!who || !who.member_id) return <span className="lg-season-row is-empty">{label}: —</span>
  return (
    <span className="lg-season-row">
      <span className="lg-season-label">{crown ? '♛ ' : ''}{label}</span>
      <Link to={`/league/${encodeURIComponent(leagueId)}/manager/${encodeURIComponent(who.member_id)}`}>
        {who.display_name}
      </Link>
    </span>
  )
}
