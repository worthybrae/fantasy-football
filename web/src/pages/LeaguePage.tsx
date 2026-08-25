import { useCallback, useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { fetchUpcomingDrafts, mintDraftToken, type UpcomingDraft } from '../api'
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

            {/* WHAT IS COMING HERE, said plainly rather than left as an empty
                page. Each of these is a real piece of work with a place
                reserved for it; the list is the page's own roadmap and it
                shrinks as they land. */}
            <section className="lg-soon">
              <h2 className="mk-h2">This league, over time</h2>
              <p className="mk-sub">
                Past seasons, draft report cards and player profiles for this
                league live here as they are built -- with waiver-wire help and
                trade suggestions to follow.
              </p>
            </section>
          </>
        )}
      </main>
    </div>
  )
}
