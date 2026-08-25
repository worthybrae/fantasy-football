import { useCallback, useEffect, useState } from 'react'
import { fetchUpcomingDrafts, mintDraftToken,
         type TokenConnectParams, type UpcomingDraft } from '../api'

// The leagues this account is in, and a button that walks into one.
//
// WHAT THIS REPLACES. The bookmarklet is a good answer to a real constraint
// -- ESPN mints a draft token only for a request carrying the account session
// cookie, and only a page on ESPN's own origin has one. But it makes the user
// the transport: find the bookmark, be on the right ESPN tab, click it, and
// click it AGAIN when the token expires mid-draft. When the helper already
// holds a session it can make that call itself, and the whole ceremony
// collapses into this card.
//
// THE PANEL IS ALSO THE SIGNED-IN CHECK, and deliberately the only one. There
// is a `/api/espn/custody` probe that answers "is there a stored session", but
// a stored session is not the same claim as a WORKING one: ESPN invalidates a
// cookie when the user logs out anywhere, and nothing local can see that
// happen. This endpoint finds out the only way anybody can -- it asks ESPN --
// so a card that renders leagues is a card whose session was good a moment
// ago. Trusting the cheaper probe would put a green "connected" banner over
// an empty list, which is the one state that would make a user doubt the
// board rather than their login.
//
// It draws NOTHING while it does not know, and nothing when there is no
// session. A visitor who has never connected anything must not be shown an
// empty frame captioned with an error about a feature they were not using --
// the bookmarklet instructions below are their path, and this card silently
// gets out of their way.

// How often the list is re-read while the page sits open. The server caches
// for two minutes, so this mostly costs nothing -- what it buys is a draft
// that flips to LIVE while somebody is looking at the page, which is the one
// moment the card most needs to be right.
const POLL_MS = 60_000

function whenLabel(at: string | null, live: boolean): string {
  if (live) return 'drafting now'
  if (at === null) return 'no date set'
  const when = new Date(at)
  const mins = Math.round((when.getTime() - Date.now()) / 60_000)
  // Under a day, a countdown is what a drafter is actually reading for; past
  // that, the day and time is more use than "in 73 hours".
  if (mins >= 0 && mins < 60) return `in ${mins} min`
  if (mins >= 60 && mins < 60 * 24) return `in ${Math.round(mins / 60)} h`
  return when.toLocaleString(undefined, {
    weekday: 'short', month: 'short', day: 'numeric',
    hour: 'numeric', minute: '2-digit',
  })
}

export default function UpcomingDrafts({ onJoin }: {
  /** Hands the caller exactly what the bookmarklet's hash carries, so the
   *  page runs its existing connect rather than a second copy of it. */
  onJoin: (params: TokenConnectParams) => void
}) {
  const [leagues, setLeagues] = useState<UpcomingDraft[] | null>(null)
  const [source, setSource] = useState<string | null>(null)
  const [joining, setJoining] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    const read = async () => {
      try {
        const body = await fetchUpcomingDrafts()
        if (cancelled) return
        // `connected: false` covers both "never connected" and "the session
        // ESPN just rejected". Either way there is nothing to draw, and the
        // server has already dropped whatever it was holding.
        setLeagues(body.connected ? body.leagues : [])
        setSource(body.connected ? (body.source ?? null) : null)
      } catch {
        // The helper being down is not this card's story to tell -- the page
        // around it has a band for that. Stay quiet.
        if (!cancelled) setLeagues([])
      }
    }
    read()
    const id = setInterval(read, POLL_MS)
    return () => { cancelled = true; clearInterval(id) }
  }, [])

  const join = useCallback(async (league: UpcomingDraft) => {
    if (!league.team_id) return
    setJoining(league.league_id)
    setError(null)
    try {
      // Two calls, and the split is the point: this one mints, the page's own
      // connect opens the socket. A token minted here and handed over is the
      // same token the bookmarklet would have delivered, so everything
      // downstream -- the progress screen, the retry, the board -- is the
      // path that already exists and is already tested.
      onJoin(await mintDraftToken(league.league_id, league.team_id, league.season))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setJoining(null)
    }
  }, [onJoin])

  if (leagues === null || leagues.length === 0) return null

  return (
    <section className="lp-drafts">
      <div className="lp-drafts-head">
        <p className="lp-cap">Your ESPN leagues</p>
        <span className="lp-drafts-source mono">
          {source === 'local' ? 'this machine’s ESPN login' : 'connected account'}
        </span>
      </div>
      <ul className="lp-drafts-list">
        {leagues.map((league) => (
          <li className={`lp-draft${league.live ? ' is-live' : ''}`} key={league.league_id}>
            <span className="lp-draft-main">
              <span className="lp-draft-name">{league.name ?? `League ${league.league_id}`}</span>
              <span className="lp-draft-meta mono">
                {[league.team_name,
                  league.teams ? `${league.teams} teams` : null,
                  league.draft_type]
                  .filter(Boolean).join(' · ')}
              </span>
            </span>
            <span className={`lp-draft-when mono${league.live ? ' is-live' : ''}`}>
              {whenLabel(league.draft_at, league.live)}
            </span>
            <button
              type="button"
              className="lp-draft-join"
              onClick={() => join(league)}
              disabled={joining !== null || !league.team_id}
            >
              {joining === league.league_id ? 'Joining…' : 'Join'}
            </button>
          </li>
        ))}
      </ul>
      {/* The one failure worth a line here: ESPN refused to mint. Everything
          else -- a dead session, a helper that is down -- has already been
          turned into "there is nothing to show" above. */}
      {error !== null && <p className="lp-drafts-error">{error}</p>}
    </section>
  )
}
