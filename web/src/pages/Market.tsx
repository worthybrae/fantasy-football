import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
  fetchMarketOverview, fetchMarketSlot, fetchUpcomingDrafts,
  type MarketOverview, type MarketSlot,
  type MarketTurn, type MarketTurnPlayer,
  type UpcomingDrafts,
} from '../api'
import { readAccount, rememberAccount } from '../lib/accountCache'
import { useDocumentMeta } from '../lib/documentMeta'
import { Logo } from '../components/Logo'
import PickOption from '../components/PickOption'
import type { ProfileSeed } from '../components/PlayerProfile'
import PlayerOverlay, { type OverlayTarget } from '../components/draft/PlayerOverlay'
import { SEASON_GAMES } from '../components/draft/weeks'
import '../market.css'

// THE ARCHIVE, READ FROM ONE SEAT.
//
// The corpus is hundreds of complete ESPN drafts made by real people, and the
// question it is uniquely able to answer is not "what is this player's ADP"
// -- everybody publishes an average -- but "from seat 6, what does the room
// actually do, and who is still there when my turn comes round". So the seat
// is the page's one control, and everything below it is that seat's own
// draft: its sixteen turns in order, the position paths people walk from it,
// and every player's distribution with THIS SEAT'S picks drawn on top.
//
// WHY THE BAR IS THE WHOLE VISUALISATION. Each turn shows who gets taken
// there as proportional segments. A turn somebody owns three times in four is
// one long block; a turn thirty different people have taken is confetti. That
// is the predictability of the pick, drawn rather than scored -- no gauge, no
// 0-100 "certainty index" that would need its own explanation and could only
// ever be a restatement of the widths already on screen.

const POSITIONS = ['QB', 'RB', 'WR', 'TE', 'K', 'DST']

/** The seat is the page. Remembered so somebody researching their own draft
 *  does not re-pick it on every visit; falls back to seat 1 rather than
 *  guessing from anything else. */
const SEAT_KEY = 'archive-seat'

function readSeat(): number {
  try {
    const stored = Number(window.localStorage.getItem(SEAT_KEY))
    return Number.isFinite(stored) && stored >= 1 ? stored : 1
  } catch {
    return 1
  }
}

function posVar(position: string): string {
  const key = position.toLowerCase()
  return POSITIONS.some((p) => p.toLowerCase() === key)
    ? `var(--pos-${key})` : 'var(--text-3)'
}

const pct = (share: number) => `${Math.round(share * 100)}%`

/** Per game, the way every other projection in this app is printed -- the
 *  board holds a season total, and quoting 238.5 here beside a 14.0 in the
 *  draft room would be one projection in two currencies. Null stays null: a
 *  player the board cannot price prints nothing rather than a zero. */
const ppg = (points: number | null): string | null =>
  points === null ? null : (points / SEASON_GAMES).toFixed(1)

// THE PROFILE OPENS OVER THE ARCHIVE, NOT INSTEAD OF IT.
//
// Same popup the draft room uses (PlayerOverlay -> PlayerProfile), for the
// same reason it exists there: a page that navigates away to a player takes
// its own state with it, and this one is a seat, a filter, a search box and a
// scroll position down a table of two hundred names. Nothing here routes.
//
// What the room passes and this page does not: `settings` (there is no league
// -- the archive is read by anybody, connected or not) and `onDraftPlayer`
// (there is no pick to make from a page about drafts other people finished).
// Both are optional and both default, so the profile simply opens without a
// footer that drafts.

/** A name under a turn bar, in the shape the profile paints on the frame it
 *  opens -- the header plus a row of figures, so the popup is readable before
 *  its own 3.5s request lands. The figures are what THIS page knows and the
 *  profile does not: how often the room takes him at this exact pick. */
function seedFromTurn(p: MarketTurnPlayer, pickNo: number): ProfileSeed {
  const perGame = ppg(p.proj_points)
  return {
    name: p.name ?? p.player_id,
    position: p.position,
    // The turn payload carries neither, and a dash invented here would be the
    // page claiming to have looked. The profile fills both in when it lands.
    team: null,
    bye: null,
    rookie: false,
    figures: [
      { label: `At ${pickNo}`, value: pct(p.share), accent: true },
      { label: 'Drafts', value: String(p.count) },
      ...(perGame === null ? [] : [{ label: 'Proj/g', value: perGame }]),
    ],
  }
}


/** What a turn's leading share means, said in words beside the number rather
 *  than instead of it. The bands are the measured ones: at this league shape
 *  a first-round turn runs 45-75%, the middle rounds 12-20%, and the late
 *  ones below 8% -- so these read as "decided", "leaning" and "open" without
 *  the label having to carry a claim the bar does not already make. */
function verdict(topShare: number): string {
  if (topShare >= 0.4) return 'spoken for'
  if (topShare >= 0.2) return 'leans one way'
  if (topShare >= 0.1) return 'open'
  return 'wide open'
}

/** One turn: who goes here, as proportional ink. The tail -- everyone outside
 *  the named few -- is kept as a striped block rather than dropped, because a
 *  bar that only showed the top six would make every turn look decided. */
export function TurnBar({ turn }: { turn: MarketTurn }) {
  const named = turn.players.reduce((sum, p) => sum + p.share, 0)
  const rest = Math.max(0, 1 - named)
  return (
    <div className="mk-bar" role="img"
         aria-label={`${turn.players.length} named picks covering ${pct(named)} of this turn`}>
      {turn.players.map((p) => (
        <span
          key={p.player_id}
          className="mk-bar-seg"
          style={{ width: `${p.share * 100}%`, background: posVar(p.position) }}
          title={`${p.name ?? p.player_id}: ${pct(p.share)} of drafts (${p.count})`}
        />
      ))}
      {rest > 0.001 && (
        <span className="mk-bar-rest" style={{ width: `${rest * 100}%` }}
              title={`${pct(rest)} went to somebody else entirely`} />
      )}
    </div>
  )
}

/** How many of a turn's names the panel shows. Four rather than three since
 *  the bar came off: a turn is a spread, and the fourth name is where the
 *  spread starts being visible -- at a middle-round turn the top three are
 *  often 17/13/13 and the fourth is 12, which is the fact that "22 names"
 *  above it only asserts. */
const TURN_NAMES = 4

/** The turn's leading names, most-taken first. The endpoint already orders
 *  them, and this sorts anyway: the panel's whole claim is that these are the
 *  four likeliest, and that claim should not depend on a server's ORDER BY
 *  staying the way it is today. */
function topPicks(turn: MarketTurn) {
  return [...turn.players].sort((a, b) => b.share - a.share).slice(0, TURN_NAMES)
}

/** A player's distribution over the pick axis, with this seat's own turns
 *  drawn on it. The ticks are the point: an average tells you where he goes,
 *  and this tells you whether he goes before YOU pick. */

export default function Market() {
  useDocumentMeta({
    title: 'Draft data – ESPN Draft Assist',
    description: 'What hundreds of recorded ESPN mock drafts do from each seat: who is taken at your turn, and who is still there at the next one.',
  })
  const navigate = useNavigate()
  const [seat, setSeat] = useState(readSeat)
  const [overview, setOverview] = useState<MarketOverview | null>(null)
  const [slot, setSlot] = useState<MarketSlot | null>(null)
  // Who is reading. The archive is free but not anonymous: reading it needs
  // a connected ESPN account, which is the page's whole abuse control --
  // there is no signup to rate-limit, so the account is the identity. Same
  // probe and same session cache as the landing page, so a reader who
  // connected there is never asked again here.
  const [account, setAccount] = useState<UpcomingDrafts | null>(readAccount)
  const [error, setError] = useState<string | null>(null)
  // The open profile, or nothing. One at a time, and closing it puts the
  // page back exactly as it was because the page never went anywhere.
  const [target, setTarget] = useState<OverlayTarget | null>(null)

  useEffect(() => {
    try { window.localStorage.setItem(SEAT_KEY, String(seat)) } catch { /* private window */ }
  }, [seat])

  useEffect(() => {
    let cancelled = false
    fetchMarketOverview()
      .then((body) => { if (!cancelled) setOverview(body) })
      .catch((err) => { if (!cancelled) setError(String(err.message || err)) })
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    let cancelled = false
    setSlot(null)
    fetchMarketSlot(seat)
      .then((one) => { if (!cancelled) setSlot(one) })
      .catch((err) => { if (!cancelled) setError(String(err.message || err)) })
    return () => { cancelled = true }
  }, [seat])

  useEffect(() => {
    let cancelled = false
    fetchUpcomingDrafts()
      .then((body) => {
        rememberAccount(body)
        if (!cancelled) setAccount(body)
      })
      .catch(() => {
        // The helper being down is not a signed-out reader. Only a tab that
        // has never had an answer falls to the gate.
        if (!cancelled && account === null) setAccount({ connected: false, leagues: [] })
      })
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // A comp clicked INSIDE the profile. The archive can seed it only if that
  // player is in this seat's own table; a comp who is not (somebody nobody in
  // the corpus drafted) opens on the skeleton rather than not opening at all.
  const openComp = (id: string) => {
    // A comp clicked inside the profile opens on the skeleton: this page
    // holds no per-player table to seed it from any more.
    setTarget({ playerId: id, seed: null })
  }

  const seats = overview?.teams ?? 8

  // The two turns worth naming outright, because they are the ones a reader
  // would go looking for: the pick that is already made for them, and the one
  // where the room has no opinion at all.
  // What the path bars are drawn against: the leading path's own share.
  const topPath = Math.max(...(slot?.sequences ?? []).map((s) => s.share), 0.0001)

  const extremes = useMemo(() => {
    const turns = (slot?.turns ?? []).filter((t) => t.observed >= 20)
    if (turns.length === 0) return null
    const sorted = [...turns].sort((a, b) => b.top_share - a.top_share)
    return { decided: sorted[0], open: sorted[sorted.length - 1] }
  }, [slot])

  // The gate. Free is not anonymous: the archive is served only to a reader
  // with a connected ESPN account (see the `account` comment above). The bar
  // stays, with the corpus counts in it -- the claim "free" beside the ask
  // "sign in" should both be visible from here. `account === null` is one
  // unanswered first frame, drawn empty rather than as either page.
  if (account === null || !account.connected) {
    return (
      <div className="mk-page">
        <header className="draft-topbar">
          <Link to="/" className="draft-topbar-title"><Logo /> ESPN Draft Assist</Link>
          <span className="draft-topbar-sep" aria-hidden="true" />
          <nav className="draft-topbar-tabs" aria-label="Views">
            <Link className="draft-tab" to="/">Drafts</Link>
            <Link className="draft-tab is-active" to="/archive" aria-current="page">Data</Link>
            <Link className="draft-tab" to="/live">Live</Link>
          </nav>
          <span className="draft-topbar-sep" aria-hidden="true" />
          <span className="draft-topbar-league">
            {overview
              ? `${overview.drafts} drafts · ${overview.picks.toLocaleString()} picks`
              : 'Reading the data…'}
          </span>
        </header>
        <main className="mk" aria-busy={account === null}>
          {account !== null && (
            <section className="mk-panel mk-gate">
              <h2 className="mk-h2">Draft data needs a signed-in account.</h2>
              <p className="mk-gate-copy">
                Draft data is free. To prevent abuse, you can only read
                it with a connected ESPN account. Connect once with the Draft
                Assistant bookmark. Then come back to this page.
              </p>
              <button type="button" className="mk-gate-cta" onClick={() => navigate('/')}>
                Get set up
              </button>
            </section>
          )}
        </main>
      </div>
    )
  }

  return (
    <div className="mk-page">
      {/* THE PRODUCT'S OWN BAR, not a document header that resembles one.
          Same `.draft-topbar` furniture as the room, the dashboard and
          /mocks -- same 44px height, same wordmark, same tab strip carrying
          the same two views the dashboard offers, with Archive marked as
          the one you are in. Walking between the pages changes what is
          under the bar and never the bar itself.

          Sticky here and nowhere else: the room is a fixed-height cockpit,
          and this is a long page somebody scrolls through a table on. The
          way back out should not be sixteen turns above them. */}
      <header className="draft-topbar">
        <Link to="/" className="draft-topbar-title"><Logo /> ESPN Draft Assist</Link>
        <span className="draft-topbar-sep" aria-hidden="true" />
        <nav className="draft-topbar-tabs" aria-label="Views">
          <Link className="draft-tab" to="/">Drafts</Link>
          <Link className="draft-tab is-active" to="/archive" aria-current="page">Data</Link>
          <Link className="draft-tab" to="/live">Live</Link>
        </nav>
        <span className="draft-topbar-sep" aria-hidden="true" />
        <span className="draft-topbar-league">
          {overview
            ? `${overview.drafts} drafts · ${overview.picks.toLocaleString()} picks`
            : 'Reading the data…'}
        </span>
        {/* The one thing on this page every number below depends on, kept
            in view once the control itself has scrolled away. */}
        <span className="draft-topbar-hint">Seat {seat}</span>
        <span className="draft-topbar-spacer" />
        <Link className="draft-topbar-link" to="/mocks">Mock drafts</Link>
        <Link className="draft-topbar-link" to="/draft">Draft room</Link>
      </header>

      <main className="mk">
        <header className="mk-mast">
          <div className="mk-mast-say">
            <h1 className="mk-title">Draft data</h1>
            <p className="mk-lede">
              Not an average draft position. The whole distribution, from the
              seat you are actually sitting in.
            </p>
            {/* Said once, at the top, because it qualifies every number below
                it: an autodrafted pick is ESPN reading its own list aloud, and
                this project's own seat drafts with this tool. Neither is the
                market. */}
            {overview && (
              <p className="mk-note">
                Behaviour is counted from the {overview.human_picks.toLocaleString()} picks
                a person made. Availability counts every pick, because a player taken by
                a bot is gone just the same.
              </p>
            )}
          </div>
          {/* The corpus, as figures rather than a sentence: these are the
              page's credentials and they should be checkable at a glance. The
              hairline between them is the same rule the tables use -- a strip
              of counted things, not four cards. */}
          {overview && (
            <dl className="mk-facts">
              <div><dt>Drafts</dt><dd className="mono">{overview.drafts}</dd></div>
              <div><dt>Picks</dt><dd className="mono">{overview.picks.toLocaleString()}</dd></div>
              <div>
                <dt>Made by people</dt>
                <dd className="mono">{overview.human_picks.toLocaleString()}</dd>
              </div>
            </dl>
          )}
        </header>

        <nav className="mk-seats" aria-label="Your draft pick">
          <span className="draft-cap">Your pick</span>
          <div className="mk-seat-row">
            {Array.from({ length: seats }, (_, i) => i + 1).map((n) => (
              <button
                key={n}
                type="button"
                className={`mk-seat${n === seat ? ' is-active' : ''}`}
                onClick={() => setSeat(n)}
                aria-pressed={n === seat}
              >
                {n}
              </button>
            ))}
          </div>
        </nav>

        {error && <p className="mk-error" role="alert">{error}</p>}

        <div className="mk-cols">
          <section className="mk-panel">
            <h2 className="mk-h2">Your turns, and what the room does with them</h2>
            {!slot && !error && <p className="mk-loading">Reading the data…</p>}
            <ol className="mk-turns">
              {(slot?.turns ?? []).map((turn) => (
                <li className="mk-turn" key={turn.pick_no}>
                  <div className="mk-turn-head">
                    <span className="mk-turn-pick mono">{turn.pick_no}</span>
                    <span className="mk-turn-round">R{turn.round}</span>
                    <span className="mk-turn-verdict">{verdict(turn.top_share)}</span>
                    <span className="mk-turn-meta mono">
                      {pct(turn.top_share)} · {turn.distinct} names · {turn.observed} drafts
                    </span>
                  </div>
                  {/* NO BAR. A stacked bar over these tiles drew the same
                      shares twice, once as ink and once as numbers, and the
                      ink was the half nobody could read a value off: the
                      three named segments were already labelled underneath,
                      and the striped tail said "somebody else" in a shape a
                      reader could not count. The head's own figures (top
                      share, how many names, how many drafts) carry what the
                      bar was for, and the room the bar took now shows a
                      fourth player.

                      The same tile the waiting room draws for the same
                      question (PickOption): face, badge, name, share,
                      projection. Clickable on this page only: the archive
                      has a profile to open over it. */}
                  <ol className="mk-turn-names">
                    {topPicks(turn).map((p) => (
                      <li key={p.player_id}>
                        <PickOption
                          name={p.name ?? p.player_id}
                          position={p.position}
                          headshot={p.headshot}
                          share={p.share}
                          projPoints={p.proj_points}
                          projChange={p.proj_change}
                          shareTitle={`${p.count} of ${turn.observed} recorded picks here`}
                          onOpen={() => setTarget({
                            playerId: p.player_id,
                            seed: seedFromTurn(p, turn.pick_no),
                          })}
                        />
                      </li>
                    ))}
                  </ol>
                </li>
              ))}
            </ol>
          </section>

          <aside className="mk-side">
            {/* The way to the rows every figure on this page is counted
                from. It used to be the list itself, at the foot of the page,
                where a reader who wanted to check a number had to scroll
                past every conclusion drawn from it first. */}
            <section className="mk-panel mk-raw">
              <span className="draft-cap">Raw data</span>
              {/* The corpus, as the two figures the rest of the page is
                  counted from. Mono, like every figure in the archive's
                  tables, because that is what they are: rows and cells. */}
              <p className="mk-raw-figures">
                <strong className="mono">{overview?.drafts ?? '—'}</strong> drafts
                <span className="mk-raw-dot" aria-hidden="true">·</span>
                <strong className="mono">
                  {overview ? overview.picks.toLocaleString() : '—'}
                </strong> picks
              </p>
              {/* Styled as a row of the table it opens: a hairline above, a
                  hairline below, the arrow at the far edge. */}
              <Link className="mk-raw-cta" to="/archive/data">
                <span>Browse every draft, pick by pick</span>
                <span className="mk-raw-arrow" aria-hidden="true">→</span>
              </Link>
            </section>

            <section className="mk-panel">
              <h2 className="mk-h2">How the room plays seat {seat}</h2>
              <p className="mk-sub">
                The first {slot?.sequence_rounds ?? 5} rounds, in order, as people
                actually took them, across {slot?.sequences_observed ?? 0} complete runs.
              </p>
              <ol className="mk-paths">
                {(slot?.sequences ?? []).map((s) => (
                  <li key={s.path.join('-')}>
                    <span className="mk-path">
                      {s.path.map((pos, i) => (
                        <span key={i} className="mk-chip"
                              style={{ color: posVar(pos),
                                       borderColor: posVar(pos) }}>{pos}</span>
                      ))}
                    </span>
                    {/* Read against the commonest path rather than against
                        100%: no opening five rounds is walked by even half a
                        room, so bars scaled to the whole would all be
                        slivers. The figure beside it is the real share. */}
                    <span className="mk-path-share">
                      <span className="mk-path-track">
                        <span className="mk-path-fill"
                              style={{ width: `${(s.share / topPath) * 100}%` }} />
                      </span>
                      <span className="mono">{pct(s.share)}</span>
                    </span>
                  </li>
                ))}
              </ol>
            </section>

            {extremes && (
              <section className="mk-panel">
                <h2 className="mk-h2">The turn you can plan, and the one you can't</h2>
                <div className="mk-extreme">
                  <span className="draft-cap">Most decided</span>
                  <p className="mk-extreme-line">
                    <strong className="mono">Pick {extremes.decided.pick_no}</strong>:{' '}
                    {extremes.decided.players[0]?.name ?? '—'} in{' '}
                    {pct(extremes.decided.top_share)} of drafts, {extremes.decided.distinct}{' '}
                    names in all.
                  </p>
                </div>
                <div className="mk-extreme">
                  <span className="draft-cap">Most open</span>
                  <p className="mk-extreme-line">
                    <strong className="mono">Pick {extremes.open.pick_no}</strong>: the
                    leading pick is only {pct(extremes.open.top_share)}, across{' '}
                    {extremes.open.distinct} different names.
                  </p>
                </div>
              </section>
            )}
          </aside>
        </div>

      </main>

      {target && (
        <PlayerOverlay
          target={target}
          onClose={() => setTarget(null)}
          onSelectPlayer={openComp}
        />
      )}
    </div>
  )
}
