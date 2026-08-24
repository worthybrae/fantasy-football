import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { connectWithToken, fetchConnectProgress, fetchLiveState,
         type ConnectProgress, type LiveState, type TokenConnectParams } from '../api'
import { BOOKMARKLET } from '../lib/bookmarklet'
import { Logo } from '../components/Logo'
import ConnectScreen from '../components/ConnectScreen'
import ReadinessStrip from '../components/ReadinessStrip'
import LobbyStrip from '../components/LobbyStrip'
import HeroBoard from '../components/HeroBoard'
// This page's own stylesheet, not App.css: see the header comment in it for
// why, and for why every class below is `lp-` prefixed.
import '../landing.css'

// The front door, and the bookmarklet onboarding.
//
// A bookmarklet cannot be detected -- browsers hide the bookmarks bar from
// pages by design -- so there is nothing to poll for and no "is it installed?"
// gate. Instead the page always shows itself, and flips to "live" the moment
// the bookmarklet is actually used: clicking it on the ESPN draft page mints a
// token there and opens THIS page in a new window with the token in the URL
// hash. That hash is the only signal that matters (did it work), and it is
// self-announcing.
//
//   connecting -> arrived from the bookmarklet with a token: open the socket
//   idle       -> the default: pitch, prove, hand over the bookmarklet
//   live       -> a connect is already running: offer the board
//
// There is no separate `error` gate any more. The connect screen renders its
// own failure (ConnectScreen's Failed), because a failure that names the
// stage it died on and how far it got is worth more than a one-line band on
// a page the user was not looking at.
type Gate = 'connecting' | 'idle' | 'live'

// How often the connect screen asks the helper what it is doing. Far faster
// than the room's own 2.5s poll, and it can afford to be: GET
// /api/live/connect-progress touches no database at all (see its docstring),
// unlike /api/live/state, which runs a COUNT(*), a pick replay and the vor
// fallback ranking on every call. Fast enough that a stage lasting a couple
// of hundred milliseconds is still seen; slow enough to be nothing next to
// the work being reported on.
const PROGRESS_POLL_MS = 400

// The token params the bookmarklet packs into the hash of the window it opens.
// Returns null unless all four load-bearing fields are present -- season is
// optional-ish (the bookmarklet always sends it, but a connect can proceed on
// the ids alone) so it is read separately.
function tokenFromHash(hash: string) {
  const p = new URLSearchParams(hash.replace(/^#/, ''))
  const leagueId = p.get('leagueId')
  const teamId = p.get('teamId')
  const swid = p.get('swid')
  const token = p.get('token')
  if (!leagueId || !teamId || !swid || !token) return null
  return { leagueId, teamId, swid, token, season: p.get('season') || '' }
}

// ---------------------------------------------------------------------------
// Can this browser actually drag a link onto the bookmarks bar?
// ---------------------------------------------------------------------------
//
// The hero's primary action is a drag target, and a drag target on a phone
// is a button that does nothing: there is no bookmarks bar to drop it on and
// no gesture that would reach one. So the page has to know, and it has to
// know before first paint (a hero that swaps its main action a moment after
// it renders is a layout shift on the most important block of the page).
//
// DELIBERATELY NOT A WIDTH TEST. Width answers a different question than the
// one being asked. A desktop window dragged down to 700px still drags
// perfectly; a 1366px-wide tablet cannot drag at all. Two capability facts
// answer the real one:
//
//   'draggable' in a <div>  -- the HTML drag-and-drop API exists here.
//   (hover: hover) and (pointer: fine) -- the browser's PRIMARY input is a
//   precise pointer that can hover, which is what dragging something out of
//   the page and onto the browser's own chrome requires. iPadOS reports
//   (hover: none) and (pointer: coarse) no matter how wide the window is; a
//   laptop reports fine/hover at 320px.
//
// A hybrid -- a touchscreen laptop, a tablet with a trackpad attached --
// reports its primary input as fine/hover and gets the drag target, which is
// the right answer: it has a mouse. Detaching that trackpad fires the change
// event below and the hero switches over.
//
// When there is no matchMedia at all, the answer is "no". The fallback is a
// working page with an honest sentence; the drag target is the branch that
// breaks if it is shown to the wrong browser, so it is the branch that has
// to be earned.
const DRAG_QUERY = '(hover: hover) and (pointer: fine)'

function canDragNow(): boolean {
  if (typeof window === 'undefined' || !window.matchMedia) return false
  if (!('draggable' in document.createElement('div'))) return false
  return window.matchMedia(DRAG_QUERY).matches
}

function useCanDrag(): boolean {
  // Computed in the initialiser, not in an effect, so the first paint is
  // already the right hero.
  const [canDrag, setCanDrag] = useState(canDragNow)
  useEffect(() => {
    if (!window.matchMedia) return
    const mq = window.matchMedia(DRAG_QUERY)
    const sync = () => setCanDrag(canDragNow())
    mq.addEventListener('change', sync)
    return () => mq.removeEventListener('change', sync)
  }, [])
  return canDrag
}

// The bookmarklet as a real `javascript:` anchor.
//
// Present from first paint so a drag to the bookmarks bar copies IT, not
// this page's URL -- setting the href after mount was too late, and the drag
// grabbed localhost instead. React strips `javascript:` from href props, so
// it is injected as raw HTML. The string has no single quotes (verified in
// bookmarklet.ts, which also explains why it is comment-free and must not be
// reformatted), so a single-quoted href is safe; onclick returns false so a
// stray click does nothing -- it is a drag target, not a button.
//
// The football rides in the anchor's TEXT, as an emoji rather than the SVG
// mark: a bookmark's title is the anchor's textContent, so an inline <svg>
// would be dropped on the way to the bar and the label would arrive naked.
// Nothing else may be added inside the anchor for the same reason.
//
// `describedBy` points at the copy hint beside it. The chip is the hero's
// first tab stop and, by design, does nothing on Enter -- a click would run
// the bookmarklet against this page rather than a draft. A keyboard reader
// who is told only "Draft Assistant" and then gets silence has hit a dead
// end; described by the hint, they are told the same thing the sighted
// reader is, that the route for them is the copy button next along.
//
// An attribute, not content: `textContent` is what becomes the bookmark's
// title, so this cannot disturb it the way an inner element would.
function bookmarkAnchor(describedBy?: string): string {
  const described = describedBy ? "aria-describedby='" + describedBy + "' " : ""
  return "<a class='lp-bookmark' title='Drag me to your bookmarks bar' "
    + described
    + "onclick='return false' href='" + BOOKMARKLET + "'>"
    + "<span aria-hidden='true'>🏈</span>&nbsp;Draft&nbsp;Assistant</a>"
}

// ---------------------------------------------------------------------------
// The keyboard path for the chip above: not dragging it, copying it.
// ---------------------------------------------------------------------------
//
// The anchor bookmarkAnchor() renders is a drag target and, deliberately,
// nothing else -- its onclick returns false so neither a stray click nor an
// Enter/Space on a focused chip does anything (see the comment on
// bookmarkAnchor). That is correct for a mouse, where the chip is never
// clicked, only dragged. It is a dead end for a keyboard: the chip is the
// hero's first tab stop, Enter does nothing, and nothing on the page says
// another route exists. Dragging is inherently mouse-only and that part
// cannot be fixed -- but the dead end is not inherent, so this is the
// button a keyboard user reaches instead. It copies the exact same
// `javascript:` string, verbatim, so it can be pasted as a hand-made
// bookmark's address.
async function copyBookmarklet(): Promise<boolean> {
  // The Clipboard API needs a secure context (https, or localhost -- both
  // covered here) and can still reject (permission denied, an iframe
  // without the right allow policy). Either way, fall through to the
  // legacy path rather than surface that as the only route.
  if (typeof navigator !== 'undefined' && navigator.clipboard && window.isSecureContext) {
    try {
      await navigator.clipboard.writeText(BOOKMARKLET)
      return true
    } catch {
      /* fall through */
    }
  }
  // document.execCommand('copy') needs no permission and no secure context,
  // which is what makes it worth keeping as a fallback rather than just
  // reporting failure. An off-screen, unfocusable-by-tab textarea holds the
  // text just long enough to select and copy it, then is removed.
  try {
    const ta = document.createElement('textarea')
    ta.value = BOOKMARKLET
    ta.setAttribute('readonly', '')
    ta.style.position = 'fixed'
    ta.style.opacity = '0'
    ta.style.left = '-9999px'
    document.body.appendChild(ta)
    ta.select()
    const ok = document.execCommand('copy')
    document.body.removeChild(ta)
    return ok
  } catch {
    return false
  }
}

// Rendered beside the chip everywhere it appears (the hero, and again in
// the setup block). `idPrefix` keeps the two instances' ids from colliding
// when both are on the page at once. The result is announced through
// role="status"/aria-live so a screen reader hears it without focus ever
// leaving the button -- the button's own label never changes.
function BookmarkCopy({ idPrefix }: { idPrefix: string }) {
  const [status, setStatus] = useState<'idle' | 'ok' | 'fail'>('idle')
  const onCopy = useCallback(() => {
    copyBookmarklet().then((ok) => setStatus(ok ? 'ok' : 'fail'))
  }, [])
  return (
    <div className="lp-copy">
      <button
        type="button"
        className="lp-copy-btn"
        onClick={onCopy}
        aria-describedby={`${idPrefix}-copy-hint`}
      >
        Copy install link
      </button>
      <p id={`${idPrefix}-copy-hint`} className="lp-copy-hint">
        Can’t drag it? Copy the link, then create a bookmark and paste it as the address.
      </p>
      <span
        className={
          status === 'idle' ? 'lp-copy-status'
            : status === 'ok' ? 'lp-copy-status is-ok' : 'lp-copy-status is-fail'
        }
        role="status"
        aria-live="polite"
      >
        {status === 'ok' && 'Copied — paste it as a bookmark’s address.'}
        {status === 'fail' && 'Couldn’t copy automatically — try dragging the chip instead.'}
      </span>
    </div>
  )
}

// The one sentence a visitor who cannot drag gets instead of the chip. Said
// once, in the hero, and again above the setup block's own drag target so
// that scrolling down does not land them back on the thing they were just
// told they cannot use.
const NO_DRAG_LINE = (
  <>
    Draft Assistant installs as a <strong>bookmark you drag to your browser’s
    bookmarks bar</strong>, so setting it up needs a desktop browser.
  </>
)

// ---------------------------------------------------------------------------
// The page's content. Every number below is a real reading off this tool, and
// each one names where it came from, because a marketing claim that has gone
// stale is worse than a weaker claim that is still true.
// ---------------------------------------------------------------------------

// The hero's board. Measured on 2026-08-19 against data/nfl.duckdb with the
// same three functions the live room runs -- build_board -> build_pool ->
// survival(n_rollouts=400) -> gain.rank_available -- for this state:
//
//   8-team PPR (the app's own league defaults), my slot 2, ON THE CLOCK at
//   pick 15 (round 2), the first fourteen players off the board in ESPN ADP
//   order, my one earlier pick a running back. Horizon: pick 27, which is
//   what horizon_target returns for that state, and survival is counted to
//   exactly that pick.
//
// Full measured rows (vor / gain / survival), tool's own ranking order:
//   1 Rashee Rice    WR  57.3  +21.22   1.75%
//   2 Trey McBride   TE  65.4  +11.34   5.00%
//   3 Brock Bowers   TE  63.6   +9.49  68.25%
//   4 Omarion Hampton RB 71.1   +7.59   1.50%
//   6 Josh Allen     QB  76.0   +3.39  92.75%
//
// Four of those are shown, ordered by value rather than by the tool's rank,
// because the point of the panel is that the two columns run in opposite
// directions: the most valuable player on the board is the one it is
// cheapest to wait on. Figures are rounded the way the room rounds them
// (ConfirmPick.fmtSigned: signed, no decimals).
const PROOF_ROWS = [
  { pos: 'QB', name: 'Josh Allen', vor: '+76', gain: '+3', tone: 'bad', take: false },
  { pos: 'RB', name: 'Omarion Hampton', vor: '+71', gain: '+8', tone: '', take: false },
  { pos: 'TE', name: 'Trey McBride', vor: '+65', gain: '+11', tone: 'good', take: true },
  { pos: 'WR', name: 'Rashee Rice', vor: '+57', gain: '+21', tone: 'good', take: true },
]

// Four things a ranked list structurally cannot tell you. Each `proof` is a
// reading, not an adjective; `note` is the part that makes the reading mean
// something, and in two cases it is the caveat rather than the boast.
const FEATURES = [
  {
    n: '01',
    title: 'It prices waiting, not just value',
    body: 'Every board ranks players by how good they are. This one measures what '
      + 'passing actually costs you — the gap between a player and the best one at '
      + 'his position the model still expects to be there when you pick again.',
    // Same row as the hero panel, same measurement.
    proof: 'Josh Allen · +76 over replacement · +3 gain vs waiting',
    note: 'the biggest number on the board, attached to the wrong pick',
  },
  {
    n: '02',
    title: 'It simulates the picks between now and your turn',
    body: 'Value only means anything against what will survive. Every time a pick '
      + 'lands, the draft ahead of you is run 400 times, opponent by opponent, out '
      + 'to a turn far enough away for the difference to be real — and every player '
      + 'is priced against the best one likely to still be on the board there.',
    // api/live.py SURVIVAL_ROLLOUTS = 400, one survival() pass per recompute,
    // and a recompute is queued on every pick the socket reports.
    proof: '400 simulated runs a pick · every opponent modelled',
    // The honest half. scoring/draft_model.cold_start_fits is what a league
    // with no imported draft history gets, which is every first connect and
    // every mock: one market-following model shared by all opponents. The
    // per-manager fits only exist once the league's own past drafts are in
    // (api/live.py's `history` stage says which of the two you got).
    note: 'a market-following model for every opponent by default; a separate model '
      + 'per manager once your league’s own past drafts are imported',
  },
  {
    n: '03',
    title: 'It knows consistency, not just averages',
    body: 'Two backs average twenty points. One gives you twenty every week; the '
      + 'other gives you five and then fifty. Consistency is scored per point of '
      + 'production, so a big scorer is not punished for scoring, and ranked '
      + 'against everyone else at the position.',
    // scoring/profile_cache: 2025, 8-game qualifier. RB3 of 97 on points per
    // game; coefficient-of-variation rank 28 of 95 (95, not 97, because two
    // qualifying backs scored <= 0 a game and have no coefficient at all).
    proof: 'Jahmyr Gibbs 2025 · RB3 of 97 by points a game · 28th steadiest of 95',
    note: 'on raw week-to-week spread the same season ranks 97th of 97 — apparently '
      + 'the most volatile back in the league',
  },
  {
    n: '04',
    title: 'It knows what seasons like this became',
    body: 'For any player it finds every comparable season since 2016 — same '
      + 'production, same point in a career — and shows what those players did the '
      + 'year after. A projection is a guess. This is a record.',
    // scoring/profile_cache.comparable_pool, Gibbs' 2025 at the +-3 ppg /
    // +-1 year band; 2016 is scoring/config.HISTORY_SEASONS' floor. The 21.5
    // is ESPN's own 2026 projection for him: 365.3 points over 17 games.
    proof: '19 comparable seasons · median −2.8 points a game · 13 of 19 declined',
    note: 'Gibbs’ own comparables, against the 21.5 a game ESPN projects for him',
  },
]

export default function Landing() {
  const [gate, setGate] = useState<Gate>('idle')
  const [error, setError] = useState<string | null>(null)
  const [progress, setProgress] = useState<ConnectProgress | null>(null)
  const [live, setLive] = useState<LiveState | null>(null)
  const [sessionReady, setSessionReady] = useState(false)
  // Bumped by Retry. Every connect-owned piece of state is keyed off it, so a
  // retry starts from a blank screen rather than from the last one's rows.
  const [attempt, setAttempt] = useState(0)
  const canDrag = useCanDrag()
  const navigate = useNavigate()

  // The token the bookmarklet delivered, kept for the whole session. It has
  // to outlive the hash (which is wiped on arrival, see below) because Retry
  // re-runs this exact connect -- there is no other copy of it anywhere, and
  // making the user click the bookmark again just to retry would be asking
  // them to fix something they cannot see.
  const paramsRef = useRef<TokenConnectParams | null>(null)
  // Which attempt has already been POSTed. A ref, not state: React's
  // StrictMode runs every effect twice in development, and without this the
  // second run fires a second connect that supersedes the first mid-build.
  const postedRef = useRef(-1)
  // Whether the record now on /api/live/connect-progress is OURS.
  //
  // The helper keeps the last connect's record after it finishes, and this
  // page starts polling before its own POST has reached the server -- so the
  // first read can be a FINISHED record belonging to an earlier attempt.
  // Caught in a real run: the second of two connects rendered the first
  // one's failed record, stopped polling on it, and showed a stage list that
  // was missing a row its own connect actually ran. A terminal record is
  // only believed once this page has seen its own connect in flight (a
  // `connecting` phase, which every connect spends seconds in -- 4-35s
  // measured) or its own POST has settled.
  const ownRecordRef = useRef(false)

  useEffect(() => {
    // Arrived from the bookmarklet? Keep the token and open the socket.
    const params = tokenFromHash(window.location.hash)
    if (params) {
      paramsRef.current = params
      setGate('connecting')
      // Drop the token out of the address bar immediately -- it should not sit
      // there, be bookmarked, or survive a refresh into a duplicate connect.
      window.history.replaceState(null, '', window.location.pathname)
    }
  }, [])

  // The connect itself: one POST per attempt, and it stays blocked for as
  // long as the work takes (4-35s measured, depending on the league). What
  // makes that bearable is the poll below, which runs concurrently against a
  // different endpoint -- the POST's own response still arrives only at the
  // end, and it is still the thing that decides success or failure.
  useEffect(() => {
    if (gate !== 'connecting' || !paramsRef.current) return
    if (postedRef.current === attempt) return
    postedRef.current = attempt
    connectWithToken(paramsRef.current)
      .then(() => { ownRecordRef.current = true; setSessionReady(true) })
      .catch((e) => {
        ownRecordRef.current = true
        setError(e instanceof Error ? e.message : String(e))
      })
  }, [gate, attempt])

  // What the connect is doing, while it does it. Stops the moment the record
  // settles (every stage terminal, or one of them failed) -- and then reads
  // the live state once, for the only fact on the handoff screen that no
  // connect stage discovers: the pick clock.
  useEffect(() => {
    if (gate !== 'connecting') return
    let cancelled = false
    let timer = 0
    let settledAt = 0

    const poll = async () => {
      try {
        const body = await fetchConnectProgress()
        if (cancelled) return
        const settled = body.phase === 'ready' || body.phase === 'failed'
        if (body.phase === 'connecting') ownRecordRef.current = true
        if (settled && !ownRecordRef.current) {
          // A finished record from an earlier connect, read before ours has
          // started. Not ours to show, and not a reason to stop.
          timer = window.setTimeout(poll, PROGRESS_POLL_MS)
          return
        }
        setProgress(body)
        if (settled && !settledAt) {
          settledAt = Date.now()
          fetchLiveState().then((s) => { if (!cancelled) setLive(s) }).catch(() => {})
        }
        // A short grace period rather than stopping dead on the first
        // terminal phase. `failed` is set the instant one stage fails, and
        // another can still be running at that moment -- the ranking pass
        // landed 50ms after a socket failure in a real run, and stopping on
        // the first terminal read left it drawn as a spinner that would
        // never resolve. Every stage terminal ends it immediately.
        const open = body.stages.some(
          (s) => s.status === 'pending' || s.status === 'running')
        if (settled && (!open || Date.now() - settledAt > 2000)) return
      } catch {
        // The helper is momentarily unreachable. Keep polling: the connect
        // itself is a separate request and is still running, and the POST is
        // what reports a real failure.
      }
      if (!cancelled) timer = window.setTimeout(poll, PROGRESS_POLL_MS)
    }
    poll()
    return () => { cancelled = true; window.clearTimeout(timer) }
  }, [gate, attempt])

  // A draft that is already running does not wait for anyone to read a
  // summary: if the socket says the draft has started, the pick clock is
  // ticking and the room is where the user needs to be. This is not a timer
  // or a minimum display time -- it fires only on a fact off the socket, and
  // only once everything is done.
  useEffect(() => {
    if (progress?.phase === 'ready' && live?.draft_started) navigate('/draft')
  }, [progress?.phase, live?.draft_started, navigate])

  // No token in the hash: this is the landing view. Still poll in case a
  // connect is already running from a click in another window -- then this
  // page can offer the board rather than pitch a tool already in use.
  useEffect(() => {
    if (gate !== 'idle') return
    let cancelled = false
    const poll = async () => {
      try {
        const state = await fetchLiveState()
        if (!cancelled && (state.active || state.token_received)) setGate('live')
      } catch {
        /* helper not up yet; the page simply stays on the landing view */
      }
    }
    poll()
    const id = setInterval(poll, 2500)
    return () => { cancelled = true; clearInterval(id) }
  }, [gate])

  const retry = useCallback(() => {
    setError(null)
    setProgress(null)
    setLive(null)
    setSessionReady(false)
    // The record on the server is still the failed/degraded one this retry is
    // replacing, so it has to be disowned too -- otherwise the new attempt
    // renders the old attempt's ending for its first few hundred ms.
    ownRecordRef.current = false
    setAttempt((n) => n + 1)
  }, [])

  // What every call-to-action on this page does.
  //
  // There is no payment integration and no hosted signup, so a button that
  // implied either would be lying about what happens next. What actually
  // starts a draft -- free or paid, mock or real -- is the bookmarklet, and
  // the honest thing a CTA can do is put it in front of you. The hero does
  // that literally now (the bookmarklet IS its primary action); every other
  // button on the page scrolls to the setup block, where the same chip sits
  // above the three steps, and nothing else claims to happen.
  //
  // `smooth` only when the visitor has not asked for less motion; a page
  // that ignores that preference to animate a scroll is the exact case the
  // preference exists for.
  const toSetup = useCallback(() => {
    const el = document.getElementById('setup')
    if (!el) return
    const reduced = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
    el.scrollIntoView({ behavior: reduced ? 'auto' : 'smooth', block: 'start' })
  }, [])

  // The screen this whole task is about. It replaces the page rather than
  // sitting above it, and it stays up until the work is genuinely finished --
  // including the two stages that land after the connect returns (the socket
  // handshake and the first ranking pass).
  if (gate === 'connecting') {
    return (
      <ConnectScreen
        progress={progress ?? {
          // Before the first poll lands there is genuinely nothing to report.
          // An empty list draws the header and an empty rule, not invented rows.
          phase: 'connecting', stages: [], facts: {}, error: null, elapsed_ms: 0,
        }}
        live={live}
        sessionReady={sessionReady}
        error={error}
        onEnter={() => navigate('/draft')}
        onRetry={retry}
        onBack={() => { paramsRef.current = null; setGate('idle') }}
      />
    )
  }

  return (
    <main className="lp">
      <header className="lp-bar">
        <span className="lp-mark">
          <Logo size={17} />
          Draft Assistant
        </span>
        <span className="lp-bar-price">
          Mock drafts free · <span className="mono">$4.99</span> a real draft
        </span>
        {/* The only way into /mocks that isn't typing the URL: a reading room
            for drafts already played, off to the side of the pitch rather
            than in it. */}
        <Link to="/mocks" className="lp-bar-link">See mock drafts</Link>
      </header>

      {gate === 'live' && (
        <div className="lp-band lp-band-live">
          <span>Your draft is synced.</span>
          <button className="lp-band-action" onClick={() => navigate('/draft')}>
            Go to the board
          </button>
        </div>
      )}

      {/* A failed connect no longer lands here as a one-line band. It keeps
          the connect screen, which can say which stage stopped, how far it
          got, and what to do about it -- see ConnectScreen's Failed. "Back to
          setup" is the way to this page, and it clears the token first so
          this view is the pitch again rather than a half-dead connect. */}
      {error !== null && gate === 'idle' && (
        <div className="lp-band lp-band-error">
          <span>Couldn’t connect: {error}</span>
          <button className="lp-band-action" onClick={() => setError(null)}>
            Dismiss
          </button>
        </div>
      )}

      {/* -- the argument, and the proof of it, above the fold -- */}
      <section className="lp-sec lp-hero">
        {/* A synthetic draft board, drafting itself, behind everything in
            this section. Decoration -- aria-hidden and pointer-events: none
            -- and absolutely positioned, so it adds nothing to the layout
            and shifts nothing when it mounts. See components/HeroBoard.tsx
            for why it is DOM rather than a video, and landing.css for the
            scrim that keeps every line of copy below at the contrast it had
            before the board existed. */}
        <HeroBoard />

        <div className="lp-hero-copy">
          <p className="lp-cap lp-cap-accent">For ESPN fantasy leagues</p>
          {/* Two sentences, two blocks rather than one string with a <br>:
              `text-wrap: balance` balances a block, so with a line break
              inside one block the second sentence broke after "who" and hung
              two words on their own line. As separate blocks each sentence
              balances itself, at every width. */}
          <h1 className="lp-title">
            <span>ESPN tells you who’s best.</span>
            <span>This tells you who to take.</span>
          </h1>
          <p className="lp-lede">
            Every draft board ranks players. None of them price what it costs to
            wait. Draft Assistant replaces your ESPN draft room with one that
            measures both — and simulates the picks between now and your next
            turn to work out the difference.
          </p>
          {/* The hero's action. There is no signup and no download: what a
              visitor actually has to do is move one object onto their
              bookmarks bar, so that object is the primary action rather than
              a button that scrolls to where it was hidden.

              Where it cannot be done, it is not offered. See `useCanDrag`
              above for what is tested; the branch is decided before first
              paint, so neither version arrives late. */}
          <div className="lp-hero-act">
            {canDrag ? (
              <div className="lp-drag">
                <p className="lp-cap lp-cap-accent lp-drag-cap">
                  <span className="lp-drag-arrow" aria-hidden="true">↑</span>
                  Drag this to your bookmarks bar
                </p>
                <div className="lp-drag-chip">
                  <span dangerouslySetInnerHTML={{ __html: bookmarkAnchor('hero-copy-hint') }} />
                  {/* The chip's onclick returns false on purpose (see
                      bookmarkAnchor) -- it does nothing for a mouse click or a
                      keyboard Enter alike. This is the route a keyboard user
                      gets instead. See copyBookmarklet/BookmarkCopy above. */}
                  <BookmarkCopy idPrefix="hero" />
                </div>
                <p className="lp-drag-note">
                  Once, ever. Then click it in your ESPN draft room — a mock
                  counts — and your board opens here, priced before the first pick.
                </p>
                <div className="lp-cta-row">
                  <button className="lp-cta lp-cta-ghost" onClick={toSetup}>
                    Try it in a mock draft
                  </button>
                  <span className="lp-cta-note">free, no account</span>
                </div>
              </div>
            ) : (
              <>
                <p className="lp-nodrag-line">{NO_DRAG_LINE}</p>
                <div className="lp-cta-row">
                  <button className="lp-cta" onClick={toSetup}>Try it in a mock draft</button>
                  <span className="lp-cta-note">free, no account</span>
                </div>
              </>
            )}
            {/* Renders nothing until ESPN's own lobby answers, and nothing at
                all if it can't -- see LobbyStrip's module comment. */}
            <LobbyStrip />
          </div>
        </div>

        {/* Not a screenshot and not an illustration: a board this tool
            actually produced, with the state it was produced from written
            underneath it. See PROOF_ROWS for the full measurement. */}
        <aside className="lp-proof">
          <div className="lp-proof-head">
            <span className="lp-cap lp-cap-accent">Round 2, on the clock</span>
            <span className="lp-proof-sub">what every other board says, against what this one says</span>
          </div>
          {PROOF_ROWS.map((r) => (
            <div className={r.take ? 'lp-proof-row is-take' : 'lp-proof-row'} key={r.name}>
              <span className="lp-proof-who">
                <span className={`lp-pos lp-pos-${r.pos.toLowerCase()}`}>{r.pos}</span>
                <span className="lp-proof-name">{r.name}</span>
              </span>
              <span className="lp-proof-fig">
                <span className="lp-cap">Over replacement</span>
                <span className="lp-proof-num mono">{r.vor}</span>
              </span>
              <span className="lp-proof-fig">
                <span className="lp-cap">Gain vs waiting</span>
                <span className={`lp-proof-num mono${r.tone ? ` is-${r.tone}` : ''}`}>{r.gain}</span>
              </span>
            </div>
          ))}
          <p className="lp-proof-foot">
            Josh Allen is the most valuable player on that board and the
            cheapest one to pass on. The next quarterback is nearly as good, and
            across 400 simulated runs to the next turn Allen was still on the
            board 93% of the time. Rashee Rice, worth nineteen points less, was
            there 2% — so the tool takes Rice and lets Allen come back round.
            <span className="lp-proof-src">
              8-team PPR, my slot on the clock at pick 15, the first fourteen
              picks gone in ESPN ADP order. Survival counted to pick 27, over
              400 simulated drafts — the same numbers the room shows on the
              night.
            </span>
          </p>
        </aside>
      </section>

      {/* -- what a ranked list cannot do -- */}
      <section className="lp-sec">
        <div className="lp-lead">
          <p className="lp-cap">What it knows that ESPN doesn’t</p>
          <h2 className="lp-h2">Four things a rankings list structurally cannot tell you.</h2>
        </div>
        <div className="lp-grid">
          {FEATURES.map((f) => (
            <article className="lp-card" key={f.n}>
              <div className="lp-card-head">
                <span className="lp-card-n mono">{f.n}</span>
                <h3 className="lp-card-title">{f.title}</h3>
              </div>
              <p className="lp-card-body">{f.body}</p>
              <div className="lp-proof-line">
                <p className="lp-proof-line-value mono">{f.proof}</p>
                <p className="lp-proof-line-note">{f.note}</p>
              </div>
            </article>
          ))}
        </div>
      </section>

      {/* -- how it works, and the objection everyone has -- */}
      <section className="lp-sec lp-split" id="setup">
        <div className="lp-lead">
          <p className="lp-cap">How it works</p>
          <h2 className="lp-h2">One click from the draft room you’re already in.</h2>

          {/* The same chip the hero hands over, kept here because this is
              where the three steps explain what to do with it. On a browser
              that cannot drag it, the honest line comes first rather than
              letting someone who was just told they need a desktop land back
              on the drag target -- the chip stays visible underneath it, for
              anyone whose browser the test read wrong. See bookmarkAnchor
              above for why this is raw HTML. */}
          {!canDrag && <p className="lp-nodrag-line">{NO_DRAG_LINE}</p>}
          <div className="lp-bookmark-row">
            <span dangerouslySetInnerHTML={{ __html: bookmarkAnchor('setup-copy-hint') }} />
            {canDrag && (
              <span className="lp-bookmark-hint">← drag this to your bookmarks bar</span>
            )}
            <BookmarkCopy idPrefix="setup" />
          </div>

          {/* Numbered because this genuinely is a sequence: each step is only
              possible once the one above it is done. */}
          <ol className="lp-steps">
            <li className="lp-step">
              <span className="lp-step-n mono">1</span>
              <span>
                Drag the button above to your bookmarks bar. Once, ever — press{' '}
                <kbd>⌘⇧B</kbd> / <kbd>Ctrl⇧B</kbd> first if the bar is hidden.
              </span>
            </li>
            <li className="lp-step">
              <span className="lp-step-n mono">2</span>
              <span>
                Open your ESPN draft room — a mock counts — and click{' '}
                <strong>🏈 Draft&nbsp;Assistant</strong> there. Your board is built,
                priced and ranked before the first pick lands.
              </span>
            </li>
            <li className="lp-step">
              <span className="lp-step-n mono">3</span>
              <span>
                Draft from the window it opens. Clock, board, roster and
                recommendation in one place, and every pick you make is sent to
                ESPN and counted only once ESPN confirms it.
              </span>
            </li>
          </ol>
        </div>

        <div className="lp-trust">
          <h2 className="lp-trust-head">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="var(--ok)"
                 strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M12 3l7.5 3v5.5c0 4.4-3.1 8.4-7.5 9.5-4.4-1.1-7.5-5.1-7.5-9.5V6z" />
            </svg>
            Your ESPN password never reaches this app
          </h2>
          <p>
            There is no account to create and no password to hand over. The
            bookmarklet runs on ESPN’s own page, where you are already signed in,
            and asks ESPN for the draft token its own draft room uses. What
            arrives here is that token plus the ids already sitting in your draft
            room’s address bar — league, team, member, season. Your ESPN login
            cookie stays in your browser.
          </p>
          <div className="lp-rule" />
          <p>
            That token is held for the draft, in a file only your own user can
            read, and dropped twelve hours later. It is what lets a crash
            mid-draft rejoin on its own instead of asking you for anything.
          </p>
          <div className="lp-rule" />
          <p>
            This window opens its own connection to ESPN’s draft socket, so draft
            here rather than in ESPN’s room — one team drafting from two sessions
            is a fight neither needs. Every pick goes back over that socket and
            your league sees an ordinary draft.
          </p>
        </div>
      </section>

      {/* -- pricing -- */}
      <section className="lp-sec">
        <div className="lp-lead">
          <p className="lp-cap">Pricing</p>
          <h2 className="lp-h2">Practise for nothing. Pay once, for the draft that counts.</h2>
        </div>
        <div className="lp-plans">
          <div className="lp-plan">
            <div className="lp-plan-head">
              <span className="lp-plan-name">Mock drafts</span>
              <span className="lp-plan-price mono">Free</span>
              <span className="lp-plan-unit">always</span>
            </div>
            <p className="lp-plan-blurb">
              The whole tool, with nothing held back. As many as you like.
            </p>
            <ul className="lp-plan-items">
              {['Every ranking and every recommendation',
                'Full player cards, history and comparables',
                'Picks sent to ESPN exactly as in a real draft'].map((t) => (
                <li key={t}>
                  <Tick />
                  <span>{t}</span>
                </li>
              ))}
            </ul>
            <button className="lp-plan-btn" onClick={toSetup}>Start a mock draft</button>
          </div>

          <div className="lp-plan lp-plan-paid">
            <div className="lp-plan-head">
              <span className="lp-plan-name">Your real draft</span>
              <span className="lp-plan-price mono">$4.99</span>
              <span className="lp-plan-unit">per draft</span>
            </div>
            <p className="lp-plan-blurb">
              One league, one draft night. No subscription, nothing to cancel.
            </p>
            <ul className="lp-plan-items">
              {['Everything in mocks, on the night it counts',
                'One price per draft, not per season',
                'Still no account and no password'].map((t) => (
                <li key={t}>
                  <Tick />
                  <span>{t}</span>
                </li>
              ))}
            </ul>
            <button className="lp-plan-btn" onClick={toSetup}>Use it for a real draft</button>
          </div>
        </div>
        {/* Said plainly rather than left for someone to discover: there is no
            payment integration in this build at all, so a page that implied a
            charge would be describing software that does not exist. */}
        <p className="lp-plans-note">
          Payment isn’t switched on yet — nothing on this page can charge you,
          and until it is, a real draft runs on the same free path a mock does.
          Both buttons take you to the setup above.
        </p>
      </section>

      <footer className="lp-foot">
        <span>
          Not affiliated with ESPN. Works with any ESPN fantasy football league
          whose draft room you can open.
        </span>
        <button className="lp-cta" onClick={toSetup}>Try a mock draft</button>
      </footer>

      {/* An operator's panel, not part of the pitch: it reports whether this
          machine's data has been refreshed and prints the command when it has
          not. It renders nothing unless the helper answers, so a visitor never
          sees it -- and whoever is running the helper still needs it on the one
          night it matters. */}
      <div className="lp-ops">
        <ReadinessStrip />
      </div>
    </main>
  )
}

/** The green check in the pricing lists. Inline because it is the only icon
 *  used more than once here, and a shared component beats six copies of the
 *  same path drifting apart. */
function Tick() {
  return (
    <svg className="lp-tick" width="11" height="11" viewBox="0 0 24 24" fill="none"
         stroke="var(--ok)" strokeWidth="3.4" strokeLinecap="round"
         strokeLinejoin="round" aria-hidden="true">
      <path d="M20 6L9 17l-5-5" />
    </svg>
  )
}
