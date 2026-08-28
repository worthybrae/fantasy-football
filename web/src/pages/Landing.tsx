import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { PaymentRequired } from '../api'
import Paywall from '../components/Paywall'
import { connectEspnAccount, connectWithToken, ensureLiveSession, fetchConnectProgress, fetchLiveState,
         fetchUpcomingDrafts,
         type ConnectProgress, type LiveState, type TokenConnectParams,
         type UpcomingDrafts } from '../api'
import { readAccount, rememberAccount } from '../lib/accountCache'
import { useDocumentMeta } from '../lib/documentMeta'
import SetupWizard, { ACCOUNT_CHANNEL, CHANNEL_ACK, CHANNEL_CONNECTED } from '../components/SetupWizard'
import ConnectScreen from '../components/ConnectScreen'
import Benefits from '../components/Benefits'
import Explainer from '../components/Explainer'
import DemoRoom from '../components/DemoRoom'
import Welcome from '../components/Welcome'
import Dashboard, { previewingSignedOut } from '../components/Dashboard'
// This page's own stylesheet, not App.css: see the header comment in it for
// why, and for why every class below is `lp-` prefixed.
import '../landing.css'

// How long the popup waits for a wizard to claim its connect before deciding
// there is no wizard and settling in as the app. A BroadcastChannel hop is
// milliseconds; the margin is for a wizard tab the browser has throttled.
const ACK_GRACE_MS = 1500

function announceConnect(): void {
  if (!('BroadcastChannel' in window)) return
  let channel: BroadcastChannel
  try {
    channel = new BroadcastChannel(ACCOUNT_CHANNEL)
  } catch {
    return
  }
  const settle = window.setTimeout(() => channel.close(), ACK_GRACE_MS)
  channel.onmessage = (event: MessageEvent) => {
    if (event.data !== CHANNEL_ACK) return
    window.clearTimeout(settle)
    channel.close()
    if (!window.opener) return
    try {
      window.close()
    } catch { /* a browser that will not let a popup close itself */ }
  }
  channel.postMessage(CHANNEL_CONNECTED)
}

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
type Gate = 'connecting' | 'idle' | 'live' | 'paywall'

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

// The other thing the bookmarklet can deliver: the ESPN account session, from
// a click anywhere on ESPN rather than inside a draft room. It rides in the
// hash for the same reason the token does -- browsers do not send fragments
// to servers, so it reaches this page without passing through an access log
// on the way -- and this page hands it to the helper and wipes it from the
// address bar in the same breath.
function sessionFromHash(hash: string) {
  const p = new URLSearchParams(hash.replace(/^#/, ''))
  const swid = p.get('swid')
  const s2 = p.get('s2')
  if (!swid || !s2) return null
  return { swid, s2 }
}

// The chip, the drag test, the copy fallback and the no-drag sentence all
// live in components/BookmarkChip.tsx, moved there verbatim when the setup
// wizard became their second consumer.

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
// PROOF_ROWS is gone with the panel it fed. It was four hand-measured rows
// standing in for a draft board; the page renders a real one now (DemoRoom),
// and a fabricated example beside a live board would be the weaker of the two
// claiming equal weight.

// Four things a ranked list structurally cannot tell you. Each `proof` is a
// reading, not an adjective; `note` is the part that makes the reading mean
// something, and in two cases it is the caveat rather than the boast.

export default function Landing() {
  const [gate, setGate] = useState<Gate>('idle')

  // THE ONE WAY INTO THE CONNECT GATE. Opening it starts two requests at
  // once -- the connect POST and the progress poll (the effects below) --
  // and they must carry the same room cookie, so the cookie is fetched
  // and awaited here before either exists. Every path that wants the gate
  // (the bookmarklet hash, the Dashboard's join, a retry) goes through
  // this, so no future path can skip it. A failure is not fatal: the
  // connect POST mints a cookie of its own, and only the first poll or two
  // would miss the room.
  const openConnectGate = useCallback(() => {
    ensureLiveSession().catch(() => {}).then(() => setGate('connecting'))
  }, [])
  // The draft the server refused for want of payment, if it did. Held rather
  // than read off `paramsRef` so the paywall names the league the SERVER
  // named -- the two agree today, and a screen that charges for a draft
  // should not depend on that staying true.
  const [owed, setOwed] = useState<{ leagueId: string; season: number } | null>(null)
  // What the room is showing: a draft going on now, or one from the archive
  // being replayed because none is. Held here for one reason -- the welcome
  // card says "running now", and it must stop saying it when that stops
  // being true.
  const [mode, setMode] = useState<'live' | 'replay' | 'none'>('none')
  // Bumped when something on the page needs an account it does not have --
  // the archive button under the corpus band. The welcome card reads it and
  // raises itself with the reason.
  const [askedAt, setAskedAt] = useState(0)
  // The account connect, which runs on its own clock: it is not a draft and
  // has no progress screen. `connected` is a counter rather than a flag so
  // the drafts card can be told to re-read (a key change) each time one
  // lands, without this page holding the list itself.
  const [connecting, setConnecting] = useState(false)
  const [connected, setConnected] = useState(0)
  // The ESPN account this browser can act as, and the leagues it holds.
  // Seeded from this tab's last answer (lib/accountCache.ts) so a return to
  // this page -- Archive -> Drafts, above all -- opens on the right page in
  // its first frame. Null only when the tab has never been answered, and
  // null paints NOTHING (see the render below): the pitch flashed at
  // somebody signed in and then swapped for the dashboard is the exact
  // stutter this state exists to prevent.
  const [account, setAccount] = useState<UpcomingDrafts | null>(readAccount)
  // The guided setup, open over the page. While it is up the dashboard is
  // held back even once the account connects -- the wizard's own finished
  // screen is showing the same fact, and the page swapping underneath it
  // would yank the dialog's ground away mid-sentence.
  const [wizard, setWizard] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [progress, setProgress] = useState<ConnectProgress | null>(null)
  const [live, setLive] = useState<LiveState | null>(null)
  const [sessionReady, setSessionReady] = useState(false)
  // Bumped by Retry. Every connect-owned piece of state is keyed off it, so a
  // retry starts from a blank screen rather than from the last one's rows.
  const [attempt, setAttempt] = useState(0)
  const navigate = useNavigate()
  useDocumentMeta({
    title: 'ESPN Draft Assist – a live draft assistant for ESPN fantasy football',
    description: 'A live draft assistant for ESPN fantasy football. It sits beside your ESPN draft room and ranks the board from real recorded ESPN mock drafts.',
    canonical: 'https://espnfantasydraft.com/',
  })

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

  // BACK FROM STRIPE, IN THE POPUP. `success_url` lands here with `?paid=1`;
  // if this window was opened by the page that started the checkout, its only
  // job is to say so and get out of the way. The opener re-checks with the
  // server before believing it -- this only saves it a poll.
  useEffect(() => {
    if (!new URLSearchParams(window.location.search).has('paid')) return
    if (!window.opener) return
    try {
      window.opener.postMessage('billing:paid', window.location.origin)
      window.close()
    } catch { /* a browser that will not let a popup close itself */ }
  }, [])

  useEffect(() => {
    // Arrived from the bookmarklet? It may carry two things, and they are
    // independent: an account session to connect, and a draft to join.
    const hash = window.location.hash
    const params = tokenFromHash(hash)
    const session = sessionFromHash(hash)
    if (params || session) {
      // FIRST, before either is used: out of the address bar. It should not
      // sit there, be bookmarked, be screenshotted, or survive a refresh into
      // a duplicate connect -- and one of these two values is an account.
      window.history.replaceState(null, '', window.location.pathname)
    }
    if (session) {
      // Connecting is not joining. It stores the session, hands this browser
      // an httpOnly cookie and answers with the leagues -- which the drafts
      // card picks up on its own poll, so nothing here has to hold the list.
      setConnecting(true)
      connectEspnAccount(session.swid, session.s2)
        .then(() => {
          setConnected((n) => n + 1)
          // Tell any tab sitting on the setup wizard that the click landed.
          // The wizard confirms with its own probe -- this only collapses
          // the seconds it would otherwise spend waiting on the next poll.
          //
          // Then listen for its answer. This window is the popup the
          // bookmarklet opened from ESPN, and if a wizard is out there, the
          // user is mid-setup with the app already open in a tab: a second
          // copy of it in a second window is the thing they did not ask for
          // and cannot tell from the first. A wizard that hears the announce
          // acks; an ack within the grace period means "the tab you started
          // in has this now", and this window gets out of the way. Silence
          // means no wizard -- the bookmark clicked on its own, with the app
          // closed -- and this window IS the app, so it stays.
          //
          // `window.opener` is the guard on closing at all: only a window
          // something else opened may close itself, and only that window is
          // a duplicate. A tab the user navigated here by hand has no opener
          // and never closes.
          announceConnect()
        })
        .catch((e) => setError(e instanceof Error ? e.message : String(e)))
        .finally(() => setConnecting(false))
    }
    if (params) {
      paramsRef.current = params
      openConnectGate()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
        // A refusal for money is not a failure to report -- the request was
        // fine and the draft costs $9.99 (api/billing.py). It gets a screen
        // with a price on it rather than a red error under a stage list.
        if (e instanceof PaymentRequired) {
          setOwed({ leagueId: e.leagueId, season: e.season })
          setGate('paywall')
          return
        }
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
          // A few seconds' tolerance, so this shares the gate poll's own read
          // rather than opening a second one for the same answer.
          fetchLiveState(5_000).then((s) => { if (!cancelled) setLive(s) }).catch(() => {})
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
  //
  // A SESSION IS NOT A DRAFT. `active` and `token_received` only say a
  // connect once happened in this server process: a token that was minted,
  // used for nothing, and left behind reads as both for as long as the
  // process lives. Measured on this deployment -- active, token_received,
  // listener_alive false, stale, zero picks -- and the page was offering
  // "your draft is synced, go to the board" over a session with no draft
  // behind it and no listener attached. So the test is whether the thing is
  // ALIVE: a listener on the socket, or picks that actually landed.
  useEffect(() => {
    if (gate !== 'idle') return
    let cancelled = false
    const poll = async () => {
      try {
        const state = await fetchLiveState()
        const running = state.active
          && (state.listener_alive || state.draft_started || state.picks_made > 0)
        if (!cancelled && running) setGate('live')
      } catch {
        /* helper not up yet; the page simply stays on the landing view */
      }
    }
    poll()
    const id = setInterval(poll, 2500)
    return () => { cancelled = true; clearInterval(id) }
  }, [gate])

  // WHO IS LOOKING, AND THEREFORE WHICH PAGE THIS IS. `/api/espn/drafts` is
  // the authoritative answer and deliberately not the cheaper
  // `/api/espn/custody` probe: a STORED session is not a WORKING one -- ESPN
  // invalidates a cookie when the user signs out anywhere, and nothing local
  // can see that happen. This endpoint finds out the only way anybody can, by
  // asking ESPN, so a page that renders the dashboard is a page whose session
  // was good a moment ago.
  //
  // Re-read on a slow poll as well as on a connect (`connected` counts those):
  // the server caches for two minutes so this mostly costs nothing, and what
  // it buys is a draft flipping to LIVE while somebody is looking at the page,
  // which is the one moment this list most needs to be right.
  useEffect(() => {
    let cancelled = false
    const read = async () => {
      try {
        const body = await fetchUpcomingDrafts()
        rememberAccount(body)
        if (!cancelled) setAccount(body)
      } catch {
        // The helper being down is not a signed-out session. Leave whatever
        // is on screen alone rather than throwing a reader back to the pitch
        // over one failed request.
        if (!cancelled && account === null) setAccount({ connected: false, leagues: [] })
      }
    }
    read()
    const id = setInterval(read, 60_000)
    return () => { cancelled = true; clearInterval(id) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [connected])

  // A join from the league list. The token was minted server-side rather than
  // by a bookmarklet on an ESPN tab, and from here on nothing knows the
  // difference: same params, same ref, same `connecting` gate, same progress
  // screen and same Retry. That is the whole reason the mint is its own
  // endpoint -- one connect path, two ways of getting a token to it.
  const joinDraft = useCallback((params: TokenConnectParams) => {
    paramsRef.current = params
    setError(null)
    setProgress(null)
    openConnectGate()
  }, [openConnectGate])

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
  // the honest thing a CTA can do is walk you through installing it. Every
  // CTA opens the guided setup (SetupWizard): the same chip, the same three
  // steps as the block further down the page, but one at a time and with the
  // last step watching for the click and finishing itself. The setup block
  // stays on the page as the readable reference for anyone who would rather
  // scroll than be walked.
  const toSetup = useCallback(() => setWizard(true), [])

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

  // Paid for, but not yet. The connect is still in memory -- that is the
  // whole reason checkout happens in a popup (see Paywall) -- so paying
  // re-runs it rather than sending the user back through the bookmark.
  if (gate === 'paywall' && owed !== null) {
    return (
      <Paywall
        leagueId={owed.leagueId}
        season={owed.season}
        onPaid={() => { setOwed(null); retry() }}
        onBack={() => { paramsRef.current = null; setOwed(null); setGate('idle') }}
      />
    )
  }

  // SIGNED IN IS A DIFFERENT PAGE, not the same page with a card on top.
  // Somebody who has connected an account has already been sold; what they
  // came back for is a list of drafts and a way into one. So the pitch, the
  // demo room and the welcome card are not rendered at all here -- see
  // Dashboard, which is the whole page for this reader.
  //
  // Below the connect screen deliberately: a connect in flight is a connect
  // in flight whoever is signed in, and the progress screen owns the page
  // while it runs.
  // A mock room is a ROUTE now (/room/:leagueId, pages/WaitingRoomPage.tsx),
  // not a view of this page: the URL names the room, refresh keeps it, and
  // the back button leaves it. Its join navigates back here with the token
  // in the hash -- the exact door the bookmarklet uses -- so this page still
  // owns the one connect path.
  if (account?.connected && !previewingSignedOut() && !wizard) {
    return (
      <Dashboard
        leagues={account.leagues}
        onJoin={joinDraft}
        onOpenRoom={(leagueId) => navigate(
          `/room/${encodeURIComponent(leagueId)}`
          + (account.season ? `?season=${account.season}` : ''))}
      />
    )
  }

  // Not yet known which of the two pages this is. An empty frame for the
  // probe's round trip -- a local lookup for a stranger, so a few
  // milliseconds; the server's cached league list for a returning reader --
  // beats guessing wrong and swapping. Only ever seen on a tab's first visit:
  // every later one is seeded from the last answer.
  if (account === null) {
    return <main className="lp" aria-busy="true" />
  }

  return (
    <main className="lp">
      {/* NO MARKETING BAR. The room below brings its own top bar -- wordmark,
          tabs, league line, pick counter, status pill -- and two bars stacked
          would put the product's chrome under an advertisement for it. The
          price moved to the ask, where somebody is actually deciding, and the
          /mocks link to the footer, where the other side routes live. */}

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

      {/* The seconds between the bookmarklet delivering a session and the
          probe coming back with the leagues it unlocked. Worth a line: it is
          the one moment this page is about to become a different page, and
          silence there reads as the click having failed. */}
      {gate === 'idle' && connecting && (
        <section className="lp-drafts">
          <p className="lp-cap">Connecting your ESPN account…</p>
        </section>
      )}

      {/* -- the argument, and the proof of it, above the fold -- */}
      {/* THE ROOM IS THE PAGE, and it is the first screen: a visitor lands
          inside the product with a real ESPN mock draft running in it. No
          headline above it, no lede explaining what they are about to see --
          the room explains itself faster than a paragraph does, and a
          paragraph would push it below the fold.

          The pitch happens after: the ask is docked at the bottom of that
          screen, and everything that argues the case is further down for
          whoever wants it. */}
      {/* `live` puts the way back to a running draft in the ROOM'S OWN top
          bar, where the rest of this page's chrome lives. It used to be a
          band across the top of the page, which pushed the whole room down
          the moment it appeared and read as an alert about something that
          had gone wrong. A link in the bar is the same offer without the
          shove. */}
      <DemoRoom onMode={setMode} live={gate === 'live'} />

      {/* The brand, and the way in. Over the room rather than instead of it,
          and it never disappears -- dismissing moves it to the corner (see
          Welcome). Rendered after the room so it stacks above without a
          z-index argument. */}
      <Welcome onStart={toSetup} live={mode === 'live'} askedAt={askedAt} />

      {/* WHAT IT KNOWS, DRAWN. This was four cards of prose about the
          model -- three hundred words a reader who has just watched the room
          run itself is not going to read, every claim of which they had to
          take on trust. Each panel now draws the thing it claims, in the
          shape the room draws it, and arrives as it is scrolled to. */}
      <Benefits onNeedAccount={() => setAskedAt((n) => n + 1)} />

      {/* The page in plain words, for readers and crawlers alike: the only
          <h1>, and the FAQ (also FAQPage structured data). Below the room
          and the drawn figures, which make the case faster for anyone who
          watched them. */}
      <Explainer onStart={toSetup} />

      <footer className="lp-foot">
        <span>
          Not affiliated with ESPN. Works with any ESPN fantasy football league
          whose draft room you can open.
          {' '}
          {/* The reading room for drafts already played. It lost its place in
              the top bar when the room took that bar over, and this is where
              it belonged anyway: a side route, not part of the pitch. */}
          <Link to="/mocks" className="lp-bar-link">See mock drafts</Link>
          {' or '}
          {/* A plain anchor, not Link: /adp is rendered by FastAPI from the
              corpus and is not a route this app's router knows. It also
              wants a real page load so the crawler follows it. Until this
              link existed the ADP pages were orphans -- served, sitemapped,
              and reachable only by typing the address. */}
          <a href="/adp" className="lp-bar-link">ADP from those drafts</a>.
        </span>
      </footer>

      {/* The guided setup, over everything. Rendered last so it stacks
          without a z-index argument, exactly as Welcome does. `onConnected`
          remembers the answer and seeds the page's own state, so the
          dashboard is already painted behind the wizard's finished screen
          the moment "See your drafts" closes it. */}
      {wizard && (
        <SetupWizard
          onClose={() => setWizard(false)}
          onConnected={(body) => {
            rememberAccount(body)
            setAccount(body)
          }}
          onDone={() => setWizard(false)}
        />
      )}
    </main>
  )
}
