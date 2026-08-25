import type { ConnectProgress, ConnectStage, LiveState } from '../api'
import { pickNumberFor } from './draft/pickOrder'

// The screen between clicking the bookmarklet and the draft room appearing.
//
// It exists because that gap is not short: measured against the real
// database, a connect to the owner's own league blocks for 32-35s (fit_all
// alone is 27.5-30.7s of it) and a connect to a fresh mock for 8.6s. All of
// that used to happen behind one spinner and the word "Connecting".
//
// Every row here is work api/live.py genuinely does, in the order it does it,
// carrying the value that step actually discovered -- see
// ConnectProgress/_STAGE_LABELS there. Nothing on this screen is timed,
// padded, or held open to be readable: a stage that finishes in 20ms flashes
// past, which is honest, and the "progress" bar measures stages completed
// (a number that exists) rather than time elapsed against a guess (one that
// does not).
//
// Four states, all four drawn in .design/onboarding: connecting (Main),
// finished (Ready), finished with something degraded (Degraded), and stopped
// (Expired).

const SCORING_LABEL: Record<string, string> = {
  ppr: 'PPR', half: 'Half', std: 'Std',
}

// The order a roster shape reads in, matching the rail's own
// (DraftRoom.tsx's POSITION_ORDER) with FLEX where the room shows it.
const ROSTER_ORDER = ['QB', 'RB', 'WR', 'TE', 'FLEX', 'K', 'DST']

function rosterShape(facts: ConnectProgress['facts']): string | null {
  if (!facts.starters) return null
  const counts = ROSTER_ORDER
    .map((pos) => (pos === 'FLEX' ? facts.flex_slots ?? 0 : facts.starters?.[pos] ?? 0))
    .filter((n) => n > 0)
  return counts.length ? counts.join('/') : null
}

// -- status glyphs. Shape, never colour alone: a check, a spinner, a dim
// dot, a warning triangle and a cross are five different silhouettes, and
// each carries its own aria-label so the row reads correctly aloud too. --

function StageGlyph({ status }: { status: ConnectStage['status'] }) {
  if (status === 'ok') {
    return (
      <svg className="cs-glyph cs-glyph-ok" viewBox="0 0 24 24" role="img" aria-label="done">
        <path d="M20 6L9 17l-5-5" />
      </svg>
    )
  }
  if (status === 'running') {
    return (
      <svg className="cs-glyph cs-glyph-run" viewBox="0 0 24 24" role="img" aria-label="in progress">
        <path d="M12 3a9 9 0 0 1 9 9" />
      </svg>
    )
  }
  if (status === 'warn') {
    return (
      <svg className="cs-glyph cs-glyph-warn" viewBox="0 0 24 24" role="img" aria-label="done with a warning">
        <path d="M12 8v5" />
        <path d="M12 17h.01" />
        <path d="M10.3 3.9 2.4 18a1.9 1.9 0 0 0 1.7 2.9h15.8a1.9 1.9 0 0 0 1.7-2.9L13.7 3.9a1.9 1.9 0 0 0-3.4 0z" />
      </svg>
    )
  }
  if (status === 'failed') {
    return (
      <svg className="cs-glyph cs-glyph-fail" viewBox="0 0 24 24" role="img" aria-label="failed">
        <path d="M18 6L6 18" />
        <path d="M6 6l12 12" />
      </svg>
    )
  }
  return <span className="cs-dot" role="img" aria-label="waiting" />
}

function StageRow({ stage }: { stage: ConnectStage }) {
  return (
    <li className={`cs-row cs-row-${stage.status}`}>
      <span className="cs-row-glyph"><StageGlyph status={stage.status} /></span>
      <span className="cs-row-label">{stage.label}</span>
      {stage.value !== null && (
        <span className="cs-row-value mono">{stage.value}</span>
      )}
    </li>
  )
}

function StageList({ stages }: { stages: ConnectStage[] }) {
  return <ul className="cs-stages">{stages.map((s) => <StageRow key={s.key} stage={s} />)}</ul>
}

// Elapsed, from the server's own monotonic clock. Whole seconds while it
// runs (the poll is 400ms, and a tenths figure that jumps by 0.4 reads as
// broken); the exact tenth once it has stopped, because then it is a
// measurement rather than a ticker.
function elapsedLabel(ms: number, running: boolean): string {
  return running ? `${Math.floor(ms / 1000)}s` : `${(ms / 1000).toFixed(1)}s`
}

export interface ConnectScreenProps {
  progress: ConnectProgress
  /** The live state, fetched once the connect settles -- only for the pick
   *  clock, which no connect stage discovers. Null until then. */
  live: LiveState | null
  /** True once POST /api/live/connect-token has returned: the session exists
   *  and the room works, whatever the trailing stages are still doing. */
  sessionReady: boolean
  /** The POST's own rejection message, when it rejected. */
  error: string | null
  onEnter: () => void
  onRetry: () => void
  onBack: () => void
}

export default function ConnectScreen(props: ConnectScreenProps) {
  const { progress, sessionReady, error, onEnter } = props
  const { stages } = progress
  const failed = progress.phase === 'failed' || error !== null
  const done = stages.filter((s) => s.status !== 'pending' && s.status !== 'running').length
  const running = progress.phase === 'connecting' && !failed
  const warned = stages.filter((s) => s.status === 'warn')

  // The bar is stages completed over stages planned -- both real counts. It
  // is deliberately NOT a time estimate: the stages differ by three orders of
  // magnitude (6ms to 30s), so any time-based bar would be a fabrication.
  const pct = stages.length ? (done / stages.length) * 100 : 0

  // The one line an assistive reader gets, rather than the whole list
  // re-announcing itself every 400ms: whatever just landed.
  const latest = [...stages].reverse().find((s) => s.value !== null)

  if (failed) return <Failed {...props} />
  if (progress.phase === 'ready' && warned.length > 0) return <Degraded {...props} warned={warned} />
  if (progress.phase === 'ready') return <Ready {...props} />

  return (
    <main className="connect-screen">
      <div
        className="cs-bar"
        role="progressbar"
        aria-label="Connect progress"
        aria-valuemin={0}
        aria-valuemax={stages.length}
        aria-valuenow={done}
      >
        <span className="cs-bar-fill" style={{ width: `${pct}%` }} />
      </div>

      <div className="cs-col">
        <header className="cs-head">
          <p className="cs-cap">ESPN Draft Assist</p>
          <h1 className="cs-title">Getting your draft ready</h1>
          <p className="cs-sub">
            Reading your league straight from ESPN. This runs once per draft.
          </p>
        </header>

        <div className="sr-only" aria-live="polite">
          {latest ? `${latest.label}: ${latest.value}` : 'Connecting'}
        </div>

        <StageList stages={stages} />

        <footer className="cs-foot">
          <span className="mono cs-elapsed">{elapsedLabel(progress.elapsed_ms, running)}</span>
          <span className="cs-footnote">Your ESPN login never leaves your browser.</span>
          {/* Never a trap: the moment the session exists the room is usable,
              even while the socket handshake or the first ranking is still
              outstanding (both land after the connect returns). */}
          {sessionReady && (
            <button type="button" className="cs-link" onClick={onEnter}>
              Enter the draft room
            </button>
          )}
        </footer>
      </div>
    </main>
  )
}

// -- finished, everything as intended -------------------------------------

function Ready({ progress, live, onEnter }: ConnectScreenProps) {
  const { facts } = progress
  const roster = rosterShape(facts)
  const clock = live?.ms_remaining != null ? `${Math.round(live.ms_remaining / 1000)}s` : null

  // Only facts something actually discovered. A cell nobody filled is
  // dropped, never rendered as a zero or a dash that reads like a real value.
  const cells: { label: string; value: string; hint?: string }[] = []
  if (facts.teams != null) cells.push({ label: 'Teams', value: String(facts.teams) })
  if (facts.scoring_format) {
    cells.push({ label: 'Scoring', value: SCORING_LABEL[facts.scoring_format] })
  }
  if (facts.rounds != null) cells.push({ label: 'Rounds', value: String(facts.rounds) })
  if (facts.players != null) cells.push({ label: 'Players', value: String(facts.players) })
  if (roster) {
    cells.push({ label: 'Roster', value: roster, hint: ROSTER_ORDER.join('/') })
  }
  if (facts.bench != null) cells.push({ label: 'Bench', value: String(facts.bench) })
  if (facts.managers != null) {
    cells.push({
      label: 'History',
      value: facts.managers ? `${facts.managers} mgrs` : 'none',
      hint: facts.managers
        ? `${facts.managers} manager models fitted from ${facts.seasons ?? 0} `
          + 'seasons of this league’s own drafts'
        : 'no imported draft history — opponents follow the market prior',
    })
  }
  if (clock) cells.push({ label: 'Clock', value: clock })
  // The grid is four columns and its 1px gaps are the border colour showing
  // through, so a part-filled last row leaves a lighter hole where a cell
  // would be. Filled with blanks rather than by padding the facts out with
  // values nobody discovered.
  const blanks = (4 - (cells.length % 4)) % 4

  return (
    <main className="connect-screen">
      <div className="cs-bar cs-bar-done" />
      <div className="cs-col">
        <header className="cs-head">
          <p className="cs-cap cs-cap-ok">
            <svg className="cs-glyph cs-glyph-ok" viewBox="0 0 24 24" aria-hidden="true">
              <path d="M20 6L9 17l-5-5" />
            </svg>
            Ready
          </p>
          {/* ESPN's own display name for the owner's team, when the team-name
              fetch answered and the slot is known. Nothing is invented in its
              place -- and not the bare word "Ready" either, which the eyebrow
              above already says. */}
          <h1 className="cs-title">{facts.my_team || 'Your draft is ready'}</h1>
          <p className="cs-sub">
            You draft from here now. Leave the ESPN tab alone — this window holds
            the draft session.
          </p>
        </header>

        {cells.length > 0 && (
          <dl className="cs-facts">
            {cells.map((c) => (
              <div className="cs-fact" key={c.label} title={c.hint}>
                <dt className="cs-cap">{c.label}</dt>
                <dd className="mono cs-fact-value">{c.value}</dd>
              </div>
            ))}
            {Array.from({ length: blanks }, (_, i) => (
              <div className="cs-fact" key={`blank${i}`} aria-hidden="true" />
            ))}
          </dl>
        )}

        <Seat facts={facts} />

        <div className="cs-actions">
          <button type="button" className="cs-btn cs-btn-primary" onClick={onEnter} autoFocus>
            Enter the draft room
          </button>
          <span className="mono cs-elapsed">{elapsedLabel(progress.elapsed_ms, false)}</span>
        </div>
      </div>
    </main>
  )
}

// The seat, drawn rather than described -- and only when it is genuinely
// known. A slot the connect could not resolve says so instead of colouring
// in a guess: a wrong slot attributes every pick to the wrong manager.
function Seat({ facts }: { facts: ConnectProgress['facts'] }) {
  const teams = facts.teams ?? 0
  const slot = facts.my_slot
  if (!teams) return null
  if (slot == null) {
    return (
      <section className="cs-seat">
        <p className="cs-cap">Your seat</p>
        <p className="cs-seat-note">
          Not known yet — ESPN names your team on the draft socket when the
          room opens.
        </p>
      </section>
    )
  }

  // Real snake arithmetic, the same helper the room's clock uses.
  const picks = facts.rounds
    ? Array.from({ length: facts.rounds }, (_, r) => pickNumberFor(r, slot, teams))
    : []
  const shown = picks.slice(0, 4)

  return (
    <section className="cs-seat">
      <p className="cs-cap">Your seat</p>
      <ol className="cs-seats" aria-label={`You pick ${slot} of ${teams}`}>
        {Array.from({ length: teams }, (_, i) => i + 1).map((n) => (
          <li key={n} className={`mono cs-seat-cell${n === slot ? ' cs-seat-mine' : ''}`}>
            {n}
            {n === slot && <span className="sr-only"> — your seat</span>}
          </li>
        ))}
      </ol>
      {shown.length > 0 && (
        <p className="cs-seat-note">
          Picks {shown.join(', ')}
          {picks.length > shown.length && ` — and ${picks.length - shown.length} more`}.
        </p>
      )}
    </section>
  )
}

// -- finished, but something fell back ------------------------------------

function Degraded(props: ConnectScreenProps & { warned: ConnectStage[] }) {
  const { progress, warned, onEnter, onRetry } = props
  const { facts } = progress
  const settingsWarn = warned.some((s) => s.key === 'settings')
  const saved = [facts.teams && `${facts.teams} teams`,
                 facts.scoring_format && SCORING_LABEL[facts.scoring_format],
                 facts.rounds && `${facts.rounds} rounds`].filter(Boolean).join(' · ')
  // An unresolved slot warns twice -- once for the slot itself and once for
  // the ranking that is waiting on it -- and they are the same sentence. Both
  // ROWS stay (both stages really did end that way); the explanation is given
  // once.
  const notes = warned.filter(
    (s) => !(s.key === 'ranking' && warned.some((w) => w.key === 'slot')))

  return (
    <main className="connect-screen">
      <div className="cs-bar cs-bar-done" />
      <div className="cs-col">
        <header className="cs-head">
          <p className="cs-cap">ESPN Draft Assist</p>
          <h1 className="cs-title">Ready, with one thing to know</h1>
          <p className="cs-sub">
            The draft is connected and you can pick. {notes.length === 1
              ? 'One step did not answer.'
              : `${notes.length} steps did not answer.`}
          </p>
        </header>

        <StageList stages={progress.stages} />

        {notes.map((stage) => (
          <div className="cs-note" key={stage.key} role="note">
            <svg className="cs-glyph cs-glyph-warn" viewBox="0 0 24 24" aria-hidden="true">
              <path d="M12 8v5" />
              <path d="M12 17h.01" />
              <path d="M10.3 3.9 2.4 18a1.9 1.9 0 0 0 1.7 2.9h15.8a1.9 1.9 0 0 0 1.7-2.9L13.7 3.9a1.9 1.9 0 0 0-3.4 0z" />
            </svg>
            <div className="cs-note-body">
              <p className="cs-note-title">{warnTitle(stage.key)}</p>
              <p className="cs-note-text">
                {warnBody(stage.key, saved, facts.settings_source)}
              </p>
            </div>
          </div>
        ))}

        <div className="cs-actions">
          {settingsWarn ? (
            <>
              <button type="button" className="cs-btn cs-btn-primary" onClick={onRetry}>
                Retry settings
              </button>
              <button type="button" className="cs-btn" onClick={onEnter}>
                Continue anyway
              </button>
            </>
          ) : (
            <>
              <button type="button" className="cs-btn cs-btn-primary" onClick={onEnter} autoFocus>
                Enter the draft room
              </button>
              <button type="button" className="cs-btn" onClick={onRetry}>
                Reconnect
              </button>
            </>
          )}
          <span className="mono cs-elapsed">{elapsedLabel(progress.elapsed_ms, false)}</span>
        </div>
      </div>
    </main>
  )
}

function warnTitle(key: string): string {
  if (key === 'settings') return "ESPN didn't return this league's settings"
  if (key === 'teams') return "ESPN didn't return the team names"
  if (key === 'slot') return 'Your draft slot is not known yet'
  if (key === 'ranking') return 'The ranked list is waiting on your slot'
  return 'One step did not answer'
}

function warnBody(key: string, saved: string, source?: string): string {
  if (key === 'settings') {
    // Which fallback, named. "Saved" and "default" are different claims: the
    // second is a generic league the owner has never seen, and it is the one
    // any league this app provisioned actually lands on.
    const where = source === 'saved'
      ? 'the roster saved on this machine for this league'
      : 'the built-in default roster'
    return `Falling back to ${where}${saved ? `: ${saved}` : ''}. `
      + 'That roster decides the round count and every replacement level, so if '
      + 'this draft uses a different one the ranking is built for the wrong shape.'
  }
  if (key === 'teams') {
    return 'The board\'s columns will read "Team 1", "Team 2" and so on. Nothing '
      + 'else is affected — picks still land in the right column.'
  }
  return 'ESPN names your team on the draft socket, which happens when the draft '
    + 'room opens. Until then the board shows every available player by value '
    + 'over replacement, and the ranking built for your roster arrives as soon '
    + 'as your seat is known.'
}

// -- stopped ---------------------------------------------------------------

function Failed({ progress, error, onRetry, onBack }: ConnectScreenProps) {
  // Only the rows that were actually reached. A pending row on a screen that
  // has stopped would claim work is still coming.
  const reached = progress.stages.filter((s) => s.status !== 'pending')
  const failure = progress.error
  const detail = failure?.detail ?? error ?? 'The connect stopped.'
  const expired = /token/i.test(detail)

  return (
    <main className="connect-screen">
      <div className="cs-bar cs-bar-fail" />
      <div className="cs-col">
        <header className="cs-head">
          <p className="cs-cap cs-cap-fail">
            <svg className="cs-glyph cs-glyph-fail" viewBox="0 0 24 24" aria-hidden="true">
              <circle cx="12" cy="12" r="9" />
              <path d="M15 9l-6 6" />
              <path d="M9 9l6 6" />
            </svg>
            Couldn’t connect
          </p>
          <h1 className="cs-title">
            {expired ? 'That draft link has expired' : 'The connect stopped'}
          </h1>
          <p className="cs-sub" role="alert">{detail}</p>
        </header>

        {reached.length > 0 && <StageList stages={reached} />}

        <div className="cs-steps">
          <p className="cs-cap">To reconnect</p>
          <ol>
            <li>
              <span className="mono cs-step-n">1</span>
              <span>Go back to your ESPN draft tab — it is still open and still yours.</span>
            </li>
            <li>
              <span className="mono cs-step-n">2</span>
              <span>
                Click the <strong>🏈 Draft Assistant</strong> bookmark again. It mints a
                fresh token.
              </span>
            </li>
            <li>
              <span className="mono cs-step-n">3</span>
              <span>This window picks up where it left off.</span>
            </li>
          </ol>
          {/* The server's own hint, unless it is the socket's -- that one
              says exactly what steps 1 and 2 above already say, and a
              recovery box that repeats itself reads like filler. */}
          {failure?.hint && failure.stage !== 'socket' && (
            <p className="cs-note-text">{failure.hint}</p>
          )}
          <div className="cs-actions">
            <button type="button" className="cs-btn cs-btn-primary" onClick={onRetry} autoFocus>
              Try again
            </button>
            <button type="button" className="cs-btn" onClick={onBack}>
              Back to setup
            </button>
          </div>
        </div>
      </div>
    </main>
  )
}
