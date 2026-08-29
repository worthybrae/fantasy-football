import { useMemo, useState } from 'react'
import {
  FAVORITES_MAX, FAVORITES_MIN, saveFavorites, type Player,
} from '../api'

// YOUR GUYS -- the five to twenty-five players you actually want on your
// team, picked once, away from a pick clock.
//
// The plan targets them (scoring/plan.py: a favourite clears a lower
// availability bar and carries a bonus), which is the whole reason this
// screen exists. It is deliberately a DASHBOARD screen and not a room one:
// the room is thirty seconds a pick, and a list of twenty-five names is not
// a decision anybody makes in thirty seconds.
//
// The universe is `/api/players` -- the same board the room ranks -- so a
// name here is a name the server can validate. Its 422 for an id that is not
// on the board is therefore unreachable from this screen, which is the point:
// the picker cannot express a list the server would refuse.

const POSITIONS = ['ALL', 'QB', 'RB', 'WR', 'TE', 'K', 'DST']

interface FavoritesPickerProps {
  /** The board, as `fetchPlayers()` serves it. */
  players: Player[]
  /** The saved list, when this is an edit rather than an onboarding. */
  initial?: string[]
  /** The saved order, handed back once the server has taken it. */
  onSaved: (players: string[]) => void
  /** Absent during onboarding: there is nothing to go back to. */
  onCancel?: () => void
}

export default function FavoritesPicker({
  players, initial = [], onSaved, onCancel,
}: FavoritesPickerProps) {
  // ORDER IS THE STATE. An array, not a Set: the server stores the order and
  // the plan reads it, so "who did you want most" survives the round trip.
  const [chosen, setChosen] = useState<string[]>(initial)
  const [search, setSearch] = useState('')
  const [pos, setPos] = useState('ALL')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const byId = useMemo(
    () => new Map(players.map((p) => [p.player_id, p])), [players])
  const picked = useMemo(() => new Set(chosen), [chosen])
  const full = chosen.length >= FAVORITES_MAX

  const q = search.trim().toLowerCase()
  // Board order (`rank`) is the order the list arrives in, which is the one
  // a reader can scan: alphabetical would open on a defense.
  const shown = players.filter((p) => {
    if (pos !== 'ALL' && p.position !== pos) return false
    if (!q) return true
    return `${p.name} ${p.team}`.toLowerCase().includes(q)
  })

  function toggle(playerId: string): void {
    setError(null)
    setChosen((prev) => (prev.includes(playerId)
      ? prev.filter((id) => id !== playerId)
      : prev.length >= FAVORITES_MAX ? prev : [...prev, playerId]))
  }

  async function save(): Promise<void> {
    setSaving(true)
    setError(null)
    try {
      onSaved(await saveFavorites(chosen))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  // The one rule this screen enforces, stated where it is enforced. The
  // server enforces it too (422), and the counter exists so nobody has to
  // meet that refusal to learn the rule.
  const ready = chosen.length >= FAVORITES_MIN && chosen.length <= FAVORITES_MAX
  const shortBy = FAVORITES_MIN - chosen.length

  return (
    <div className="fav">
      <div className="fav-head">
        <div className="fav-head-copy">
          <h3 className="fav-title">Pick your guys</h3>
          <p className="fav-note">
            The players you want on your team this year. The draft room stars
            them, and the plan reaches for them a round earlier than it would
            reach for anybody else.
          </p>
        </div>
        <div className="fav-actions">
          {/* The count and the rule in one place, and the only thing between
              a reader and the Save button. A bare "Save" that refuses to
              work teaches nothing; this says what it is waiting for.

              The ratio alone still made the reader do the arithmetic, so
              the sentence beside it does it for them: how many more before
              Save turns on, or that the list is full and a name has to come
              off before another goes on. Nothing at all in between, where
              the ratio is the whole story. */}
          <span className={`fav-count mono${ready ? ' is-ready' : ''}`}>
            {chosen.length} / {FAVORITES_MIN}-{FAVORITES_MAX}
          </span>
          {shortBy > 0 && (
            <span className="fav-need">
              {shortBy} more to save
            </span>
          )}
          {full && <span className="fav-need">Full — drop one to add another</span>}
          {onCancel && (
            <button type="button" className="fav-cancel" onClick={onCancel}
                    disabled={saving}>
              Cancel
            </button>
          )}
          <button type="button" className="fav-save" onClick={() => void save()}
                  disabled={!ready || saving}>
            {saving ? 'Saving…' : 'Save'}
          </button>
        </div>
      </div>

      {error !== null && <p className="fav-error" role="alert">{error}</p>}

      {/* THE CHOSEN, IN ORDER, above the board rather than beside it: this is
          the thing being built, and a reader adding a twentieth name should
          not have to look somewhere else to see the other nineteen. */}
      <ol className="fav-chosen">
        {chosen.map((id, i) => {
          const player = byId.get(id)
          const name = player?.name ?? id
          return (
            <li key={id}>
              {/* Named, because a row of twenty-five identical "Remove"
                  buttons tells a screen reader nothing about which name it
                  is about to drop. */}
              <button type="button" className="fav-chip" onClick={() => toggle(id)}
                      aria-label={`Remove ${name}`} title={`Remove ${name}`}>
                <span className="fav-chip-ord mono">{i + 1}</span>
                <span className="fav-chip-name">{name}</span>
                <span className="fav-chip-x" aria-hidden="true">×</span>
              </button>
            </li>
          )
        })}
        {chosen.length === 0 && (
          <li className="fav-chosen-empty">
            Nobody yet — pick at least {FAVORITES_MIN}.
          </li>
        )}
      </ol>

      <div className="fav-toolbar">
        <input
          type="text"
          className="fav-search"
          placeholder="Search players"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          aria-label="Search players"
        />
        <div className="fav-pills">
          {POSITIONS.map((p) => (
            <button
              key={p}
              type="button"
              className={`avail-pill${pos === p ? ' is-active' : ''}`}
              onClick={() => setPos(p)}
            >
              {p}
            </button>
          ))}
        </div>
      </div>

      <ul className="fav-grid">
        {shown.map((p) => {
          const on = picked.has(p.player_id)
          return (
            <li key={p.player_id}>
              {/* Full, and not one of the chosen: the row is disabled rather
                  than clickable-and-refused. A control that does nothing on
                  click is a control that looks broken. */}
              {/* Named for a screen reader rather than left to read as
                  "Player RB · DET ★": the row's four parts are one control,
                  and `aria-pressed` already carries whether he is in. */}
              <button
                type="button"
                className={`fav-row${on ? ' is-on' : ''}`}
                aria-label={p.name}
                aria-pressed={on}
                disabled={full && !on}
                onClick={() => toggle(p.player_id)}
              >
                {p.headshot ? (
                  <img className="fav-shot" src={p.headshot} alt="" loading="lazy" />
                ) : (
                  <span className={`pos-badge pos-badge-${p.position.toLowerCase()}`}>
                    {p.position}
                  </span>
                )}
                <span className="fav-row-id">
                  <span className="fav-row-name">{p.name}</span>
                  <span className="fav-row-sub mono">{p.position} · {p.team}</span>
                </span>
                <span className="fav-row-mark" aria-hidden="true">{on ? '★' : '+'}</span>
              </button>
            </li>
          )
        })}
        {/* Named, so the reader can see WHICH of the two filters emptied the
            board and undo that one. "No players match that" was true of a
            search, a position pill and both at once. */}
        {shown.length === 0 && (
          <li className="fav-empty">
            {q
              ? `Nobody on the board matches “${search.trim()}”${pos === 'ALL' ? '' : ` at ${pos}`}.`
              : `No ${pos} on the board.`}
          </li>
        )}
      </ul>
    </div>
  )
}
