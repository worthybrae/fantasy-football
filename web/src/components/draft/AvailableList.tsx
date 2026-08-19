import { useState, type ReactNode } from 'react'
import type { LiveCandidate, Player } from '../../api'
import { riskTone } from './tone'

// duplicated from RosterPanel.tsx/DraftBoardGrid.tsx (unexported in both) --
// same precedent as PlayerCard.tsx's depthSlotLabel/sosLabel: a four-line
// pure function isn't worth a shared module between four views now.
function posBadge(position: string): ReactNode {
  return <span className={`pos-badge pos-badge-${position.toLowerCase()}`}>{position}</span>
}

// Signed for display: `gain_now`/`vor_points` can be negative (waiting
// genuinely beats taking him -- e.g. the next-best survivor at his
// position already outvalues him), and a bare template literal would print
// "+-4" for that case. Positive gets an explicit "+" (nothing else on this
// screen implies sign the way a raw number does); zero and negative print
// as-is. `null` (no roster to rank this pick for yet -- my_slot not
// resolved, see LiveCandidate's own comment in api.ts) renders as a dash,
// never as "0" or "+0" -- either would read as a real, computed zero gain.
function fmtSigned(n: number | null): string {
  if (n === null) return '—'
  const r = Math.round(n)
  return r > 0 ? `+${r}` : `${r}`
}

// Same shape as RankingsPanel.tsx's fmtRank -- one decimal only when the
// aggregate ADP isn't a whole number, dash when the player has no market
// coverage at all.
function fmtRank(n: number | null): string {
  if (n === null) return '—'
  return Number.isInteger(n) ? String(n) : n.toFixed(1)
}

// Whether `fills` names a slot in the actual starting lineup -- FLEX
// counts, since a flex slot still starts. `BENCH`, gain.py's `—` ("no slot
// left at all, even the bench is full"), and `null` (no roster to fill a
// slot on yet -- my_slot not resolved) all read muted: the server already
// decided which is which (need_kind/fills_slot in scoring/gain.py) for the
// first two, and null is simply not an answer at all. TopThree.tsx applies
// the identical rule to its own three-figure row for the same reason
// RosterPanel colors an open starter slot and not an open bench one -- one
// meaning, drawn the same way everywhere it appears.
function fillsIsOpenSlot(fills: string | null): boolean {
  return fills !== null && fills !== 'BENCH' && fills !== '—'
}

const POSITIONS = ['ALL', 'QB', 'RB', 'WR', 'TE', 'K', 'DST']

interface AvailableListProps {
  candidates: LiveCandidate[]
  players: Record<string, Player>
  onDraft: (c: LiveCandidate) => void
  // Not in the task brief's own signature for this component -- added
  // because "the draft button is disabled unless it is your turn" (the
  // brief's own rule) has nowhere else to come from: this component has no
  // access to `state`, only to the ranked list and the join table. DraftRoom
  // computes it once (on_the_clock === my_slot, the same test ClockPanel's
  // youAreUp already uses) and hands it down, so both this list and
  // TopThree gate their buttons on the exact same boolean rather than each
  // re-deriving "is it my turn" from state fields they don't have.
  isMyTurn: boolean
  // The pick the server measured this list against ("pick 18", "the end of
  // the draft"), or null when there is no gain-ranked list yet. Same value
  // TopThree's hint names -- it is here because the "He lasts" column is a
  // probability of surviving to THAT pick, not to your immediately-next
  // one (scoring/draft_sim.horizon_picks skips turns too close to measure),
  // and an unlabelled 0% reads as the wrong claim.
  horizonLabel: string | null
  // Opens the player's profile over the room (DraftRoom's PlayerOverlay).
  // The whole row is not the target -- only the name -- because every other
  // cell in this row is a number the eye is comparing down a column, and the
  // last cell is the Draft button. One deliberate target, nowhere near the
  // irreversible one.
  onOpenPlayer: (c: LiveCandidate) => void
}

// The ranked available pool: search + position filter above a table sorted
// by `gain_now` (the server's own order -- never re-sorted client-side).
// Both filters are client-side per the task brief ("the server sends the
// whole ranked list") -- the pool tops out in the low hundreds, cheap
// enough to filter on every keystroke without debouncing.
export default function AvailableList({
  candidates, players, onDraft, isMyTurn, horizonLabel, onOpenPlayer,
}: AvailableListProps) {
  const [search, setSearch] = useState('')
  const [pos, setPos] = useState('ALL')

  const q = search.trim().toLowerCase()
  const visible = candidates.filter((c) => {
    if (pos !== 'ALL' && c.position !== pos) return false
    if (!q) return true
    const player = players[c.player_id]
    const haystack = `${player?.name ?? c.player_id} ${player?.team ?? ''}`.toLowerCase()
    return haystack.includes(q)
  })

  return (
    <div className="avail">
      <div className="avail-toolbar">
        <input
          type="text"
          className="avail-search"
          placeholder="Search players"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          aria-label="Search available players"
        />
        <div className="avail-pills">
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

      <table className="avail-table">
        <thead>
          <tr>
            <th className="avail-col-rank">#</th>
            <th className="avail-col-pos">Pos</th>
            <th>Player</th>
            <th className="avail-col-num">Proj</th>
            <th className="avail-col-num">Over repl</th>
            <th className="avail-col-num">Gain now</th>
            {/* Header names the horizon when there is one; the column is
                "chance he is still on the board at that pick". Without the
                label a reader takes it for "lasts to my next pick", which
                at a wheel or a short gap is a different pick entirely. */}
            <th className="avail-col-num">
              {horizonLabel !== null ? `Lasts to ${horizonLabel}` : 'He lasts'}
            </th>
            <th className="avail-col-num">ADP</th>
            <th className="avail-col-fills">Fills</th>
            <th className="avail-col-btn" />
          </tr>
        </thead>
        <tbody>
          {visible.map((c) => {
            const player = players[c.player_id]
            return (
              <tr key={c.player_id}>
                <td className="avail-col-rank mono">{c.rank}</td>
                <td>{posBadge(c.position)}</td>
                <td>
                  {/* A button, not a link: this opens an overlay over the
                      room, and an <a href> here would offer a navigation
                      that no longer happens on click. The board grid keeps
                      its real href for exactly the opposite reason -- see
                      DraftBoardGrid.tsx. */}
                  <button
                    type="button"
                    className="avail-name-btn"
                    onClick={() => onOpenPlayer(c)}
                    title="Open profile"
                  >
                    <span className="avail-name">{player?.name ?? c.player_id}</span>
                  </button>
                  {player && (
                    <span className="avail-meta mono">
                      {player.team} · BYE {player.bye ?? '—'}
                    </span>
                  )}
                </td>
                <td className="avail-col-num mono avail-proj">{Math.round(c.proj_points)}</td>
                <td className="avail-col-num mono avail-vor">{fmtSigned(c.vor_points)}</td>
                {/* The one column that decides the pick -- see the module
                    comment and the task brief's own framing: vor_points can
                    be the biggest number on the board and still be the
                    wrong reason to draft someone, if the next player at his
                    position is nearly as good. Full emphasis here, muted
                    two columns to its left, is the argument made visually. */}
                <td className="avail-col-num mono avail-gain">{fmtSigned(c.gain_now)}</td>
                {/* null survive_pct (no roster to survive FOR yet) gets no
                    riskTone color at all -- riskTone's red/amber/green ramp
                    is a claim about a real probability, and coloring a dash
                    would imply one exists. */}
                <td
                  className="avail-col-num mono"
                  style={c.survive_pct === null ? undefined : { color: riskTone(c.survive_pct) }}
                >
                  {c.survive_pct === null ? '—' : `${Math.round(c.survive_pct)}%`}
                </td>
                <td className="avail-col-num mono avail-adp">{fmtRank(player?.market_rank ?? null)}</td>
                <td className="avail-col-fills">
                  <span className={`avail-fills${fillsIsOpenSlot(c.fills) ? ' is-open' : ''}`}>
                    {c.fills ?? '—'}
                  </span>
                </td>
                <td className="avail-col-btn">
                  <button
                    type="button"
                    className="avail-draft-btn"
                    disabled={!isMyTurn}
                    title={isMyTurn ? undefined : 'Not your turn yet'}
                    onClick={() => onDraft(c)}
                  >
                    Draft
                  </button>
                </td>
              </tr>
            )
          })}
          {visible.length === 0 && (
            <tr>
              <td colSpan={10} className="avail-empty">
                {candidates.length === 0 ? 'No candidates yet.' : 'No players match this filter.'}
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  )
}
