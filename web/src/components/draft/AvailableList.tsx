import { useState, type ReactNode } from 'react'
import type { LiveCandidate, Player } from '../../api'
import { riskTone } from './tone'

// duplicated from RosterPanel.tsx/DraftBoardGrid.tsx (unexported in both) --
// same precedent as PlayerCard.tsx's depthSlotLabel/sosLabel: a four-line
// pure function isn't worth a shared module between four views now.
function posBadge(position: string): ReactNode {
  return <span className={`pos-badge pos-badge-${position.toLowerCase()}`}>{position}</span>
}

// Same shape as RankingsPanel.tsx's fmtRank -- one decimal only when the
// aggregate ADP isn't a whole number, dash when the player has no market
// coverage at all. Used for both market ranks in this table: `market_rank`
// (the five-source median, scoring/market.py) and `espn_ppr_rank` (a single
// source, integral in practice, but formatted the same way so the two
// columns line up digit for digit).
function fmtRank(n: number | null): string {
  if (n === null) return '—'
  return Number.isInteger(n) ? String(n) : n.toFixed(1)
}

const POSITIONS = ['ALL', 'QB', 'RB', 'WR', 'TE', 'K', 'DST']

// -- sorting ---------------------------------------------------------------
//
// WHY THIS TABLE NO LONGER SHOWS `gain_now`, `vor_points` OR `fills`:
// the model is unchanged. `gain_now` still ranks this list server-side (see
// scoring/gain.py and api/live.py's _recompute), it is still what `c.rank`
// -- the `#` column and the default sort here -- counts down, and it still
// picks and orders the three recommendation cards above this table, which
// explain the pick in a sentence. What changed is that the table stopped
// showing the working: "Gain now", "Over repl" and "Fills" were three
// columns nobody could read without a paragraph of explanation, so they
// were deliberately deleted at the owner's request. They were NOT lost in a
// refactor -- do not "restore" them. If a number here ever needs defending
// again, the place for it is TopThree's sentence, not a fourth column of
// jargon. (Removing `fills` also removed the accent treatment that marked
// an open starter slot; that signal lives on in RosterPanel, which is where
// a reader looks for "what do I still need" anyway. It is deliberately not
// re-drawn here.)

type SortKey = 'rank' | 'pos' | 'player' | 'proj' | 'lasts' | 'adp' | 'espn'
type SortDir = 'asc' | 'desc'

// The direction a column gets on its FIRST click -- "best first" for that
// particular column, which is not the same arrow everywhere: rank/ADP/ESPN
// are ranks (1 is best, so ascending), proj/lasts are quantities (bigger is
// better, so descending). Clicking an already-sorted header flips it, so
// both directions stay reachable on every column; this only decides which
// one you land on without having to click twice.
const NATURAL_DIR: Record<SortKey, SortDir> = {
  rank: 'asc', pos: 'asc', player: 'asc', proj: 'desc', lasts: 'desc', adp: 'asc', espn: 'asc',
}

// Position sorts in the pill row's order (QB, RB, WR, TE, K, DST), not
// alphabetically -- alphabetical would open with DST and K, which is the
// order nobody thinks about a draft in, and it would disagree with the
// filter pills sitting directly above the header. Anything the server sends
// that isn't in the pill list lands after everything that is.
const POS_ORDER = POSITIONS.slice(1)
function posIndex(position: string): number {
  const i = POS_ORDER.indexOf(position)
  return i === -1 ? POS_ORDER.length : i
}

// The one value a column sorts on. `null` means "this player has no such
// number" and is handled by the comparator, never coerced to 0 -- a player
// no market source covers is not ADP 0, i.e. the best pick on the board.
// Every key reads from exactly the same place the cell renders from, so
// what you see sorted is what you see printed.
function sortValue(
  key: SortKey, c: LiveCandidate, player: Player | undefined,
): number | string | null {
  switch (key) {
    case 'rank': return c.rank
    case 'pos': return posIndex(c.position)
    case 'player': return (player?.name ?? c.player_id).toLowerCase()
    case 'proj': return c.proj_points
    case 'lasts': return c.survive_pct
    case 'adp': return player?.market_rank ?? null
    case 'espn': return player?.espn_ppr_rank ?? null
  }
}

// NULLS ALWAYS LAST, in both directions -- deliberately not "smallest" or
// "largest". A missing ESPN rank is an absence, not a value: treating it as
// -Infinity would put every uncovered rookie above Ja'Marr Chase on one
// click and below him on the next, and either way the dashes would be
// interleaved through the rows you were actually trying to compare.
// Sinking them keeps the comparable rows contiguous at the top and makes
// the flip button do one predictable thing.
//
// Ties (and null-vs-null) fall back to the server's own `rank`, ascending,
// which is never null and is unique per row -- so the sort is total and
// stable-looking regardless of the engine, and sorting by POS, say, leaves
// each position group in board order rather than in arrival order. The
// tie-break is NOT flipped with `dir`: within one position, or one ADP
// value, board order is the right order either way.
function compareRows(
  a: LiveCandidate, b: LiveCandidate,
  key: SortKey, dir: SortDir, players: Record<string, Player>,
): number {
  const av = sortValue(key, a, players[a.player_id])
  const bv = sortValue(key, b, players[b.player_id])
  if (av === null || bv === null) {
    if (av !== null) return -1
    if (bv !== null) return 1
    return a.rank - b.rank
  }
  const d = typeof av === 'string' && typeof bv === 'string'
    ? av.localeCompare(bv)
    : (av as number) - (bv as number)
  if (d !== 0) return dir === 'asc' ? d : -d
  return a.rank - b.rank
}

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
  // TopThree's hint names -- it is here because the "Lasts" column is a
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

// The ranked available pool: search + position filter above a table that
// opens in the server's own `gain_now` order (`#`) and can be re-sorted by
// any column from its header.
//
// Client-side sorting is confined to THIS component's own copy of the list
// on purpose. `candidates` is DraftRoom's `state.candidates`, handed to
// TopThree as well; the three recommendation cards must always be the
// server's top three in the server's order no matter what this table is
// sorted by, so nothing here may reorder the prop itself. `.filter()`
// already returns a fresh array and `.sort()` below only ever touches that
// -- the shared array is never mutated.
//
// Both filters are client-side per the task brief ("the server sends the
// whole ranked list") -- the pool tops out in the low hundreds, cheap
// enough to filter AND sort on every keystroke without debouncing or memos.
export default function AvailableList({
  candidates, players, onDraft, isMyTurn, horizonLabel, onOpenPlayer,
}: AvailableListProps) {
  const [search, setSearch] = useState('')
  const [pos, setPos] = useState('ALL')
  // Opens on the server's ranking, which is the whole point of the list --
  // any other default would hide the model's answer behind a click.
  const [sort, setSort] = useState<{ key: SortKey; dir: SortDir }>({ key: 'rank', dir: 'asc' })

  const q = search.trim().toLowerCase()
  const visible = candidates.filter((c) => {
    if (pos !== 'ALL' && c.position !== pos) return false
    if (!q) return true
    const player = players[c.player_id]
    const haystack = `${player?.name ?? c.player_id} ${player?.team ?? ''}`.toLowerCase()
    return haystack.includes(q)
  })
  // `visible` is already a fresh array from `.filter()`; sorting it in place
  // cannot reach `candidates`. Written as a copy anyway so that stays true
  // if the filter is ever short-circuited away for the unfiltered case.
  const rows = visible.slice().sort((a, b) => compareRows(a, b, sort.key, sort.dir, players))

  function toggleSort(key: SortKey): void {
    setSort((prev) => (prev.key === key
      ? { key, dir: prev.dir === 'asc' ? 'desc' : 'asc' }
      : { key, dir: NATURAL_DIR[key] }))
  }

  // A sortable header. The clickable thing is a real <button> inside the
  // <th>, not a click handler on the cell: this table is driven under a
  // pick clock and has to stay tabbable, and `aria-sort` on the header is
  // how a screen reader gets the same "sorted by, this way" the caret gives
  // everyone else.
  function sortableTh(key: SortKey, label: ReactNode, className?: string): ReactNode {
    const active = sort.key === key
    return (
      <th
        className={`${className ?? ''}${active ? ' is-sorted' : ''}`.trim() || undefined}
        aria-sort={active ? (sort.dir === 'asc' ? 'ascending' : 'descending') : 'none'}
      >
        <button type="button" className="avail-th-btn" onClick={() => toggleSort(key)}>
          <span>{label}</span>
          {/* The caret is ALWAYS in the markup, transparent until the column
              is the sorted one (or hovered) -- rendering it only when active
              made every header jump sideways by its own width on each sort
              click, which under a pick clock reads as the table twitching.
              Idle, it points the way that column's first click will sort
              (NATURAL_DIR), so hovering PROJ shows ▼ and hovering ADP ▲ --
              the affordance and the promise in one glyph. */}
          <span
            className={`avail-sort${active ? '' : ' is-idle'}`}
            aria-hidden="true"
          >
            {(active ? sort.dir : NATURAL_DIR[key]) === 'asc' ? '▲' : '▼'}
          </span>
        </button>
      </th>
    )
  }

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
            {sortableTh('rank', '#', 'avail-col-rank')}
            {sortableTh('pos', 'Pos', 'avail-col-pos')}
            {sortableTh('player', 'Player')}
            {sortableTh('proj', 'Proj', 'avail-col-num')}
            {/* Header names the horizon when there is one; the column is
                "chance he is still on the board at that pick". Without the
                label a reader takes it for "lasts to my next pick", which
                at a wheel or a short gap is a different pick entirely. */}
            {sortableTh(
              'lasts',
              horizonLabel !== null ? `Lasts to ${horizonLabel}` : 'Lasts',
              'avail-col-num',
            )}
            {sortableTh('adp', 'ADP', 'avail-col-num')}
            {/* ESPN's own PPR rank, always on screen next to this board's
                `#` and the market's ADP -- the owner asked to be able to see
                where ESPN has a player against where this board has him,
                without going into the profile for it. */}
            {sortableTh('espn', 'ESPN', 'avail-col-num')}
            <th className="avail-col-btn" />
          </tr>
        </thead>
        <tbody>
          {rows.map((c) => {
            const player = players[c.player_id]
            return (
              <tr key={c.player_id}>
                <td className="avail-col-rank mono">{c.rank}</td>
                <td className="avail-col-pos">{posBadge(c.position)}</td>
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
                <td className="avail-col-num mono avail-adp">{fmtRank(player?.espn_ppr_rank ?? null)}</td>
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
          {rows.length === 0 && (
            <tr>
              {/* 8 = the seven columns above plus the draft button's. */}
              <td colSpan={8} className="avail-empty">
                {candidates.length === 0 ? 'No candidates yet.' : 'No players match this filter.'}
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  )
}
