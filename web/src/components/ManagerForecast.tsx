import type { Manager, ManagerHistory, SimBoard, SimBoardCell } from '../api'

interface ManagerForecastProps {
  board: SimBoard
  managers: Manager[]
  history: ManagerHistory[]
  onSelectPlayer: (playerId: string) => void
}

// Round-bucket keys ManagerHistory.shape carries, in draft order, with the
// label the card prints for each -- the same "early/mid/late" split
// api/main.py's _history_round_bucket cuts on.
const HISTORY_BUCKETS: { key: 'early' | 'mid' | 'late'; label: string }[] = [
  { key: 'early', label: 'R1-3' },
  { key: 'mid', label: 'R4-8' },
  { key: 'late', label: 'R9+' },
]

// Position order the shape row counts in -- matches the order --pos-* tokens
// are declared in (index.css) and the order pos-badge-* rules follow in
// App.css, so a card's shape row and the grid's border colors never
// disagree about which position reads as which hue.
const SHAPE_POSITIONS = ['QB', 'RB', 'WR', 'TE', 'K', 'DST'] as const

// How many rounds of "next picks" a card shows -- round 1-4 is what's still
// live in a manager's hands on draft night; rounds 5+ are what the
// positional shape row is for instead of a longer list nobody reads.
const NEXT_PICKS_SHOWN = 4

// The board's primary (alt_rank 0) prediction for every pick, grouped by
// slot and sorted round-first -- same "one predicted player per cell" the
// grid renders (DraftGrid's `byCell`), just keyed by slot alone since a
// manager's card has no round x manager matrix to place cells into.
function primaryPicksBySlot(cells: SimBoardCell[]): Map<number, SimBoardCell[]> {
  const map = new Map<number, SimBoardCell[]>()
  for (const c of cells) {
    if (c.alt_rank !== 0) continue
    const list = map.get(c.slot) ?? []
    list.push(c)
    map.set(c.slot, list)
  }
  for (const list of map.values()) list.sort((a, b) => a.round - b.round)
  return map
}

// Counts across a manager's WHOLE predicted draft (every round, not just the
// next-picks list) -- the number that answers "is he taking a QB early" even
// when that QB lands in round 6, past what the pick list shows.
function shapeCounts(picks: SimBoardCell[]): Partial<Record<string, number>> {
  const counts: Partial<Record<string, number>> = {}
  for (const p of picks) {
    if (!p.position) continue
    counts[p.position] = (counts[p.position] ?? 0) + 1
  }
  return counts
}

// The per-manager story pulled out of the grid: one card per team, in draft
// slot order, answering "who is this, what do they do, what are they about
// to take" without reading the whole round x manager matrix. Mounted as a
// sibling view to DraftGrid inside the same board.run/board.cells gate in
// DraftBoardPage, so it inherits that page's empty-state handling (no run,
// no league, a run that predates per-pick cells) instead of repeating it.
export default function ManagerForecast({ board, managers, history, onSelectPlayer }: ManagerForecastProps) {
  const bySlot = primaryPicksBySlot(board.cells)
  const byName = new Map(managers.map((m) => [m.manager, m]))
  const historyByName = new Map(history.map((h) => [h.manager, h]))
  const mySlot = board.run?.my_slot ?? null

  return (
    <div className="forecast-grid">
      {board.order.map((entry) => {
        const picks = bySlot.get(entry.slot) ?? []
        const next = picks.slice(0, NEXT_PICKS_SHOWN)
        const counts = shapeCounts(picks)
        const m = byName.get(entry.manager)
        const h = historyByName.get(entry.manager)
        const isMe = entry.slot === mySlot

        return (
          <article
            key={entry.slot}
            className={isMe ? 'forecast-card forecast-card-me' : 'forecast-card'}
          >
            <header className="forecast-card-header">
              <span className="slot-no">{entry.slot}</span>
              <h3 className="forecast-name">{entry.manager}</h3>
            </header>

            <div className="forecast-chips">
              {/* The model doesn't predict the user -- it plans for them.
                  Marked here rather than folded into the "personal
                  model"/"league average" chip below, since that pair
                  describes every OTHER manager's card too. */}
              {isMe && <span className="rail-chip rail-chip-you">Your plan</span>}
              {m && (
                <span
                  className={m.uses_personal ? 'rail-chip rail-chip-personal' : 'rail-chip rail-chip-pooled'}
                >
                  {m.uses_personal ? 'personal model' : 'league average'}
                </span>
              )}
            </div>

            {/* The evidence the forecast below rests on -- what this manager
                has actually done, not what the model guesses they'll do.
                Kept in its own quiet, monochrome panel (see .forecast-history
                in App.css) so it reads as fact next to the colored,
                probability-tagged prediction below it, never as more of the
                same guess. Omitted entirely (not an empty placeholder) when
                no history is imported for this manager at all -- see
                DraftBoardPage's fetchManagerHistory().catch(() => []). */}
            {h && (
              <section className="forecast-history">
                <div className="forecast-history-head">
                  <span className="forecast-history-label">Actual history</span>
                  <span className="forecast-history-meta">
                    {h.seasons} draft{h.seasons === 1 ? '' : 's'} &middot; {h.total_picks} picks
                  </span>
                </div>

                {h.first_rounders.length > 0 && (
                  <ol className="forecast-history-firsts">
                    {h.first_rounders.map((fr) => (
                      <li key={fr.season} className="forecast-history-first">
                        <span className="forecast-history-year mono">{fr.season}</span>
                        {fr.position && <span className="hist-badge">{fr.position}</span>}
                        <span className="forecast-history-name">
                          {fr.player_name ?? 'unidentified pick'}
                        </span>
                        {fr.keeper && <span className="forecast-history-keeper">kept</span>}
                      </li>
                    ))}
                  </ol>
                )}

                <div className="forecast-history-shape">
                  {HISTORY_BUCKETS.map(({ key, label }) => {
                    const counts = h.shape[key] ?? {}
                    const nonZero = SHAPE_POSITIONS.filter((p) => (counts[p] ?? 0) > 0)
                    return (
                      <div key={key} className="forecast-history-bucket">
                        <span className="forecast-history-bucket-label mono">{label}</span>
                        <span className="forecast-history-bucket-pills">
                          {nonZero.length === 0 ? (
                            <span className="forecast-history-bucket-empty">&mdash;</span>
                          ) : (
                            nonZero.map((p) => (
                              <span key={p} className="hist-badge">{p} {counts[p]}</span>
                            ))
                          )}
                        </span>
                      </div>
                    )
                  })}
                </div>
              </section>
            )}

            {/* Backend's own text already says "league average, not enough
                signal" verbatim when uses_personal is false (draft_model.py)
                -- rendered as-is, not re-worded, so this card never claims
                more confidence than the model has. */}
            <p className="forecast-summary">
              {m ? m.summary : 'No model fitted for this manager yet.'}
            </p>

            {next.length === 0 ? (
              <p className="forecast-empty">No predicted picks yet.</p>
            ) : (
              <ol className="forecast-picks">
                {next.map((cell) => {
                  const pos = (cell.position ?? '').toLowerCase()
                  return (
                    <li key={cell.overall_pick} className="forecast-pick">
                      <span className="forecast-pick-round mono">
                        {cell.round}.{cell.round_pick}
                      </span>
                      <button
                        type="button"
                        className="forecast-pick-btn"
                        style={pos ? { borderLeftColor: `var(--pos-${pos})` } : undefined}
                        onClick={() => onSelectPlayer(cell.player_id)}
                      >
                        {cell.position && (
                          <span className={`pos-badge pos-badge-${pos}`}>{cell.position}</span>
                        )}
                        <span className="forecast-pick-name">{cell.name}</span>
                        <span className="forecast-pick-team">{cell.team ?? ''}</span>
                      </button>
                      {/* Three, mutually exclusive trailing states, matching
                          DraftGrid's own certain/mine distinction: a pick
                          already made (certain) needs no odds at all; the
                          user's own future pick is a plan, not a forecast,
                          so it gets a label instead of a probability that
                          would overstate what the model is doing; everyone
                          else's future pick is the one place a probability
                          belongs. */}
                      {cell.certain ? (
                        <span className="forecast-pick-tag forecast-pick-drafted">drafted</span>
                      ) : isMe ? (
                        <span className="forecast-pick-tag forecast-pick-plan">plan</span>
                      ) : (
                        <span className="forecast-pick-tag mono">{Math.round(cell.prob * 100)}%</span>
                      )}
                    </li>
                  )
                })}
              </ol>
            )}

            <div className="forecast-shape">
              <span className="forecast-shape-label">Draft shape</span>
              <div className="forecast-shape-row">
                {SHAPE_POSITIONS.map((pos) => {
                  const count = counts[pos] ?? 0
                  return (
                    <span
                      key={pos}
                      className={
                        count > 0
                          ? `pos-badge pos-badge-${pos.toLowerCase()}`
                          : `pos-badge pos-badge-${pos.toLowerCase()} forecast-shape-zero`
                      }
                    >
                      {pos} {count}
                    </span>
                  )
                })}
              </div>
            </div>
          </article>
        )
      })}
    </div>
  )
}
