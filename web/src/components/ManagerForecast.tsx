import type { Manager, ManagerHistory, ManagerTendencies, SimBoard, SimBoardCell } from '../api'

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

// A card is 260px wide at its narrowest, so the tendency rows show the head
// of each ranked list rather than all of it: the position someone opens with
// is a fact, the position they opened with once in six years is trivia.
const OPENERS_SHOWN = 2
const REACH_POSITIONS_SHOWN = 2

// Bucket key -> the label the shape rows already print for it, so the reach
// split and the positional shape read as the same three rounds of the draft.
const BUCKET_LABEL = new Map(HISTORY_BUCKETS.map((b) => [b.key as string, b.label]))

// "+4.2" / "-3.4". The sign carries the whole meaning of every gap number on
// the card (earlier than the board vs later), so it is never dropped, and a
// value that rounds to zero still shows which side of zero it came from.
function signed(value: number, digits = 1): string {
  return `${value < 0 ? '-' : '+'}${Math.abs(value).toFixed(digits)}`
}

// "RB in 5 of 6 drafts" -- and a second opener when they have one, since
// "RB in 4, WR in 2" is a different manager from "RB in 4" alone.
function opensText(t: ManagerTendencies): string | null {
  const top = t.first_pick.slice(0, OPENERS_SHOWN)
  if (top.length === 0) return null
  const rest = top.slice(1).map((f) => `${f.position} in ${f.drafts}`)
  return [`${top[0].position} in ${top[0].drafts} of ${top[0].of} drafts`, ...rest].join(', ')
}

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
// sibling view to DraftGrid whenever there's a league to show (same
// board.order.length > 0 gate DraftGrid has always rendered under, run or
// no run), so it degrades the same honest way DraftGrid's own empty
// skeleton does -- a card with no cells yet still shows the manager's
// identity, real draft history and tendency text, just an empty picks list.
export default function ManagerForecast({ board, managers, history, onSelectPlayer }: ManagerForecastProps) {
  const bySlot = primaryPicksBySlot(board.cells)
  const byName = new Map(managers.map((m) => [m.manager, m]))
  const historyByName = new Map(history.map((h) => [h.manager, h]))
  const mySlot = board.run?.my_slot ?? null

  return (
    <div className="forecast-grid">
      {board.order.map((entry) => {
        const picks = bySlot.get(entry.slot) ?? []
        // predict_board reports already-made picks as certain and simulates
        // every REMAINING round to the end of the draft, not just the first
        // four -- filtering before slicing is what keeps this list showing
        // "what's next" instead of getting stuck on round 1-4 (all `certain`)
        // for the rest of draft night once the real draft passes round 4.
        const upcoming = picks.filter((p) => !p.certain)
        const next = upcoming.slice(0, NEXT_PICKS_SHOWN)
        // The one or two picks immediately behind "next" -- cheap context
        // (one row) for what this manager just did, once a draft is live and
        // the early rounds have scrolled out of the upcoming list above.
        const justTook = picks.filter((p) => p.certain).slice(-2)
        const counts = shapeCounts(picks)
        const m = byName.get(entry.manager)
        const h = historyByName.get(entry.manager)
        const t = h?.tendencies ?? null
        const opens = t ? opensText(t) : null
        const reaches = (t?.reach_by_position ?? []).filter((p) => p.mean_gap > 0)
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

                {/* Measured tendencies -- counted from this manager's real
                    picks, so they say something specific about them even
                    while the forecast below runs on league-average
                    coefficients. Every row is omitted rather than zero-filled
                    when the data behind it is missing. */}
                {t && (
                  <dl className="forecast-tendency">
                    {opens && (
                      <div className="forecast-tendency-row">
                        <dt className="forecast-tendency-label">Opens</dt>
                        <dd className="forecast-tendency-value">{opens}</dd>
                      </div>
                    )}

                    {t.reach && (
                      <div className="forecast-tendency-row">
                        <dt className="forecast-tendency-label">Board</dt>
                        <dd className="forecast-tendency-value">
                          {/* The direction is spelled out in the text, not
                              parked in a hover title: the badges beside it
                              carry their own titles, which shadow a title on
                              this row, so hovering "R9+ +8.4" would never have
                              explained what the sign meant. */}
                          <span className="mono">{signed(t.reach.mean_gap)}</span>
                          <span>{t.reach.mean_gap < 0 ? 'behind board' : 'ahead of board'}</span>
                          <span className="forecast-tendency-note">
                            over {t.reach.n} picks, no K/DST
                          </span>
                          {t.reach_by_bucket.map((b) => (
                            <span
                              key={b.bucket}
                              className="hist-badge"
                              title={`${b.n} picks`}
                            >
                              {BUCKET_LABEL.get(b.bucket) ?? b.bucket} {signed(b.mean_gap)}
                            </span>
                          ))}
                        </dd>
                      </div>
                    )}

                    {/* Only positions they take AHEAD of the board: a
                        negative gap here means "waits on it", which the
                        Board row above already covers in aggregate. */}
                    {reaches.length > 0 && (
                      <div className="forecast-tendency-row">
                        <dt className="forecast-tendency-label">Reaches</dt>
                        <dd className="forecast-tendency-value">
                          {reaches.slice(0, REACH_POSITIONS_SHOWN).map((p) => (
                            <span key={p.position} className="hist-badge" title={`${p.n} picks`}>
                              {p.position} {signed(p.mean_gap)}
                            </span>
                          ))}
                        </dd>
                      </div>
                    )}

                    {t.first_at_position.length > 0 && (
                      <div className="forecast-tendency-row">
                        <dt className="forecast-tendency-label">First</dt>
                        <dd className="forecast-tendency-value">
                          <span className="forecast-tendency-note">
                            avg round they take one
                          </span>
                          {t.first_at_position.map((f) => (
                            // `f.drafts` alone, never "x of h.seasons": drafts
                            // comes from the precomputed table and seasons is
                            // counted live, so an import without a refit would
                            // print two denominators on one card.
                            <span
                              key={f.position}
                              className="hist-badge"
                              title={`${f.drafts} draft${f.drafts === 1 ? '' : 's'}`}
                            >
                              {f.position} R{f.mean_round.toFixed(1)}
                            </span>
                          ))}
                        </dd>
                      </div>
                    )}
                  </dl>
                )}

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
                    // Named distinctly from the outer `counts` (this card's
                    // PREDICTED shape, computed above from `picks`) -- this
                    // one is a bucket of this manager's ACTUAL history, a
                    // different thing that happened to share a name.
                    const bucketCounts = h.shape[key] ?? {}
                    const nonZero = SHAPE_POSITIONS.filter((p) => (bucketCounts[p] ?? 0) > 0)
                    return (
                      <div key={key} className="forecast-history-bucket">
                        <span className="forecast-history-bucket-label mono">{label}</span>
                        <span className="forecast-history-bucket-pills">
                          {nonZero.length === 0 ? (
                            <span className="forecast-history-bucket-empty">&mdash;</span>
                          ) : (
                            nonZero.map((p) => (
                              <span key={p} className="hist-badge">{p} {bucketCounts[p]}</span>
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

            {justTook.length > 0 && (
              <p className="forecast-recent">
                Just took: {justTook.map((c) => `${c.name} (${c.round}.${c.round_pick})`).join(', ')}
              </p>
            )}

            {picks.length === 0 ? (
              <p className="forecast-empty">No predicted picks yet.</p>
            ) : upcoming.length === 0 ? (
              // Every round for this slot came back `certain` -- their draft
              // is done, not merely unpredicted. Distinct from the message
              // above so a finished manager doesn't read as a data gap.
              <p className="forecast-empty">Draft complete for this manager.</p>
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
                      {/* `next` is already filtered to `!certain` above, so
                          only two trailing states are reachable here: the
                          user's own future pick is a plan, not a forecast,
                          so it gets a label instead of a probability that
                          would overstate what the model is doing; everyone
                          else's future pick is the one place a probability
                          belongs. (A pick already made -- `certain` -- gets
                          no row here at all; see `justTook` above for that.) */}
                      {isMe ? (
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
