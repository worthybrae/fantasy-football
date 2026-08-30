import {
  memo, useCallback, useEffect, useLayoutEffect, useRef, useState,
  type ReactNode,
} from 'react'
import { fetchMarketOverview, type LiveCandidate, type Player } from '../../api'
import { positionTip, TIP_DELAY_MS } from './AvailableList'
import { LastsChip } from './TakeNowCard'

// THE BOARD, WITH FOUR THINGS ON A ROW.
//
// The cheat sheet's table has fourteen columns and every one of them earns
// its place for a reader who wants them. This is the same board for the
// reader who does not: who he is, where he plays, will he be there, and is he
// worth taking now. Nothing is re-ranked and nothing new is computed -- the
// order is ESPN's own (`rank`, the order the server built the list in) and
// both figures are the payload's, printed under the words the vocabulary
// table settled on.
//
// The two columns the cheat sheet calls Lasts and Edge are called "Will he be
// there?" and "Worth grabbing now" here, and that is the whole difference:
// same number, same tooltip's worth of explanation behind it, asked as the
// question the reader actually has.

function posBadge(position: string): ReactNode {
  return <span className={`pos-badge pos-badge-${position.toLowerCase()}`}>{position}</span>
}

const POSITIONS = ['ALL', 'QB', 'RB', 'WR', 'TE', 'K', 'DST']

// The filter chip's own cut point, and deliberately the same 40 that tints a
// chip red (`lastsBand`): a reader who filters to "likely gone" should get
// exactly the rows he can see are red, not a second, invisible threshold.
const LIKELY_GONE_BELOW = 40

/** Whole points, signed -- the same rounding the cheat sheet's Edge column
 *  uses, since the first decimal of a difference between two projections is
 *  arithmetic rather than information. */
function fmtSigned(n: number): string {
  const r = Math.round(n)
  return r > 0 ? `+${r}` : `${r}`
}

// Which header the floating panel is describing. Four of the six carry one;
// the position badge and the star column say what they are.
type TipId = 'name' | 'lasts' | 'edge' | 'star' | 'draft'

interface RoomSimpleListProps {
  candidates: LiveCandidate[]
  players: Record<string, Player>
  onDraft: (c: LiveCandidate) => void
  /** His turn, on a live socket, in a room he has paid for -- computed once
   *  by DraftRoom and shared with every other Draft button in the room. */
  isMyTurn: boolean
  onOpenPlayer: (c: LiveCandidate) => void
  /** Player ids the BOARD reports as drafted -- current the moment a pick
   *  lands, roughly a second before the candidate list catches up. */
  draftedIds: Set<string>
}

// One row, memoized on primitives for the same reason the cheat sheet's is:
// every recompute hands down a brand-new `candidates` array, and a room that
// re-rendered a few hundred rows on every poll would spend a pick clock's
// worth of main thread on rows that did not change.
const SimpleRow = memo(function SimpleRow({
  c, player, isMyTurn, onOpenPlayer, onDraft,
}: {
  c: LiveCandidate
  player: Player | undefined
  isMyTurn: boolean
  onOpenPlayer: (c: LiveCandidate) => void
  onDraft: (c: LiveCandidate) => void
}): ReactNode {
  const edge = c.edge_pts
  return (
    <tr
      data-pid={c.player_id}
      // The whole row opens the profile, with the cheat sheet's own two
      // guards: a click that landed on a button belongs to that button, and
      // a click that ends a text selection is somebody copying a number.
      onClick={(e) => {
        if ((e.target as HTMLElement).closest('button')) return
        if (window.getSelection()?.toString()) return
        onOpenPlayer(c)
      }}
    >
      <td className="rsl-col-name">
        <button
          type="button"
          className="rsl-name-btn"
          onClick={() => onOpenPlayer(c)}
          title="Open profile"
        >
          {player?.name ?? c.player_id}
        </button>
        {player && (
          <span className="rsl-meta mono">
            {player.team} · BYE {player.bye ?? '—'}
          </span>
        )}
      </td>
      <td className="rsl-col-pos">{posBadge(c.position)}</td>
      <td className="rsl-col-lasts">
        <LastsChip pct={c.lasts_pct} />
      </td>
      {/* Muted at or below zero rather than red: a player who is not worth
          taking now is not a warning, he is simply somebody to wait on, and
          the column that says "worth grabbing now" should go quiet when the
          answer is no. A dash is a figure nobody computed -- no next turn to
          price him against -- and is not a zero. */}
      <td className={`rsl-col-edge mono${edge !== null && Math.round(edge) > 0 ? '' : ' is-flat'}`}>
        {edge === null ? '—' : `${fmtSigned(edge)} pts`}
      </td>
      <td className="rsl-col-star">
        {c.favourite && (
          <span role="img" aria-label="One of your guys">★</span>
        )}
      </td>
      <td className="rsl-col-btn">
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
})

export default function RoomSimpleList({
  candidates, players, onDraft, isMyTurn, onOpenPlayer, draftedIds,
}: RoomSimpleListProps) {
  const [search, setSearch] = useState('')
  const [pos, setPos] = useState('ALL')
  const [goingFast, setGoingFast] = useState(false)

  // HOW MANY REAL DRAFTS THE PERCENTAGE IS COUNTED FROM, fetched the first
  // time somebody asks what the column means and never otherwise. The number
  // belongs in the sentence -- "the share of 854 real ESPN drafts" is a claim
  // a reader can weigh, "the share of drafts" is not -- but it is worth
  // nothing until a tooltip is open, and the draft room should not spend a
  // request at mount on a sentence nobody has asked for yet.
  const [drafts, setDrafts] = useState<number | null>(null)
  const askedRef = useRef(false)
  const askCorpus = useCallback(() => {
    if (askedRef.current) return
    askedRef.current = true
    fetchMarketOverview()
      .then((o) => setDrafts(o.drafts))
      // The sentence stands without the count; it just stops being a number
      // the reader can weigh. Never a guessed one.
      .catch(() => {})
  }, [])

  // One floating panel for every header, positioned off the header's own
  // rect -- the cheat sheet's mechanism exactly (`positionTip`, the 130ms
  // hover intent, `.avail-th-tip`), imported rather than rebuilt so a
  // tooltip reads the same in both views.
  const [tip, setTip] = useState<{ id: TipId; rect: DOMRect } | null>(null)
  const [tipPos, setTipPos] = useState<{ left: number; top: number } | null>(null)
  const tipRef = useRef<HTMLDivElement>(null)
  const tipTimer = useRef<number | null>(null)

  const hideTip = useCallback(() => {
    if (tipTimer.current !== null) window.clearTimeout(tipTimer.current)
    tipTimer.current = null
    setTip(null)
  }, [])

  const showTip = useCallback((id: TipId, el: HTMLElement, now: boolean) => {
    if (tipTimer.current !== null) window.clearTimeout(tipTimer.current)
    askCorpus()
    const rect = el.getBoundingClientRect()
    if (now) {
      // A Tab press is already one deliberate move; making its answer wait
      // reads as the control lagging rather than as considerate pacing.
      setTip({ id, rect })
      return
    }
    tipTimer.current = window.setTimeout(() => setTip({ id, rect }), TIP_DELAY_MS)
  }, [askCorpus])

  // Measured after it renders and before the paint. `drafts` is in the deps
  // because the count can land while the panel is open, which makes the
  // sentence a line longer.
  useLayoutEffect(() => {
    if (!tip || !tipRef.current) {
      setTipPos(null)
      return
    }
    const { width, height } = tipRef.current.getBoundingClientRect()
    setTipPos(positionTip(tip.rect, { width, height }))
  }, [tip, drafts])

  useEffect(() => {
    if (!tip) return
    window.addEventListener('scroll', hideTip, true)
    return () => window.removeEventListener('scroll', hideTip, true)
  }, [tip, hideTip])

  const q = search.trim().toLowerCase()
  // ESPN'S ORDER, WHICH IS THE ONE THE SERVER ALREADY BUILT. `rank` is the
  // row's position in that list (ESPN's own, with the players ESPN does not
  // rank after it by consensus) and is the one field guaranteed non-null, so
  // sorting on it needs no null rule and cannot disagree with the board.
  const rows = candidates
    .filter((c) => {
      if (draftedIds.has(c.player_id)) return false
      if (pos !== 'ALL' && c.position !== pos) return false
      if (goingFast && !(c.lasts_pct !== null && c.lasts_pct < LIKELY_GONE_BELOW)) return false
      if (!q) return true
      const player = players[c.player_id]
      return `${player?.name ?? c.player_id} ${player?.team ?? ''}`
        .toLowerCase().includes(q)
    })
    .slice()
    .sort((a, b) => a.rank - b.rank)

  // The pick both figures are measured to -- the reader's own next turn. Off
  // the payload rather than inferred from the round he thinks he is in, and
  // named in the tooltips rather than in the headers, which would then need
  // rewriting twice a minute.
  const atPick = candidates.find((c) => c.lasts_at_pick !== null)?.lasts_at_pick ?? null
  const when = atPick === null ? 'your next turn' : `pick ${atPick}`
  const counted = drafts === null
    ? 'real ESPN drafts' : `${drafts.toLocaleString()} real ESPN drafts`

  const tipCopy: Record<TipId, string> = {
    name: 'His name, NFL team and bye week -- click any row for his full '
      + 'profile.',
    lasts: `The share of ${counted} where he was still on the board at `
      + `${when}, which is when you pick again.`,
    edge: `How many points you gain by taking him now instead of the best `
      + `player at his position you could still expect at ${when}, weighted `
      + `by how likely each of them is to be there in those same ${counted}.`,
    star: 'The players you picked as my guys -- your draft plan leans towards '
      + 'them when they are close.',
    draft: 'Sends the pick to ESPN. Only enabled on your turn.',
  }

  /** A header with an explanation behind it. The dotted underline is the
   *  affordance a reader can see before hovering; the panel is the same one
   *  the cheat sheet opens. */
  function tipTh(id: TipId, label: ReactNode, className: string): ReactNode {
    return (
      <th
        className={className}
        onMouseEnter={(e) => showTip(id, e.currentTarget, false)}
        onMouseLeave={hideTip}
        onFocus={(e) => showTip(id, e.currentTarget, true)}
        onBlur={hideTip}
        onKeyDown={(e) => { if (e.key === 'Escape') hideTip() }}
      >
        <span className="avail-th-hint" tabIndex={0} aria-describedby="rsl-th-tip">
          {label}
        </span>
      </th>
    )
  }

  return (
    <div className="rsl">
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
        {/* A FILTER, NOT A LEGEND. The cheat sheet's key of the same name
            explains why some of its rows are breathing; here there is no
            pulse to explain and the words are worth more as the question
            they ask: show me only the players I will lose if I wait. */}
        <button
          type="button"
          className={`rsl-chip${goingFast ? ' is-active' : ''}`}
          aria-pressed={goingFast}
          onClick={() => setGoingFast((on) => !on)}
        >
          {/* The mark, so the control reads as a switch rather than as a
              second search box sitting beside the first one. Red, because
              what it selects is the red chips. */}
          <span className="rsl-chip-dot" aria-hidden="true" />
          Likely gone by your next pick
        </button>
      </div>

      <table className="rsl-table">
        <thead>
          <tr>
            {tipTh('name', 'Player', 'rsl-col-name')}
            <th className="rsl-col-pos">Pos</th>
            {tipTh('lasts', 'Will he be there?', 'rsl-col-lasts')}
            {tipTh('edge', 'Worth grabbing now', 'rsl-col-edge')}
            {tipTh('star', 'My guys', 'rsl-col-star')}
            {tipTh('draft', <span className="sr-only">Draft</span>, 'rsl-col-btn')}
          </tr>
        </thead>
        <tbody>
          {rows.map((c) => (
            <SimpleRow
              key={c.player_id}
              c={c}
              player={players[c.player_id]}
              isMyTurn={isMyTurn}
              onOpenPlayer={onOpenPlayer}
              onDraft={onDraft}
            />
          ))}
          {rows.length === 0 && (
            <tr>
              {/* Six: name, position, the two figures, the star and the
                  button's own blank one. */}
              <td colSpan={6} className="avail-empty">
                {candidates.length === 0
                  ? 'No players yet.'
                  : 'No players match this filter.'}
              </td>
            </tr>
          )}
        </tbody>
      </table>

      {tip && (
        <div
          id="rsl-th-tip"
          ref={tipRef}
          role="tooltip"
          className="avail-th-tip"
          style={tipPos
            ? { left: tipPos.left, top: tipPos.top, visibility: 'visible' }
            : { left: 0, top: 0, visibility: 'hidden' }}
        >
          {tipCopy[tip.id]}
        </div>
      )}
    </div>
  )
}
