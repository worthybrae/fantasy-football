import type { MarketTurn } from '../api'
import '../market.css'

// ONE TURN OF A DRAFT, DRAWN. Lifted out of pages/Market.tsx when the routes
// became lazy: the landing page's argument (components/benefitFigures.tsx)
// draws a real one of these, and a component the landing page imports from
// the archive page keeps the whole archive page in the first chunk however
// the route is declared.

export const POSITIONS = ['QB', 'RB', 'WR', 'TE', 'K', 'DST']

export function posVar(position: string): string {
  const key = position.toLowerCase()
  return POSITIONS.some((p) => p.toLowerCase() === key)
    ? `var(--pos-${key})` : 'var(--text-3)'
}

export const pct = (share: number) => `${Math.round(share * 100)}%`

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
