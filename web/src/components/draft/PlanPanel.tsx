import type { ReactNode } from 'react'
import type { LivePlanTurn, Player } from '../../api'
import { riskTone } from './tone'

// THE REST OF THE DRAFT, ONE ROW PER TURN.
//
// The cards above the table answer "who now"; this answers "and then what",
// which is the question a drafter actually holds in his head for the whole
// middle of a draft and which this room had no answer to at all. Every row is
// one of the reader's remaining turns with the player the plan means to take
// on it, how likely he is to still be there, and what taking him is worth
// against waiting again.
//
// Nothing here is computed: the turns, the targets, the alternates and the
// reasons all come down in `/api/live/state`'s `plan` (scoring/plan.py). A
// panel that re-derived any of it could disagree with the cards, and two
// answers to "who am I taking in round 4" is worse than none.

// duplicated from RosterPanel.tsx / AvailableList.tsx (unexported in both):
// a four-line pure function isn't worth a shared module between four views.
function posBadge(position: string | undefined): ReactNode {
  if (!position) return null
  return <span className={`pos-badge pos-badge-${position.toLowerCase()}`}>{position}</span>
}

function fmtSigned(n: number | null): string {
  if (n === null) return '—'
  const r = Math.round(n)
  return r > 0 ? `+${r}` : `${r}`
}

/** The sign, as the room's own two tones. Rounded before the sign is read,
 *  and null carries no direction, so it carries no colour -- the same rule
 *  the table's Edge column and the target cards already follow. */
function edgeTone(edge: number | null): string {
  if (edge === null) return ''
  const shown = Math.round(edge)
  return shown > 0 ? 'is-up' : shown < 0 ? 'is-down' : ''
}

interface PlanPanelProps {
  /** The reader's remaining turns, in pick order. */
  plan: LivePlanTurn[]
  /** The one-time `/api/players` join table -- names, positions, faces. A
   *  player it has not resolved yet renders as his id, the same fallback
   *  the available list makes. */
  players: Record<string, Player>
  /** Player ids the reader has starred. The plan already leans on them
   *  (they clear a lower availability bar and carry a bonus), so the panel
   *  says which rows that applies to rather than leaving the reader to
   *  guess why a name is there. */
  favourites?: Set<string>
  /** Opens the profile over the room. Optional: the landing page's
   *  spectator room has a plan to show and nobody to open it for. */
  onOpenPlayer?: (playerId: string) => void
}

export default function PlanPanel({
  plan, players, favourites, onOpenPlayer,
}: PlanPanelProps) {
  // No plan is not an error -- a spectator has no turns, and the panel is
  // simply absent rather than standing there empty saying so.
  if (plan.length === 0) return null
  // EVERY REMAINING TURN, not the next few. The panel is a scroll region of
  // its own (`.plan-panel`), so the whole plan costs nothing but scrolling
  // -- and a panel that quietly stopped at six turns while its own header
  // counted eleven would be the one thing a plan must never be, which is
  // wrong about itself.
  const turns = plan

  return (
    <div className="plan-panel">
      <div className="plan-panel-head">
        <span className="draft-cap">The plan</span>
        <span className="plan-panel-count mono">
          {plan.length} {plan.length === 1 ? 'turn' : 'turns'} left
        </span>
      </div>
      <ul className="plan-list">
        {turns.map((turn) => {
          const target = turn.target
          if (target === null) {
            // The plan's honest answer for a turn nobody cleared: say so,
            // in the turn's own slot, so the reader sees the gap rather
            // than a plan that skips a round.
            return (
              <li key={turn.pick_no} className="plan-turn plan-turn-none">
                <div className="plan-turn-head">
                  <span className="plan-pick mono">Pick {turn.pick_no}</span>
                  <span className="plan-round">Round {turn.round}</span>
                </div>
                <div className="plan-target plan-none">No clear target</div>
              </li>
            )
          }
          const player = players[target.player_id]
          const name = player?.name ?? target.player_id
          const body = (
            <>
              {/* The face, at rail scale. Same rule the cards follow: the
                  badge is what a player with no photograph gets, not a
                  second thing beside the photograph. */}
              {player?.headshot ? (
                <img className="plan-shot" src={player.headshot} alt=""
                     loading="lazy" />
              ) : posBadge(player?.position)}
              <span className="plan-name">
                {favourites?.has(target.player_id) && (
                  <span className="plan-star" role="img"
                        aria-label="One of your guys">★</span>
                )}
                {name}
              </span>
              {/* THE TWO NUMBERS, IN THE ROOM'S OWN COLOURS. They were both
                  grey, one step apart on the text ramp, which made the pair
                  read as a single meaningless "0% +39". The table and the
                  target cards already tone these two -- survival on the
                  red-amber-green ramp, points by their sign -- and a rail
                  that toned them differently would be the third answer to a
                  question that has one. A null in either gets no colour at
                  all: the ramp is a claim about a real number. */}
              <span className="plan-nums mono">
                <span
                  className="plan-lasts"
                  style={target.lasts_pct === null
                    ? undefined : { color: riskTone(target.lasts_pct) }}
                >
                  {target.lasts_pct === null ? '—' : `${Math.round(target.lasts_pct)}%`}
                </span>
                <span className={`plan-edge delta-tone ${edgeTone(target.edge_pts)}`}>
                  {fmtSigned(target.edge_pts)}
                </span>
              </span>
            </>
          )
          return (
            <li key={turn.pick_no} className="plan-turn">
              <div className="plan-turn-head">
                <span className="plan-pick mono">Pick {turn.pick_no}</span>
                <span className="plan-round">Round {turn.round}</span>
              </div>
              {/* A real <button> only when there is a room to open a profile
                  in -- the same "clickable only when it can honour it" rule
                  RosterPanel's rows follow. */}
              {onOpenPlayer ? (
                <button type="button" className="plan-target"
                        onClick={() => onOpenPlayer(target.player_id)}>
                  {body}
                </button>
              ) : (
                <div className="plan-target">{body}</div>
              )}
              {/* The plan's own words, for and against, exactly as the cards
                  print them. Pros first, in the order scoring/plan.py built
                  them: the strongest argument is the one it wrote first. */}
              {(target.pros.length > 0 || target.cons.length > 0) && (
                <ul className="plan-reasons">
                  {target.pros.map((pro) => (
                    <li key={pro} className="plan-reason is-pro">{pro}</li>
                  ))}
                  {target.cons.map((con) => (
                    <li key={con} className="plan-reason is-con">{con}</li>
                  ))}
                </ul>
              )}
              {turn.alternates.length > 0 && (
                <div className="plan-alts">
                  <span className="plan-alts-cap">or</span>
                  {turn.alternates.map((alt) => (
                    <span key={alt.player_id} className="plan-alt">
                      {players[alt.player_id]?.name ?? alt.player_id}
                    </span>
                  ))}
                </div>
              )}
            </li>
          )
        })}
      </ul>
    </div>
  )
}
