import { useEffect, useRef, useState, type CSSProperties } from 'react'
import type { BoardPlayer, PlayerProfileData } from '../../api'
import {
  ChangeMeter, HealthMeter, SteadyMeter, healthLevel, steadyLevel,
} from './AvailableList'
import { CellTip, loadProfile } from './CellTip'
import { SEASON_GAMES } from './weeks'

// THE BOARD'S HOVER CARD: a condensed player profile, not a stat dump.
//
// What was here before was a list of the fields the payload happened to
// carry -- "#41 overall", "Tier 4", "VOR 12.3" -- three of which are internal
// quantities a drafter has no feel for, and none of which answer the question
// somebody hovering a pick on the board is actually asking: who is this, and
// was taking him here any good.
//
// So this is the top of the player profile, in a box: the face, what he is
// projected to score, where the two markets had him, how his last season
// actually went, and the three ratings the room draws everywhere else. Every
// piece is a component the product already uses at full size, so nothing
// here can drift from what the popup says when it is opened.
//
// FETCHED, AND CHEAP. The board cell carries a name, a photo and two market
// ranks; everything else comes from `loadProfile`, which is the same cache
// the available table's own hover panels use -- so hovering the board warms
// the profile a reader is about to open, and opening it afterwards costs
// nothing.

/** m:ss-style signed number for the two market deltas: how far past his
 *  price he went. Positive is a fall (a bargain), which is the ticker's own
 *  reading, and the tone follows it. */
function move(pick: number, rank: number | null) {
  if (rank === null) return null
  const slots = Math.round(pick - rank)
  if (slots === 0) return null
  return { slots: Math.abs(slots), steal: slots > 0 }
}

export default function BoardPeek({ player, pick, style }: {
  player: BoardPlayer
  /** The overall pick he went at, for the two market comparisons. */
  pick: number | null
  style: CSSProperties
}) {
  const [profile, setProfile] = useState<PlayerProfileData | null>(null)
  const wanted = useRef(player.player_id)

  useEffect(() => {
    wanted.current = player.player_id
    setProfile(null)
    loadProfile(player.player_id)
      .then((body) => {
        if (wanted.current === player.player_id) setProfile(body)
      })
      // No profile is not an error here: the card still names him, shows his
      // photo and both market ranks, which is more than the list it replaced.
      .catch(() => { /* the header alone is a real card */ })
  }, [player.player_id])

  const header = profile?.header
  const projPpg = player.proj_ppg
    ?? (header?.proj_points == null ? null : header.proj_points / SEASON_GAMES)
  const health = healthLevel(header?.career_games_pg)
  const steady = steadyLevel(header?.consistency_pct)
  // Where his projection places him among his own position, best first.
  // Only the profile endpoint computes it, so it appears when the fetch
  // lands and the card simply omits it before that.
  const posFinish = header?.proj_pos_finish ?? null
  // How the projection compares with what he actually did: the profile's own
  // `proj_delta` (proj_ppg minus his recent per-game, scoring/profile.py).
  // Attached to the ppg figure, because that is the number it is a delta OF;
  // a rookie has no last year and shows none.
  const projDelta = profile?.summary?.proj_delta ?? null
  const ppgMove = projDelta === null || Math.abs(projDelta) < 0.05
    ? null : { value: Math.abs(projDelta).toFixed(1), up: projDelta > 0 }
  const adp = pick === null ? null : move(pick, player.market_rank)
  const espn = pick === null ? null : move(pick, player.espn_ppr_rank)

  return (
    <div className="board-peek" style={style} role="tooltip">
      <div className="board-peek-head">
        {player.headshot
          ? <img className="board-peek-face" src={player.headshot} alt="" loading="lazy" />
          : <span className={`board-peek-badge pos-badge pos-badge-${player.position.toLowerCase()}`}>
              {player.position}
            </span>}
        <span className="board-peek-who">
          <span className="board-peek-name">{player.name}</span>
          <span className="board-peek-sub mono">
            {player.position}
            {player.team ? ` · ${player.team}` : ''}
            {player.bye !== null ? ` · BYE ${player.bye}` : ''}
          </span>
        </span>
        {/* WHAT HE IS, top right, beside his name: what he scores and where
            that puts him among his position. The two market ranks sit at the
            foot of the card instead -- they are what this pick is being
            judged against, not what he is. */}
        <span className="board-peek-top">
          {/* MIRRORS THE NAME BLOCK: headline over a quiet meta line. The
              left half is who he is (name over position/team/bye); this half
              is what he is worth (the number over where that puts him and
              how it compares with last year). One loud thing per corner. */}
          <span className="board-peek-proj mono">
            {projPpg === null ? '—' : projPpg.toFixed(1)}
            <span className="board-peek-unit">/g</span>
          </span>
          {(posFinish !== null || ppgMove !== null) && (
            <span className="board-peek-rank mono"
                  title={ppgMove === null ? undefined
                    : `Projected ${ppgMove.up ? 'up' : 'down'} ${ppgMove.value} points a game on last season`}>
              {posFinish !== null && <>{player.position}{posFinish}</>}
              {posFinish !== null && ppgMove !== null && (
                <span className="board-peek-dot" aria-hidden="true">·</span>
              )}
              {ppgMove && (
                <span className={`board-peek-move ${ppgMove.up ? 'is-steal' : 'is-reach'}`}>
                  {ppgMove.up ? '▲' : '▼'}{ppgMove.value}
                </span>
              )}
            </span>
          )}
        </span>
      </div>

      {/* THE GAME LOG, which is the chart a reader hovering a pick actually
          wants: not "he averaged 14.9" but which weeks carried it. The
          table's own `games` panel, unmodified -- same fetch, same columns,
          same colours, opponents under the bars -- so the board and the
          table cannot draw his season two different ways. */}
      <div className="board-peek-log">
        <CellTip kind="games" playerId={player.player_id} />
      </div>

      {/* The three ratings, in the room's own meters. A player with none
          measured (a defense, a rookie) gets no row rather than three
          dashes. */}
      {/* The three ratings, in the room's own meters, with the market at the
          far end of the same row: two ranks, and how far past each of them he
          actually went. A player with nothing measured (a defense, a rookie)
          gets no meters rather than three dashes -- the market still shows. */}
      <div className="board-peek-foot">
        {health !== null && (
          <span className="board-peek-meter">
            <span className="draft-cap">Health</span>
            <HealthMeter level={health} gamesPg={header?.career_games_pg as number} />
          </span>
        )}
        {steady !== null && (
          <span className="board-peek-meter">
            <span className="draft-cap">Steady</span>
            <SteadyMeter level={steady} cv={header?.consistency_cv ?? null} />
          </span>
        )}
        {header?.proj_change != null && (
          <span className="board-peek-meter">
            <span className="draft-cap">Growth</span>
            <ChangeMeter change={header.proj_change} />
          </span>
        )}
        <span className="board-peek-market">
          <span className="board-peek-mkt">
            <span className="draft-cap">ADP</span>
            <span className="board-peek-mkt-val mono">
              {player.market_rank === null ? '—' : Math.round(player.market_rank)}
              {adp && (
                <span className={`board-peek-move ${adp.steal ? 'is-steal' : 'is-reach'}`}>
                  {adp.steal ? '▲' : '▼'}{adp.slots}
                </span>
              )}
            </span>
          </span>
          <span className="board-peek-mkt">
            <span className="draft-cap">ESPN</span>
            <span className="board-peek-mkt-val mono">
              {player.espn_ppr_rank === null ? '—' : Math.round(player.espn_ppr_rank)}
              {espn && (
                <span className={`board-peek-move ${espn.steal ? 'is-steal' : 'is-reach'}`}>
                  {espn.steal ? '▲' : '▼'}{espn.slots}
                </span>
              )}
            </span>
          </span>
        </span>
      </div>
    </div>
  )
}
