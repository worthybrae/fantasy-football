import type { SimilarPeer } from '../../api'
import { SEASON_GAMES } from '../draft/weeks'
import PopCard from './PopCard'
import { CARD_HINTS } from './hints'
import { fmtSigned } from './payload'

// Who else is on the shelf, as faces. The card beside it (Comparable
// seasons) looks backwards -- seasons that already happened and what
// followed them -- and this one looks sideways: the players at his own
// position, on THIS year's board, that the model reads as the same kind of
// player.
//
// A strip of photographs rather than a table of numbers, because recognition
// is the whole job here. A drafter does not read "Khalil Shakir, 79.6%, 171
// points" and learn something; he sees a face he knows and thinks "right,
// THAT kind of receiver" in less time than the sentence takes. The two facts
// that survive beside the face are the ones he cannot get from it: the name,
// for the faces he does not know, and the projection, which is what he would
// be getting instead.
//
// The match percentage is gone from the tile and NOT from the card: the
// order is still the ranking -- leftmost is closest -- and the number rides
// on each tile's tooltip. Five percentages under five faces was the row
// reading as a chart of scores, which put the emphasis on the arithmetic
// rather than on the players.
const PEERS = 5

// What the score is computed over, said once in the head. A reader who is
// not told what "closest" means here reads it as ADP proximity, which it
// deliberately is not.
const NOTE = 'projection · stat line · age · build'

// The signed gap has to clear this before it is coloured. A tenth of a point
// a game is nine tenths of a point across a season and two projections that
// close are the same projection; painting that green would make five nearly
// identical players look like five different decisions.
const SAME = 0.5

export default function SimilarPlayers({ data, projPpg, onSelectPlayer }: {
  data: { season: number | null; position: string; players: SimilarPeer[] } | null
  /** The player this popup is about, per game, off `summary.proj_ppg`. It is
   *  the same arithmetic the tiles do -- projected season points over
   *  SEASON_GAMES -- so the two numbers subtract without conversion. Null for
   *  a player ESPN does not project, and then the tiles carry no gap. */
  projPpg: number | null
  onSelectPlayer: (id: string) => void
}) {
  const rows = (data?.players ?? []).slice(0, PEERS)
  const gap = (p: SimilarPeer): number | null =>
    p.proj_points === null || projPpg === null
      ? null : p.proj_points / SEASON_GAMES - projPpg
  const gapTone = (d: number | null): string =>
    d === null || Math.abs(d) < SAME ? ''
      : d > 0 ? 'delta-tone is-up' : 'delta-tone is-down'
  // Nothing to say rather than a frame of dashes: a player the board does
  // not carry, or the only one left at his position.
  if (rows.length === 0) return null

  return (
    <PopCard title="Similar players" note={NOTE} hint={CARD_HINTS.similar}>
      {/* Ordered, and the payload sorts it: the first tile is the closest
          player at his position and the markup says so. */}
      <ol className="pp-pop-sims">
        {rows.map((p) => (
          <li key={p.player_id}>
            <button
              type="button"
              className="pp-pop-sim"
              onClick={() => onSelectPlayer(p.player_id)}
              title={`${p.name} — ${Math.round(p.similarity)}% match`}
            >
              {p.headshot ? (
                // `alt=""` -- the name is directly under it and the button
                // carries the whole tile as its label, so a described image
                // would be the third telling.
                <img className="pp-pop-sim-face" src={p.headshot} alt=""
                     width={52} height={52} loading="lazy" />
              ) : (
                // A player nflverse has no photo of -- every defense, and a
                // rookie the CDN has not caught up with. An initial in the
                // same circle rather than a grey hole, so the row of five
                // keeps its rhythm.
                <span className="pp-pop-sim-face is-blank" aria-hidden="true">
                  {p.name.slice(0, 1)}
                </span>
              )}
              <span className="pp-pop-sim-name">{p.name}</span>
              <span className="mono pp-pop-sim-proj">
                {p.proj_points === null
                  ? '—' : `${(p.proj_points / SEASON_GAMES).toFixed(1)}`}
                <span className="pp-pop-sim-unit"> / g</span>
              </span>
              {/* What taking him instead would cost or buy, per game. The
                  projection above answers "how good is he"; a row of five
                  of those still leaves the reader doing five subtractions
                  in his head against the player he is actually looking at,
                  which is the only question this card exists to answer. */}
              <span className={`mono pp-pop-sim-gap ${gapTone(gap(p))}`}>
                {gap(p) === null ? '' : fmtSigned(gap(p), 1)}
              </span>
            </button>
          </li>
        ))}
      </ol>
    </PopCard>
  )
}
