import { Fragment } from 'react'

// The landing hero's backdrop: a draft board drafting itself.
//
// NOT a video and not an image, on purpose. A screen recording of the real
// room would be a few megabytes, would blur on a 2x display, would carry
// whatever palette the theme happened to be on the day it was captured, and
// would need a poster frame to satisfy `prefers-reduced-motion`. This is a
// few kilobytes of DOM that is crisp at any resolution, reads the same
// tokens the room reads (so a change to --pos-wr or --bg-1 carries here on
// its own), and freezes to a static filled board with four lines of CSS.
//
// Everything in it is invented. There is no fetch, no import from `api.ts`
// and no player who exists: the names below were made up for this file, and
// the point of the panel next to it (`PROOF_ROWS` in pages/Landing.tsx) is
// that THOSE numbers are real. A backdrop that quietly showed real players
// would blur that line, which is the one thing this page cannot afford.
//
// What it borrows from the real board (components/DraftBoardGrid.tsx) is the
// *structure*, because that is what makes it read as the product rather than
// as motion: a round-label column with the snake's direction arrow, cells on
// a hairline grid, the overall pick number sitting in a slot nobody has
// taken yet, the same `pos-badge` classes the room's own cells use, and one
// column tinted as yours. Fill order is the snake, so the wave visibly turns
// around at the end of every round.
//
// Accessibility and motion, both non-negotiable:
//   * the whole backdrop carries aria-hidden and pointer-events: none -- it
//     is decoration, and nothing in it is content a screen reader should
//     ever meet or a pointer should ever hit.
//   * only `opacity` and `transform` are animated. Not `background-color`,
//     and NOT `box-shadow`: box-shadow is a paint property and cannot be
//     handed to the compositor, which this repo already learned the hard way
//     on the available-list pulse (see PULSE_FLOOR in draft/AvailableList.tsx).
//   * `prefers-reduced-motion: reduce` stops the animation dead and leaves
//     every cell filled -- see landing.css.
//   * it is absolutely positioned inside the hero, so it contributes nothing
//     to layout and cannot shift anything when it mounts.

// A ten-team board, eight rounds deep. Ten because that is a real league
// size and it gives cells wide enough for a name at hero width; eight
// because the board is cropped by the hero's bottom edge anyway, and the
// first eight rounds of a fantasy draft contain no kickers or defences --
// which keeps the palette to the four hues anyone recognises.
const TEAMS = 10
const ROUNDS = 8
// Zero-based column that gets the "this one is yours" accent, the same
// treatment `.board-col-mine` gives it in the room. Right of centre so the
// tinted rule emerges below the proof panel rather than running behind the
// headline.
const MY_SLOT = 6

// Seconds between one pick landing and the next. Slow on purpose: at this
// rate a visitor notices the board is moving several seconds after they
// notice the headline, which is the correct order.
const STEP_S = 0.55
// One full turnover. Equal to the total stagger, so the wave rolls straight
// into itself: the moment the last pick lands, the first cell is clearing.
const CYCLE_S = TEAMS * ROUNDS * STEP_S

// How far into the cycle the board already is when the page paints, as a
// fraction of it. This is the difference between landing on an empty grid
// that takes forty seconds to become recognisable and landing on a draft
// already seven rounds deep with the last few picks still to come -- which
// is both the better first impression and the more honest one, since that
// is what walking into a live draft room actually looks like. Implemented as
// a negative animation-delay, which starts an animation already in progress.
const PRELOAD = 0.92

type Pos = 'QB' | 'RB' | 'WR' | 'TE'

// Position by board slot, not by pick order: backs and receivers early,
// quarterbacks and tight ends filling in from the third round on. Shaped by
// hand rather than generated, because a random scatter of four hues looks
// like confetti and a real board does not.
const POSITIONS: Pos[][] = [
  ['RB', 'WR', 'WR', 'RB', 'WR', 'TE', 'RB', 'WR', 'RB', 'WR'],
  ['WR', 'RB', 'WR', 'WR', 'RB', 'QB', 'WR', 'RB', 'TE', 'WR'],
  ['RB', 'WR', 'QB', 'WR', 'RB', 'WR', 'TE', 'WR', 'RB', 'QB'],
  ['WR', 'QB', 'RB', 'TE', 'WR', 'RB', 'WR', 'QB', 'RB', 'WR'],
  ['TE', 'WR', 'RB', 'QB', 'WR', 'RB', 'WR', 'TE', 'QB', 'RB'],
  ['QB', 'RB', 'WR', 'WR', 'TE', 'RB', 'QB', 'WR', 'RB', 'WR'],
  ['WR', 'TE', 'RB', 'QB', 'WR', 'RB', 'WR', 'WR', 'QB', 'TE'],
  ['RB', 'WR', 'TE', 'RB', 'WR', 'QB', 'WR', 'RB', 'TE', 'WR'],
]

// Eighty invented players. Written out one by one rather than combined from
// a pool of first and last names, so that every name on the page could be
// read and checked against nobody in the league -- a generator makes eighty
// combinations nobody has looked at.
const NAMES: string[][] = [
  ['Deshun Ferrell', 'Malik Ostrander', 'Corbin Latham', 'Nico Halvorsen', 'Jaylen Prichard',
   'Kade Bellamy', 'Damari Whitlock', 'Rashad Kingsley', 'Elias Thornbury', 'Marquan Doyle'],
  ['Tyrese Vandermeer', 'Bryce Ashworth', 'Jamarr Ledoux', 'Owen Castellano', 'Deion Rutledge',
   'Xavier Bonham', 'Trell Hutchins', 'Amari Lockridge', 'Cade Brubaker', 'Isaiah Pomeroy'],
  ['Roman Devereaux', 'Jaxon Priestley', 'Keyshawn Alder', 'Tobias Renfroe', 'Devante Cordell',
   'Micah Sutcliffe', 'Zion Harlow', 'Braylen Ostrom', 'Caleb Winstead', 'Dontae Marchetti'],
  ['Nasir Colburn', 'Eli Vanderkamp', 'Rondell Pierce', 'Kian Mattox', 'Jerome Ashby',
   'Trevonte Sallis', 'Grayson Kettles', 'Dashawn Ricci', 'Malachi Fontaine', 'Booker Lindsay'],
  ['Terrell Nakamura', 'Casey Ballard', 'Jamil Ordway', 'Weston Pardue', 'Kavon Trice',
   'Silas Brenner', 'Aiden Mulholland', 'Rico Vandenberg', 'Josiah Kemper', 'Landon Sowell'],
  ['Demetrius Falk', 'Ryder Novak', 'Tarik Ellsworth', 'Beau Ivory', 'Zaire Fontenot',
   'Nate Kirkwood', 'Julius Rainey', 'Trey Osgood', 'Marlon Sheppard', 'Colby Truitt'],
  ['Devin Ottoman', 'Khalil Barstow', 'Emory Lattimore', 'Payton Vance', 'Jarrell Duquette',
   'Tucker Amos', 'Sione Fifita', 'Bryson Halliwell', 'Marcus Delgado', 'Kyren Wooldridge'],
  ['Dane Kupfer', 'Ronan Stipe', 'Jalen Mercado', 'Chase Pomerantz', 'Tavien Rourke',
   'Miles Ackerley', 'Zeke Dunwoody', 'Preston Vialpando', 'Andre Kilgore', 'Braxton Ferrer'],
]

// Real league abbreviations against invented players, the same way the room's
// own cell carries one. A three-letter code is not data about anybody.
const CLUBS = [
  'BUF', 'SF', 'DET', 'KC', 'PHI', 'DAL', 'MIA', 'BAL', 'CIN', 'GB',
  'HOU', 'LAR', 'NYJ', 'ATL', 'TB', 'SEA', 'CHI', 'JAX', 'LAC', 'MIN',
  'IND', 'NO', 'PIT', 'CLE', 'DEN', 'ARI', 'WAS', 'NE', 'LV', 'TEN',
  'NYG', 'CAR',
]

// Where a (round, column) lands in the draft order. Odd rounds run left to
// right, even rounds come back -- the snake, exactly as the room's own
// `overallAt` computes it.
function overallAt(round: number, col: number): number {
  return (round - 1) * TEAMS + (round % 2 === 1 ? col + 1 : TEAMS - col)
}

const ROUND_LIST = Array.from({ length: ROUNDS }, (_, i) => i + 1)
const COL_LIST = Array.from({ length: TEAMS }, (_, i) => i)

export default function HeroBoard() {
  return (
    <div className="lp-hb" aria-hidden="true">
      <div className="lp-hb-grid" style={{ gridTemplateColumns: `26px repeat(${TEAMS}, minmax(0, 1fr))` }}>
        {/* Your column, tinted its whole height whether or not a pick has
            landed in it yet -- which is why it is one element behind the cells
            rather than a class on them: an empty slot in your column is still
            your slot. Placed explicitly, and so is every cell below it, because
            grid auto-placement would otherwise flow the cells AROUND this. */}
        <div
          className="lp-hb-mine"
          style={{ gridColumn: MY_SLOT + 2, gridRow: `1 / span ${ROUNDS}` }}
        />

        {ROUND_LIST.map((round) => (
          <Fragment key={round}>
            <div className="lp-hb-round" style={{ gridColumn: 1, gridRow: round }}>
              <span className="lp-hb-round-n mono">{round}</span>
              <span className="lp-hb-dir">{round % 2 === 1 ? '→' : '←'}</span>
            </div>

            {COL_LIST.map((col) => {
              const overall = overallAt(round, col)
              const pos = POSITIONS[round - 1][col]
              return (
                <div
                  className="lp-hb-cell"
                  key={col}
                  style={{ gridColumn: col + 2, gridRow: round }}
                >
                  <span className="lp-hb-n mono">{overall}</span>
                  {/* The pick itself, over the empty slot. This is the only
                      animated element in the component, and it animates
                      opacity and transform and nothing else. The delay is what
                      puts it in draft order; negative delays (everything up to
                      PRELOAD of a cycle) start already filled. */}
                  <span
                    className="lp-hb-pick"
                    style={{ animationDelay: `${((overall - 1) * STEP_S - PRELOAD * CYCLE_S).toFixed(2)}s` }}
                  >
                    <span className="lp-hb-top">
                      <span className={`pos-badge pos-badge-${pos.toLowerCase()}`}>{pos}</span>
                      <span className="lp-hb-club mono">{CLUBS[(overall - 1) % CLUBS.length]}</span>
                    </span>
                    <span className="lp-hb-name">{NAMES[round - 1][col]}</span>
                  </span>
                </div>
              )
            })}
          </Fragment>
        ))}
      </div>
      {/* The scrim. Static -- it never animates -- and heavy enough over
          every band the hero's own text occupies that the copy's contrast is
          the same whether the cells behind it are lit or dark. See
          landing.css for the arithmetic. */}
      <div className="lp-hb-scrim" />
    </div>
  )
}
