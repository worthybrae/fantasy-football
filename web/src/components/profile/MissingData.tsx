import type { ProfilePayload } from './payload'

const FACTOR_KEYS = ['production', 'durability', 'role', 'environment', 'schedule'] as const

// The board's placeholder for a factor it cannot compute is exactly 50.0
// (scoring/board.py fills K and DST with it). A genuinely computed
// percentile landing on exactly 50 is possible and would be misread here as
// a placeholder -- which is why this section says "sits at exactly 50" and
// what that usually means, rather than "was not computed".
const NEUTRAL = 50

function list(words: string[]): string {
  if (words.length <= 1) return words[0] ?? ''
  return `${words.slice(0, -1).join(', ')} and ${words[words.length - 1]}`
}

// What this card cannot show, and why. A defense's card is thin BY
// CONSTRUCTION -- no weekly rows exist for one, so there is no average, no
// spread, no game log and no cohort -- and the honest response to that is
// not six panels of em-dashes. It is one page that makes the claim the data
// does support and then says plainly what is missing and what would fill it.
export default function MissingData({ payload }: { payload: ProfilePayload }) {
  const { header, bio, factors, seasons, schedule, similar } = payload
  const gaps: { what: string; why: string }[] = []

  if (seasons.length === 0) {
    if (header.position === 'DST') {
      gaps.push({
        what: 'No weekly history',
        why: 'Defenses have no per-player weekly rows, so there is no per-game'
          + ' average, no week-to-week spread, no game log and no cohort of'
          + ' comparable seasons to read.',
      })
    } else if (header.position === 'K') {
      gaps.push({
        what: 'No weekly history',
        why: 'A kicker’s weeks are blanked whenever the league prices no'
          + ' kicking: every row would score zero, which charts the scoring'
          + ' rules rather than the kicker. Connect a league that scores field'
          + ' goals and this fills in on its own.',
      })
    } else if (bio.nfl_season === null || bio.nfl_season <= 1) {
      gaps.push({
        what: 'No NFL games yet',
        why: 'There is nothing to average, nothing to be steady or wild about,'
          + ' and no season to find comparables for. Projection and market are'
          + ' all anyone has on him, including everyone else in the room.',
      })
    } else {
      gaps.push({
        what: 'No weekly history',
        why: 'The stats table carries no regular-season rows for him, so every'
          + ' section built on them is empty rather than estimated.',
      })
    }
  }

  const pinned = FACTOR_KEYS.filter((k) => factors[k] === NEUTRAL)
  if (pinned.length === FACTOR_KEYS.length) {
    gaps.push({
      what: 'The five factors read neutral',
      why: 'Production, durability, role, environment and schedule all sit at'
        + ' exactly 50 — the board fills kickers and defenses with a neutral'
        + ' value because the scoring formula prices none of the stats those'
        + ' factors would be built from. A number there would be invented.',
    })
  } else if (pinned.length > 0) {
    const real = FACTOR_KEYS.filter((k) => factors[k] !== NEUTRAL)
    gaps.push({
      what: `${pinned.length} of the five factors read neutral`,
      why: `${list([...pinned])} sit at exactly 50, the board’s placeholder`
        + ` for a factor it cannot compute for him. ${list([...real])}`
        + ` ${real.length === 1 ? 'is' : 'are'} measured and worth reading.`,
    })
  }

  if (similar.mode === 'value_neighbors') {
    gaps.push({
      what: 'No stat twins',
      why: 'With no stat line to match, comparables fall back to whoever sits'
        + ' nearest on the board by value — neighbours in price, not in shape.',
    })
  }

  if (schedule.length === 0) {
    gaps.push({
      what: 'No matchup ranks',
      why: 'Weekly difficulty is built from what each defence allowed this'
        + ' position last season, and this position has no rows to have been'
        + ' allowed.',
    })
  }

  if (gaps.length === 0) return null

  return (
    <ul className="pp-gaps">
      {gaps.map((g) => (
        <li className="pp-gap-item" key={g.what}>
          <svg className="pp-gap-icon" viewBox="0 0 24 24" aria-hidden="true">
            <circle cx="12" cy="12" r="9" />
            <path d="M12 16v-5" />
            <path d="M12 8h.01" />
          </svg>
          <div>
            <div className="pp-gap-what">{g.what}</div>
            <div className="pp-gap-why">{g.why}</div>
          </div>
        </li>
      ))}
    </ul>
  )
}
