import { depthGroup, type ProfilePayload } from './payload'

function list(words: string[]): string {
  if (words.length <= 1) return words[0] ?? ''
  return `${words.slice(0, -1).join(', ')} and ${words[words.length - 1]}`
}

/** How the last sentence closes, indexed by how many of the three parts of
 *  the popup are real. A defense has none of them and gets no sentence. */
const CLOSE = ['', 'is real.', 'are both real.', 'are all real.']

/** Why there are no seasons, in the fewest words that are true for THIS
 *  player. A rookie and a defense are both empty here for completely
 *  different reasons, and "no data" would describe neither: one has not
 *  played yet, the other never will have per-player weeks. */
function reason(position: string, nflSeason: number | null): { lead: string; why: string } {
  if (position === 'DST') {
    return {
      lead: 'No per-player weeks for a defense.',
      why: 'are built from weekly rows a defense never has',
    }
  }
  if (position === 'K') {
    // A kicker's weeks are blanked whenever the league prices no kicking:
    // every row would score zero, which charts the scoring rules rather than
    // the kicker. Connect a league that scores field goals and this fills in
    // on its own.
    return {
      lead: 'No scored kicking weeks.',
      why: 'need weeks this league puts a price on',
    }
  }
  if (nflSeason === null || nflSeason <= 1) {
    return { lead: 'No NFL seasons yet.', why: 'need games played' }
  }
  return {
    lead: 'No weekly history.',
    why: 'need regular-season rows the stats table does not carry for him',
  }
}

// The dashed line under the status line, on a popup whose panels have
// nothing to draw.
//
// A popup that silently drops half its cards reads as broken, and a popup
// that draws them as frames of em-dashes is worse -- it presents an absence
// as a measurement. So the thin state says the two things a reader needs
// before he looks at what is left: which parts are empty, and which parts
// are real. Both halves are read off the payload rather than assumed, so the
// sentence cannot promise a projection that is not there.
export default function MissingData({ payload }: { payload: ProfilePayload }) {
  const { header, bio, seasons, game_log: gameLog, summary, schedule, depth_chart: depth } = payload
  // Nothing missing, nothing to say. The four panels above are drawn from
  // `seasons`, so that is the test -- never the position, because a kicker
  // has seasons exactly when his league prices kicking.
  if (seasons.length > 0) return null

  const { lead, why } = reason(header.position, bio.nfl_season)

  const empty = ['Health', 'finish', 'steadiness']
  // The week chart is its own card off its own key: a player can have last
  // season's game log without a summarised season row, and naming a chart
  // that is on screen as missing would be the notice getting it wrong.
  if (gameLog.length === 0) empty.push('the week chart')

  const real: string[] = []
  if (summary.proj_ppg !== null) real.push('the projection')
  if (schedule.some((w) => w.pct !== null)) real.push('the schedule')
  // The card's own question, asked through the card's own function: a notice
  // that promised a room the popup does not draw would be worse than no
  // notice. Same for the other two -- `proj_ppg` is what the Per game panel
  // draws its one column from, and a week with no percentile is a week the
  // schedule strip has nothing to say about.
  if (depthGroup(depth, header.position) !== null) real.push('the room')
  const realSentence = list(real)

  return (
    <div className="pp-pop-notice">
      <svg className="pp-pop-notice-icon" viewBox="0 0 24 24" aria-hidden="true">
        <circle cx="12" cy="12" r="9" />
        <path d="M12 8v5" />
        <path d="M12 16.5v.01" />
      </svg>
      <p className="pp-pop-notice-text">
        <span className="pp-pop-notice-lead">{lead}</span>
        {` ${list(empty)} ${why}, so they are empty rather than zero.`}
        {real.length > 0 && ` ${realSentence.charAt(0).toUpperCase()}${
          realSentence.slice(1)} ${CLOSE[real.length]}`}
      </p>
    </div>
  )
}
