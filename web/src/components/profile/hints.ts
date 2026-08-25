// WHAT EACH CARD IN THE POPUP MEANS, in one file.
//
// Gathered here rather than written inline on fourteen cards for the reason
// the available table's own tooltip copy is (`tipCopy` in AvailableList.tsx):
// these have to agree with each other. Health, Steady and the position rank
// are three different scales drawn as the same five-bar picture, and three
// separately-written explanations drift into describing them as if they were
// the same measurement.
//
// Each states the actual meaning, not a paraphrase, and names the direction
// where the direction is not obvious -- Schedule's bars are tall for an EASY
// week and the position rank counts up from the best, both of which read
// backwards if a reader assumes otherwise.
//
// Delivery is `CardHint`, through PopCard's own `hint` prop.

/** Fixed titles. The two cards whose titles carry a season or a position
 *  build their own copy below. */
export const CARD_HINTS = {
  health: 'Games he played in each season, out of the games that season had. '
    + 'Seasons he missed entirely count as zero rather than being skipped, '
    + 'which is what separates a player who is durable from one who has been '
    + 'lucky with short stretches.',

  perGame: 'Fantasy points per game in each season, under this league’s '
    + 'own scoring rather than a generic one. The hollow last column is this '
    + 'season’s projection, and the figure beside the title is how far '
    + 'that projection sits above or below his last season.',

  steady: 'How much his scoring moved around week to week inside a season. '
    + 'Each column is his place among every player measured that year, '
    + 'steadiest first, so a tall bar is a player you can plan a lineup '
    + 'around and a short one is boom or bust. It says nothing about how good '
    + 'he was, only how repeatable.',

  schedule: 'Each week of the coming season graded against the defence he '
    + 'faces, on how much that defence gave up to his position last year. '
    + 'Taller and greener is an EASIER week, which is the opposite of a '
    + 'difficulty rank. The figure beside the title places his whole season '
    + 'among the 32 teams, easiest first.',

  depth: 'His own position room on his team, in depth-chart order. The men '
    + 'under him are the ones who take his touches when a coach changes his '
    + 'mind; the men above him are the ones he has to pass.',

  usage: 'How much of the offence he was, season by season: his share of the '
    + 'targets and carries, and the per-game rates behind them. The last '
    + 'column is this season’s projection.',

  oline: 'The five linemen in front of him, placed among all 32 lines. Same '
    + 'five is how much of last season the current starters actually played '
    + 'together, Available is how often those five have been active across '
    + 'their careers, Returning is how much of the line is back, and '
    + 'Experience is their average years in the league.',

  market: 'Where the rest of the world has him. Each row is one site’s '
    + 'own rank with his position rank beside it, and the rows underneath '
    + 'compare their consensus with this board’s rank. A positive figure '
    + 'means the market takes him later than this board would, which is the '
    + 'direction you shop in.',

  vegas: 'What the betting market expects of his offence: where his team '
    + 'ranks by the points its lines imply, and the totals and spreads behind '
    + 'that rank. It is the one reading in this popup that comes from money '
    + 'rather than from a projection.',

  comps: 'Seasons from any year that look like the one he is projected for, '
    + 'scored for closeness, and what each of those players did the season '
    + 'after. Next is the column that matters. The figure beside the title is '
    + 'the average change across every matched season, not just the rows on '
    + 'screen.',

  similar: 'The players closest to him at his position on this year’s '
    + 'board, matched on projection, stat line, age and build, closest first. '
    + 'This is who you would be choosing between now, not who he resembles '
    + 'historically.',

  news: 'Recent stories about him, newest first. Some are matched to him by '
    + 'name rather than tagged by the source, and the label under each '
    + 'headline says which.',
} as const

/** The finish panel, whose title is the position itself. */
export function finishHint(position: string): string {
  return `Where he finished among every ${position} in each season, by this `
    + 'league’s scoring. The count runs from the best down, so a taller '
    + 'bar is a better finish, and the colour is whether that finish would '
    + `start for a team here. The hollow last column is this season’s `
    + 'projected finish.'
}

/** The week-by-week card, whose title is the season it is drawing. */
export function weeksHint(season: number | string): string {
  return `Every week of ${season}: what he did, who he did it against, and `
    + 'what the line scored under this league’s rules. The arrows walk '
    + 'back and forward through the rest of his career.'
}
