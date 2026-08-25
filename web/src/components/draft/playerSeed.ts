import type { BoardPlayer, LiveCandidate, Player } from '../../api'
import type { ProfileSeed } from '../PlayerProfile'

// What the room already knows about a player, turned into the header +
// figure row PlayerProfile paints on the frame the overlay opens (see
// ProfileSeed's own comment for why that matters: the profile request is a
// measured 3.5s, and the pick clock is 30).
//
// Three builders because the room holds three different rows about the same
// player and they carry different things: the ranked list's LiveCandidate
// (gain vs waiting, survival, which roster slot he fills -- the numbers the
// pick is actually made on, but no name or team at all), the board grid's
// BoardPlayer (a landed pick: what he cost against ADP, his VOR, his
// projection), and the one-time /api/players join table's Player (identity,
// tier, ADP, the model's own rank). Each seeds what it knows and no more --
// a figure nobody can source is left out rather than dashed in.

// duplicated from AvailableList.tsx/TopThree.tsx/ConfirmPick.tsx -- same
// rule and same reason as posBadge is duplicated across four views here: a
// three-line pure function isn't worth a shared module in this codebase.
// `null | undefined` rather than just null, and `== null` rather than
// `=== null`: these read a JSON payload, and a field the server left out
// arrives as undefined, which walks past a `=== null` guard and turns into
// `NaN` here (or a throw at `.toFixed` below). See the seed builder's own
// note -- this is a click handler, and a throw in one just makes the click
// do nothing at all.
function fmtSigned(n: number | null | undefined): string {
  if (n == null) return '—'
  const r = Math.round(n)
  return r > 0 ? `+${r}` : `${r}`
}

// Same shape as AvailableList's own fmtRank -- one decimal only when the
// consensus ADP isn't a whole number.
function fmtRank(n: number | null | undefined): string {
  if (n == null) return '—'
  return Number.isInteger(n) ? String(n) : n.toFixed(1)
}

function fmtTier(tier: number | null | undefined): string {
  return tier === null || tier === undefined ? '—' : `T${tier}`
}

// Same rule as `.confirm-figure.is-open` -- FLEX
// counts as a starting slot, `BENCH` and gain.py's `—` do not. (The
// available table's own copy of this rule went with its Roster slot column.)
function fillsIsOpenSlot(fills: string | null): boolean {
  return fills !== null && fills !== 'BENCH' && fills !== '—'
}

// A row of the ranked available list (the Available tab, and the top three
// cards above it). `player` is the join-table row for the same id, which may
// legitimately be missing: the list is keyed on player_id and renders that
// id as the name until /api/players resolves, exactly as AvailableList and
// TopThree already do.
export function seedFromCandidate(c: LiveCandidate, player: Player | undefined): ProfileSeed {
  return {
    name: player?.name ?? c.player_id,
    position: c.position,
    team: player?.team ?? null,
    bye: player?.bye ?? null,
    rookie: player?.rookie ?? false,
    // Same captions as TopThree's cards and ConfirmPick's dialog, word
    // for word -- see AvailableList.tsx's comment for what each means.
    figures: [
      { label: 'Proj', value: String(Math.round(c.proj_points)) },
      { label: 'Over replacement', value: fmtSigned(c.vor_points) },
      { label: 'Gain vs waiting', value: fmtSigned(c.gain_now) },
      // Deliberately not "Lasts to pick N" here: the horizon that number is
      // measured against is a sentence DraftRoom builds, and a figure
      // caption in a 6-column strip has no room for it. The list this
      // overlay opened from carries the labelled version.
      { label: 'He lasts', value: c.survive_pct === null ? '—' : `${Math.round(c.survive_pct)}%` },
      { label: 'Roster slot', value: c.fills ?? '—', accent: fillsIsOpenSlot(c.fills) },
      { label: 'ADP', value: fmtRank(player?.market_rank ?? null) },
      { label: 'Tier', value: fmtTier(player?.tier) },
    ],
  }
}

// A landed pick on the snake board. No gain/survival/fills: he is already
// drafted, so there is no pick to gain anything by, and inventing those
// three from the ranked list would be describing a decision nobody is
// making. `value` is signed against ADP -- positive means he fell (a
// steal), the same convention DraftBoardGrid's own cells draw.
export function seedFromBoardPlayer(p: BoardPlayer, player: Player | undefined): ProfileSeed {
  return {
    name: p.name,
    position: p.position,
    team: p.team,
    bye: p.bye,
    rookie: player?.rookie ?? false,
    figures: [
      // `== null`, not `=== null`: a payload that omits the field entirely
      // (api/demo.py did) sends `undefined`, and `undefined.toFixed(1)`
      // throws inside whichever click handler called this -- which reads, to
      // anyone using the room, as a pick that simply will not open.
      { label: 'Proj/g', value: p.proj_ppg == null ? '—' : p.proj_ppg.toFixed(1) },
      { label: 'Over replacement', value: fmtSigned(p.vor) },
      { label: 'ADP', value: fmtRank(p.market_rank) },
      { label: 'vs ADP', value: fmtSigned(p.value) },
      { label: 'Tier', value: fmtTier(p.tier) },
    ],
  }
}

// The join table alone -- the only source available when the overlay swaps
// itself to a similar player from inside the profile, where all that is
// known is an id.
export function seedFromPlayer(p: Player): ProfileSeed {
  return {
    name: p.name,
    position: p.position,
    team: p.team,
    bye: p.bye,
    rookie: p.rookie,
    figures: [
      { label: 'Rank', value: `#${p.rank}` },
      { label: 'Tier', value: fmtTier(p.tier) },
      { label: 'ADP', value: fmtRank(p.market_rank) },
      { label: 'Edge', value: fmtSigned(p.edge) },
    ],
  }
}
