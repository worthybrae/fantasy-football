"""League structure, derived from ESPN settings when available.

Everything the board and simulator need to know about league shape lives in
one immutable `LeagueSettings`. Without an ESPN import, `load` returns
`default_settings()`, which reproduces the constants in `scoring/config.py`
exactly -- so a database with no `league` table produces the board it always
produced.
"""
import json
from dataclasses import dataclass, asdict

from pipeline.db import read_table
from scoring.config import (FLEX_SHARES, LEAGUE_TEAMS, REPLACEMENT_RANK,
                            STREAMED_REPLACEMENT_RANK)
from scoring.ppr import DEFAULT_RULES

# ESPN scoring statId -> the nflverse weekly column(s) it scores. A statId
# that fans out to several columns gives each of them the same points, which
# is how one ESPN stat is expressed when nflverse splits it (ESPN carries one
# "fumble lost" where nflverse splits it by how the fumble happened, so 72
# becomes three columns; ESPN counts a blocked kick as a miss where nflverse
# keeps `_missed` and `_blocked` apart, so 85 becomes two).
#
# HOW THE KICKING IDS BELOW WERE ESTABLISHED -- not from memory, and not from
# the point values, which only ever suggest. Every one was measured:
#
#   1. ESPN's own 2025 actuals were read back per player, keyed by statId,
#      from the read-only league view (`?view=kona_player_info`, the same
#      host pipeline/espn_league.py already fetches) for the 60 most-owned
#      kickers, and joined by name to this database's `weekly` table.
#   2. Each candidate column was compared row for row against ESPN's number
#      for all 40 kickers that joined. A mapping is written here ONLY where
#      it agreed on 40 of 40. Anything that did not is listed under
#      DELIBERATELY NOT MAPPED below, with the measurement that rejected it.
#   3. The whole set was then checked end to end: rebuilding two real
#      kickers' 2025 season totals from these columns and pricing them with
#      the owner's own ESPN scoring reproduces ESPN's published season total
#      to the decimal -- Brandon Aubrey 184.6 and Joshua Karty 53.0, exactly.
#      tests/test_ppr.py pins both against the real weekly rows.
#
# The 0-39 and 50+ bands are ESPN's own buckets, not a choice made here: 80
# is a single item that ESPN computes as every field goal under 40 yards
# (verified 40/40 against nflverse's three sub-buckets summed), and 74 is
# every field goal from 50 out (verified 40/40 against 50-59 plus 60+).
# ESPN publishes 74 AND its finer 198/201 split in the same payload, so a
# league that priced both would double-count in ESPN's own scoring too;
# mirroring ESPN is the correct behaviour, not a bug to guard against. The
# owner's league prices 198/201 and not 74.
#
# A kicking item's points come from `points`, NOT from `pointsOverrides` --
# see `from_espn` for why that distinction is the whole reason defenses are
# still unmapped.
ESPN_STAT_COLUMNS = {
    3: ["passing_yards"], 4: ["passing_tds"], 20: ["passing_interceptions"],
    24: ["rushing_yards"], 25: ["rushing_tds"],
    53: ["receptions"], 42: ["receiving_yards"], 43: ["receiving_tds"],
    72: ["sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost"],
    19: ["passing_2pt_conversions"], 26: ["rushing_2pt_conversions"],
    44: ["receiving_2pt_conversions"],

    # -- kicking, all verified 40/40 as described above --
    74: ["fg_made_50_59", "fg_made_60_"],           # FG made 50+
    77: ["fg_made_40_49"],                          # FG made 40-49
    80: ["fg_made_0_19", "fg_made_20_29",           # FG made 0-39
         "fg_made_30_39"],
    83: ["fg_made"],                                # FG made, any distance
    84: ["fg_att"],                                 # FG attempted
    # ESPN's "FG missed" INCLUDES blocked kicks; nflverse does not (measured:
    # `fg_missed` alone disagreed with ESPN for 16 of the 40 kickers, every
    # one of them by exactly their blocked count, and `fg_missed +
    # fg_blocked` agreed 40/40). The owner's league prices this at -1.0, so
    # getting it wrong would silently credit a kicker a point per blocked
    # kick -- Joshua Karty's 2025 has two of them.
    85: ["fg_missed", "fg_blocked"],                # FG missed (blocked counts)
    86: ["pat_made"],                               # PAT made
    87: ["pat_att"],                                # PAT attempted
    88: ["pat_missed", "pat_blocked"],              # PAT missed (blocked counts)
    198: ["fg_made_50_59"],                         # FG made 50-59
    201: ["fg_made_60_"],                           # FG made 60+
}

# WHAT WAS IDENTIFIED AND DELIBERATELY NOT MAPPED, with the reason for each.
# These stay in `unmapped_scoring`, which is the existing "we see it and we
# do not score it" channel and is already surfaced through the API
# (api/main.py). Silence would be the failure mode here, not the omission --
# a scoring rule that is wrong mis-prices a whole position invisibly, while
# an id sitting in `unmapped_scoring` is visible and can be fixed later.
#
#   * 75, 78, 81, 199, 202 (FG attempted, per distance band): nflverse
#     carries made and missed per band but no attempts per band, and
#     attempts cannot be rebuilt from them because the band a BLOCKED kick
#     belongs to is not recorded (only the bandless `fg_blocked` is).
#   * 76, 79, 82, 200, 203 (FG missed, per distance band): same gap seen
#     from the other side. ESPN's per-band miss counts include blocked kicks
#     (measured: they disagreed with nflverse's per-band `fg_missed_*` for
#     8, 9, 6, 7 and 1 of the 40 kickers respectively -- exactly the kickers
#     with blocks), and there is no per-band blocked column to add back.
#   * 63 (offensive fumble-recovery TD), 101 (kickoff-return TD),
#     102 (punt-return TD), 103 (interception-return TD),
#     104 (opponent-fumble-recovery TD): identified by matching ESPN's 2025
#     actuals to named players in this database -- e.g. Ray Davis (31
#     kickoff returns, no punt returns) carries 101; Parker Washington and
#     Chimere Dike (punt returners) carry 102 twice each, matching their
#     `special_teams_tds`; Tyler Lockett carries 104 and has exactly one
#     `fumble_recovery_tds` off one `fumble_recovery_opp`. They are NOT
#     mapped because nflverse collapses them: `special_teams_tds` is one
#     column covering both 101 and 102, and `def_tds` covers both 103 and
#     104, so a league pricing a pair would score every such touchdown
#     twice. A double-counted 6-point rule is worse than a missing one.
#     (This also corrects a stale note in tests/test_league.py, which said
#     101 is unmappable because nflverse does not carry it. It does carry
#     it; the problem is that it cannot tell 101 from 102.)
#   * 89-92, 121-125, 187-195 (points-allowed tiers) and 120/187 (points
#     allowed): the tier families are real -- for all 32 defenses the nine
#     ids in each family sum to exactly 17, the games played -- but this
#     database CANNOT compute points allowed. `schedules` holds only the
#     upcoming season and every `home_score`/`away_score` in it is null
#     (verified: 0 of 272 rows scored), and reconstructing each team's score
#     from `weekly` scoring plays reproduced ESPN's season total for only 8
#     of 32 teams, off by as much as 38 points. Which tier a game falls in
#     therefore cannot be determined at all, let alone the boundaries.
#   * 128-136 (yards-allowed tiers) and 127 (yards allowed): these ARE fully
#     derivable and were verified exactly -- opponents' `passing_yards +
#     rushing_yards + sack_yards_lost` reproduces ESPN's season yards
#     allowed for 31 of 32 teams (the last off by 5), and bucketing that
#     per game at <100/100-199/200-299/300-349/350-399/400-449/450-499/
#     500-549/550+ reproduces all 288 of ESPN's per-team tier counts. They
#     are still not mapped, because they are not what this map is: a tier is
#     a step function over a per-GAME team aggregate, and every rule here is
#     `column x points` summed over a player's rows. See `from_espn`.
#   * 93 (blocked-kick-return TD), 95 (interceptions), 96 (fumble
#     recoveries), 97 (blocked kicks), 98 (safeties), 99 (sacks), 206, 209:
#     team-defense stats with no DST row to attach to (nflverse `weekly` has
#     no team-defense rows at all -- only individual defenders, who are not
#     on this board). Summing individual defenders by team gets close but
#     not exact anyway: interceptions matched ESPN for all 32 teams, but
#     sacks for only 25, fumble recoveries 28 and safeties 26. 209 could not
#     be identified at all -- no player and no defense in the 2025 payload
#     carries a non-zero value for it.

_FLEX_POSITIONS = ("RB", "WR", "TE")


@dataclass(frozen=True)
class LeagueSettings:
    season: int
    teams: int
    starters: dict
    flex_slots: int
    bench: int
    scoring: dict
    draft_type: str
    pick_order: tuple = ()
    unmapped_scoring: tuple = ()

    @property
    def rounds(self) -> int:
        return sum(self.starters.values()) + self.flex_slots + self.bench

    @property
    def replacement_ranks(self) -> dict:
        """Last starter-caliber player at each position.

        `teams * starters + share of the league's flex slots`, plus a
        one-player buffer for positions no flex slot accepts -- without it a
        single-slot position like QB would put replacement level at the very
        last startable player, which is a cliff rather than a baseline.

        K and DST do not come from that arithmetic at all. They are a flat
        calibration (STREAMED_REPLACEMENT_RANK, with the full argument and
        the measured before/after in scoring/config.py) because a position
        that is streamed off waivers every week has no "last starter-caliber
        player" in the sense the rest of this rule means -- two-thirds of
        both positions is free all season. Deliberately NOT derived from
        `teams`: the streaming pool is deep at any league size this tool
        will see, so team count is not what binds. Every other position is
        untouched by it.
        """
        total_flex = self.teams * self.flex_slots
        out = {}
        for pos, n in self.starters.items():
            base = self.teams * n
            if pos in STREAMED_REPLACEMENT_RANK:
                out[pos] = STREAMED_REPLACEMENT_RANK[pos]
            elif pos in _FLEX_POSITIONS:
                out[pos] = base + round(total_flex * FLEX_SHARES.get(pos, 0.0))
            else:
                out[pos] = base + 1
        return out


def from_espn(settings: dict) -> LeagueSettings:
    """LeagueSettings from ESPN's parsed `settings` payload.

    A scoring item's value is read from `points`. That is right for every
    rule this map carries, and it was checked rather than assumed: pricing
    two real kickers' 2025 seasons with the owner's league's own `points`
    values reproduces ESPN's published season totals exactly (184.6 and
    53.0).

    WHY DEFENSES ARE NOT HERE, which is the thing to know before adding
    them. Reading the owner's live ESPN settings turned up a second field
    this parser has never looked at: `pointsOverrides`, a map from LINEUP
    SLOT id to points. Every team-defense item in that league carries
    `points: 0.0` with the real value hidden in `pointsOverrides["16"]` --
    16 being the D/ST slot (pipeline/espn_league.ESPN_SLOT_POSITIONS). So a
    defense's scoring is not merely unmapped here, it is invisible to a
    parser that reads `points`: mapping those ids without also reading the
    override would quietly price every defensive rule at zero.

    Reading the override is not the missing piece either, and this is the
    real blocker. Two of the largest DST components -- points allowed and
    yards allowed -- are scored by TIER: nine ids, each worth points for
    each GAME whose team total fell in a band. That is a step function over
    a per-game team aggregate, and this map's contract is `column x points`
    summed over a player's own rows. There is no honest way to express one
    as the other, and no team-defense row in `weekly` to sum over in the
    first place. See ESPN_STAT_COLUMNS above for which defensive ids are
    identified, which are computable from this data (yards allowed: all of
    them, verified exactly) and which are not (points allowed: none of
    them, because this database holds no game scores).
    """
    from pipeline.espn_league import (ESPN_SLOT_POSITIONS, ESPN_FLEX_SLOT,
                                      ESPN_BENCH_SLOT)
    slots = {int(k): int(v) for k, v in (settings.get("lineup_slots") or {}).items()}
    starters = {pos: slots.get(slot_id, 0)
                for slot_id, pos in ESPN_SLOT_POSITIONS.items()}
    starters = {pos: n for pos, n in starters.items() if n > 0}

    scoring, unmapped = {}, []
    for item in settings.get("scoring_items") or []:
        cols = ESPN_STAT_COLUMNS.get(item.get("statId"))
        if not cols:
            unmapped.append(str(item.get("statId")))
            continue
        for col in cols:
            scoring[col] = float(item.get("points") or 0.0)

    return LeagueSettings(
        season=settings["season"],
        teams=settings["teams"],
        starters=starters,
        flex_slots=slots.get(ESPN_FLEX_SLOT, 0),
        bench=slots.get(ESPN_BENCH_SLOT, 0),
        scoring=scoring,
        draft_type=settings.get("draft_type") or "SNAKE",
        pick_order=tuple(settings.get("pick_order") or ()),
        unmapped_scoring=tuple(unmapped),
    )


def default_settings() -> LeagueSettings:
    """Today's hardcoded league, expressed as a LeagueSettings.

    Kept in sync with REPLACEMENT_RANK by a test, not by discipline.
    """
    return LeagueSettings(
        season=0, teams=LEAGUE_TEAMS,
        starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1},
        flex_slots=2, bench=5, scoring=dict(DEFAULT_RULES),
        draft_type="SNAKE",
    )


def scoring_format(settings: "LeagueSettings | None") -> str:
    """League scoring format as one of `'ppr'` | `'half'` | `'std'`.

    The market consensus (scoring.market) picks each ADP source's rows for
    the league's format, so a half-PPR or standard league sees a consensus
    ADP built from that format's drafts rather than always PPR's. Only the
    reception point value separates the three common formats, so it is the
    whole signal:

        pts >= 0.75         -> 'ppr'   (full-point PPR, the real league's 1.0)
        0.25 <= pts < 0.75  -> 'half'
        else                -> 'std'   (0-point, standard)

    Unknown settings stay PPR -- today's behavior. `None` is "no ESPN import"
    and an empty `scoring` dict is `from_espn` finding nothing that maps
    (see scoring.ppr on why {} is not silently treated as full PPR
    *scoring*); for *format* selection, though, both mean "assume PPR", which
    is the format every source has always been read as.
    """
    if settings is None or not settings.scoring:
        return "ppr"
    pts = settings.scoring.get("receptions", 0)
    if pts >= 0.75:
        return "ppr"
    if pts >= 0.25:
        return "half"
    return "std"


def to_json(settings: LeagueSettings) -> str:
    return json.dumps(asdict(settings))


def from_json(blob: str) -> LeagueSettings:
    d = json.loads(blob)
    d["pick_order"] = tuple(d.get("pick_order") or ())
    d["unmapped_scoring"] = tuple(d.get("unmapped_scoring") or ())
    return LeagueSettings(**d)


def load(conn) -> LeagueSettings:
    table = read_table(conn, "league")
    if table.empty:
        return default_settings()
    newest = table.sort_values("season").iloc[-1]
    return from_json(newest["settings_json"])
