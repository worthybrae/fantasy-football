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

# ESPN scoring statId -> the nflverse weekly column(s) it scores. ESPN carries
# one "fumble lost" stat where nflverse splits it by how the fumble happened,
# so 72 fans out to all three and each gets the same points.
ESPN_STAT_COLUMNS = {
    3: ["passing_yards"], 4: ["passing_tds"], 20: ["passing_interceptions"],
    24: ["rushing_yards"], 25: ["rushing_tds"],
    53: ["receptions"], 42: ["receiving_yards"], 43: ["receiving_tds"],
    72: ["sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost"],
    19: ["passing_2pt_conversions"], 26: ["rushing_2pt_conversions"],
    44: ["receiving_2pt_conversions"],
}

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
