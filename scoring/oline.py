"""Offensive-line quality: continuity, availability and experience for a
team's front five.

Nothing here is a "grade." PFF grades, pressure rates, sack attribution and
run-block win rate do not exist in this database. What does exist is who
played, how many snaps they took, and whether they were still there the
week after -- so that is all `line_quality` is built from. See
`oline_id_coverage` for how much of the league those tables actually let
us resolve to a real person.

No lookahead: `line_quality(conn, season)` and `line_units(conn, season)`
read `snap_counts` seasons strictly before `season`, plus `depth_charts`,
which the live pipeline only ever stores as a single current snapshot --
no season column, no history, just "whoever's listed as of the last
scrape." That snapshot stands in for "preseason facts about `season`" and
nothing else; calling this for a season other than the one currently being
drafted reads today's roster against a different season's history, which
is only meaningful because `scoring/config.py`'s CURRENT_SEASON makes the
same "today's snapshot is the current season's" assumption for the rest of
the pipeline.
"""
import re
import unicodedata

import numpy as np
import pandas as pd

from pipeline.db import read_table
from scoring.factors import normalize_within_position

# snap_counts' position labels for offensive linemen. "OL" is the
# unsplit label some seasons use instead of T/G/C; a handful of rows also
# carry "OG"/"OT" instead of "G"/"T" (see oline_id_coverage's docstring for
# how rare that is).
OL_SNAP_POSITIONS = ("T", "G", "C", "OL", "OG", "OT")
# depth_charts' pos_name values for the five line spots that block for the
# skill positions a manager actually drafts.
LINE_SLOTS = ("Left Tackle", "Left Guard", "Center", "Right Guard", "Right Tackle")
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

# How much rookie_season is allowed to postdate a player's first recorded
# snap before a name match is ruled implausible. +1, not 0: a UDFA can log
# a garbage-time snap the summer before nflverse marks his rookie season.
_ROOKIE_SEASON_SLACK = 1

# How many seasons of daylight it takes to call a same-name collision.
#
# Eight is about the length of a career, and it is chosen against both sides
# of the evidence rather than picked round. The three real collisions this
# exists for clear it easily: Aaron Jones by 29 seasons, Marvin Harrison Jr.
# by 28, DJ Moore by 9. The case that must STAY ambiguous is the one
# tests/test_oline.py builds -- a 2022 pfr debut against a 2015 rookie and a
# 2021 rookie, a margin of 6 -- because a player seven years into his career
# really could be taking those snaps, and there is no father-and-son story to
# tell them apart. Below eight, that fixture starts resolving, which is the
# guess this whole path refuses to make.
_ERA_GAP = 8

# Continuity is the best-documented public predictor of line quality (see
# `_continuity`); availability is the user's explicitly-requested
# first-class output and gets nearly as much weight. `returning` and
# `experience` are secondary signals. There is no ground truth to fit these
# to -- same judgment-call status as DEFAULT_WEIGHTS in scoring/config.py.
WEIGHTS = {"continuity": 0.35, "availability": 0.30, "returning": 0.20, "experience": 0.15}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

LINE_UNITS_COLUMNS = [
    "team", "position", "gsis_id", "player_name", "snap_share_last_season",
    "games_played", "team_games_possible", "availability", "seasons_in_league",
]
LINE_QUALITY_COLUMNS = [
    "team", "continuity_raw", "continuity_n", "availability_raw", "availability_n",
    "returning_raw", "returning_n", "experience_raw", "experience_n", "line_quality",
]


def _fold_name(name) -> str:
    """Same normalization as board.py's `_norm_name`, duplicated rather
    than imported: it's six lines, and this module has no other reason to
    depend on the fantasy-board module."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[.'\-]", "", s.lower())
    return " ".join(p for p in s.split() if p not in _SUFFIXES)


def reconcile_pfr_to_gsis(conn, *, positions=OL_SNAP_POSITIONS,
                          snaps: pd.DataFrame | None = None) -> pd.DataFrame:
    """pfr_player_id -> gsis_id for every offensive lineman `snap_counts`
    has ever logged.

    `snap_counts` keys on Pro-Football-Reference ids; every other table
    here keys on `gsis_id`, and there is no crosswalk table between them.
    This joins on folded display name -- the same join `board.py` uses
    between ADP and nflverse for the identical reason (two sources, no
    shared id) -- then drops any candidate whose `rookie_season` postdates
    the player's first recorded snap by more than a year, which resolves
    most collisions between two players who happen to share a name (e.g.
    two "Michael Jordan"s at guard).

    What's still ambiguous after that, or has no name match at all, is
    dropped rather than guessed at. Callers that need to know how much was
    dropped call `oline_id_coverage`, not `len` on this result.

    `positions` NARROWS the snap rows the crosswalk is built from; the
    default is this module's own subject, the five line spots, so every
    existing caller is unchanged. `None` means "every position", which is
    what the player profile needs: it wants a running back's per-game
    `offense_pct`, and a mapping restricted to OL_SNAP_POSITIONS resolves
    849 of the table's 5,890 pfr ids and NONE of the skill players (checked
    against data/nfl.duckdb: Jahmyr Gibbs' `GibbJa01` is absent from the
    default mapping and present in the unfiltered one). Unfiltered is also
    the more accurate `first_season` for anyone whose listed position moved
    between seasons -- the plausibility filter compares `rookie_season`
    against the first snap on record, and a position-filtered scan can only
    ever push that first snap later.

    `snaps` lets a caller that has already read `snap_counts` hand the frame
    over instead of paying for a second read (0.165s on data/nfl.duckdb) --
    the same shape `find_twins(season_features=...)` and
    `season_summaries(snap_share=...)` already take, and for the same
    reason. scoring/profile_cache.py builds this inside a `_build` that has
    the table in hand. `conn` is still required: the `players` half of the
    join is read from it either way.
    """
    if snaps is None:
        snaps = read_table(conn, "snap_counts")
    # Column guard, not just `.empty`: this used to be reached only from
    # line_quality, whose callers all hand it the real nflverse schema. The
    # profile reaches it on every database the app has, including test
    # fixtures whose snap_counts is (player, team, season, offense_pct) and
    # nothing else -- which raised KeyError on `snaps["position"]`. An
    # unresolvable crosswalk is an empty crosswalk, not an exception.
    need = {"pfr_player_id", "player", "season"}
    if positions is not None:
        need.add("position")
    if snaps.empty or not need.issubset(snaps.columns):
        return pd.DataFrame(columns=["pfr_player_id", "gsis_id"])
    ol = snaps if positions is None else snaps[snaps["position"].isin(positions)]
    if ol.empty:
        return pd.DataFrame(columns=["pfr_player_id", "gsis_id"])

    ids = ol[["pfr_player_id", "player"]].drop_duplicates()
    ids["norm"] = ids["player"].map(_fold_name)
    first_season = ol.groupby("pfr_player_id")["season"].min().rename("first_season")
    ids = ids.merge(first_season, on="pfr_player_id")

    people = read_table(conn, "players")
    # Same widened guard as above, same reason: `_experience` already
    # tolerates a `players` without `rookie_season`; this did not.
    if people.empty or not {"gsis_id", "display_name",
                            "rookie_season"}.issubset(people.columns):
        return pd.DataFrame(columns=["pfr_player_id", "gsis_id"])
    people = people[["gsis_id", "display_name", "rookie_season"]].copy()
    people["norm"] = people["display_name"].map(_fold_name)

    merged = ids.merge(people[["gsis_id", "norm", "rookie_season"]], on="norm", how="left")
    plausible = merged["gsis_id"].notna() & (
        merged["rookie_season"].isna()
        | (merged["rookie_season"] <= merged["first_season"] + _ROOKIE_SEASON_SLACK))
    matched = merged[plausible]

    # A name match is only usable if it points at exactly one gsis_id --
    # two candidates surviving the rookie_season filter means the
    # collision is real (both plausible), not a data-quality accident.
    counts = matched.groupby("pfr_player_id")["gsis_id"].nunique()
    unique_ids = counts[counts == 1].index
    out = matched[matched["pfr_player_id"].isin(unique_ids)][["pfr_player_id", "gsis_id"]]

    # The plausibility test above is one-sided: it rejects a candidate whose
    # career began AFTER these snaps, and never one whose career began far
    # too early. So a son collides with his father and both survive as
    # "plausible" -- Marvin Harrison Jr.'s 2024 snaps matched Marvin Harrison
    # (rookie 1996) as readily as himself, and the pair was dropped as
    # ambiguous. Measured on data/nfl.duckdb that cost 7.6% of recent
    # player-seasons their per-game snap share, including Aaron Jones
    # (against a 1988 namesake), DJ Moore (a 2009 D.J. Moore) and Harrison.
    # The card showed an em dash per week and said nothing about why.
    #
    # So: when several candidates survive, the one whose career actually
    # explains these snaps wins -- but only when it is not a close call.
    # `_ERA_GAP` seasons of daylight between the best candidate and the next
    # is the bar, which the three above clear by 29, 9 and 28. Anything
    # tighter stays dropped, because the docstring's promise holds: a real
    # same-era collision (two guards named Michael Jordan) is not something
    # to guess at, and a wrong per-game snap share is a lie told seventeen
    # times rather than a gap admitted once.
    contested = matched[~matched["pfr_player_id"].isin(unique_ids)].copy()
    if not contested.empty:
        contested["_distance"] = (contested["first_season"]
                                  - contested["rookie_season"]).abs()
        contested = contested.sort_values(["pfr_player_id", "_distance"])
        # `cumcount` rather than `nth`: the group key's placement in what
        # `nth` returns has moved between pandas versions, and this is a
        # position within an already-sorted frame either way.
        contested["_place"] = contested.groupby("pfr_player_id").cumcount()
        best = contested[contested["_place"] == 0].set_index("pfr_player_id")
        runner = contested[contested["_place"] == 1].set_index("pfr_player_id")
        margin = runner["_distance"] - best["_distance"].reindex(runner.index)
        decided = margin[margin >= _ERA_GAP].index
        resolved = (best.loc[best.index.isin(decided), ["gsis_id"]]
                    .reset_index()[["pfr_player_id", "gsis_id"]])
        out = pd.concat([out, resolved], ignore_index=True)

    return out.drop_duplicates().reset_index(drop=True)


def oline_id_coverage(conn) -> dict:
    """How much of the league's O-line history actually resolves to a
    `gsis_id`, as plain counts and percentages.

    Reported rather than assumed: `reconcile_pfr_to_gsis` silently drops
    whatever it can't match, so a rating built on top of it needs this
    number in the open rather than buried in a row count that shrank
    quietly.
    """
    snaps = read_table(conn, "snap_counts")
    empty = {"pfr_ids": 0, "resolved_ids": 0, "resolved_pct": 0.0,
             "rows": 0, "rows_resolved": 0, "rows_resolved_pct": 0.0}
    if snaps.empty:
        return empty
    ol = snaps[snaps["position"].isin(OL_SNAP_POSITIONS)]
    if ol.empty:
        return empty

    mapping = reconcile_pfr_to_gsis(conn)
    total_ids = ol["pfr_player_id"].nunique()
    resolved_ids = mapping["pfr_player_id"].nunique()
    total_rows = len(ol)
    rows_resolved = int(ol["pfr_player_id"].isin(mapping["pfr_player_id"]).sum())
    return {
        "pfr_ids": int(total_ids),
        "resolved_ids": int(resolved_ids),
        "resolved_pct": round(resolved_ids / total_ids * 100, 1) if total_ids else 0.0,
        "rows": int(total_rows),
        "rows_resolved": rows_resolved,
        "rows_resolved_pct": round(rows_resolved / total_rows * 100, 1) if total_rows else 0.0,
    }


def _team_games(snap_counts: pd.DataFrame) -> pd.DataFrame:
    """Games a team actually played, by season -- the denominator for every
    availability figure below. Counted from `snap_counts`' own weeks rather
    than hardcoded per era (17 games since 2021, 16 before), which keeps it
    right for a partial current season too."""
    if snap_counts.empty:
        return pd.DataFrame(columns=["team", "season", "team_games"])
    reg = snap_counts[snap_counts["game_type"] == "REG"]
    return reg.groupby(["team", "season"])["week"].nunique().rename(
        "team_games").reset_index()


def _ol_snaps_with_gsis(conn, snaps: pd.DataFrame | None = None) -> pd.DataFrame:
    """Regular-season O-line snap rows, restricted to the ones that
    reconciled to a `gsis_id`. See `oline_id_coverage` for how much of the
    table this drops.

    `snaps` is the already-read `snap_counts`; see `line_quality` for why
    the frame is threaded through rather than read again here."""
    if snaps is None:
        snaps = read_table(conn, "snap_counts")
    cols = ["season", "week", "team", "gsis_id", "offense_snaps", "offense_pct"]
    if snaps.empty:
        return pd.DataFrame(columns=cols)
    ol = snaps[(snaps["position"].isin(OL_SNAP_POSITIONS))
               & (snaps["game_type"] == "REG")].copy()
    mapping = reconcile_pfr_to_gsis(conn, snaps=snaps)
    return ol.merge(mapping, on="pfr_player_id", how="inner")


def _current_starters(conn) -> pd.DataFrame:
    """The five projected starters per team, right now.

    `depth_charts` is a live scrape with no season column and no history --
    every row it has ever held reflects "as of `dt`," so the newest `dt` is
    the only preseason snapshot this database can produce. Ties at the top
    depth rank (a real but rare data glitch) resolve to whichever row sorts
    first, which is arbitrary but affects a handful of rows league-wide.
    """
    empty = pd.DataFrame(columns=["team", "position", "gsis_id", "player_name"])
    dc = read_table(conn, "depth_charts")
    if dc.empty:
        return empty
    dc = dc[dc["pos_name"].isin(LINE_SLOTS)]
    if dc.empty:
        return empty
    latest = dc["dt"].max()
    dc = dc[dc["dt"] == latest]
    starters = (dc.sort_values("pos_rank")
                  .groupby(["team", "pos_name"], as_index=False)
                  .first())
    return starters.rename(columns={"pos_name": "position"})[
        ["team", "position", "gsis_id", "player_name"]]


def _availability(ol_snaps: pd.DataFrame, team_games: pd.DataFrame, season: int) -> pd.DataFrame:
    """Games actually played (offense_snaps > 0) versus games the player's
    team-that-season played, summed across every prior season on record.
    The same `played_share` convention `scoring/player_history.py` uses for
    skill players, extended across seasons rather than kept per-season, so
    a lineman's whole injury history counts, not just his last year."""
    cols = ["gsis_id", "games_played", "team_games_possible", "availability"]
    prior = ol_snaps[ol_snaps["season"] < season]
    if prior.empty:
        return pd.DataFrame(columns=cols)
    played = (prior[prior["offense_snaps"] > 0]
              .groupby(["gsis_id", "team", "season"])["week"].nunique()
              .rename("games").reset_index())
    played = played.merge(team_games, on=["team", "season"], how="left")
    agg = played.groupby("gsis_id").agg(
        games_played=("games", "sum"),
        team_games_possible=("team_games", "sum")).reset_index()
    agg["availability"] = agg["games_played"] / agg["team_games_possible"].replace(0, np.nan)
    return agg[cols]


def _last_season_snap_share(ol_snaps: pd.DataFrame, season: int) -> pd.DataFrame:
    """This player's average offense_pct in the most recent prior season he
    has snap data for -- how big a role he actually held last time he
    played, not just whether he was on the field at all."""
    cols = ["gsis_id", "snap_share_last_season"]
    prior = ol_snaps[ol_snaps["season"] < season]
    if prior.empty:
        return pd.DataFrame(columns=cols)
    latest_season = prior.groupby("gsis_id")["season"].transform("max")
    last = prior[prior["season"] == latest_season]
    return last.groupby("gsis_id")["offense_pct"].mean().rename(
        "snap_share_last_season").reset_index()[cols]


def _experience(conn, season: int) -> pd.DataFrame:
    """Seasons in the league as of `season`, from `players.rookie_season` --
    a fact knowable before the season starts, not a result of it."""
    cols = ["gsis_id", "seasons_in_league"]
    people = read_table(conn, "players")
    if people.empty or "rookie_season" not in people.columns:
        return pd.DataFrame(columns=cols)
    out = people[["gsis_id", "rookie_season"]].copy()
    out["seasons_in_league"] = (season - out["rookie_season"]).clip(lower=0)
    return out[cols]


def _continuity(ol_snaps: pd.DataFrame, season: int) -> pd.DataFrame:
    """Share of a team's O-line snaps in `season` taken by its five most-
    used linemen -- how much of the season a stable core, rather than a
    rotating cast, actually played.

    This is a looser operationalization than the literal "same five started
    every game together": that stricter version was measured (see
    `.superpowers/sdd/oline-quality-report.md`) at r ~= 0.08 against team
    rushing yards/game across 315 reconciled team-seasons (2016-2025) --
    indistinguishable from no correlation. This snap-share version holds up
    a bit better (r ~= 0.24) while still capturing the same idea: five
    semi-permanent starters versus a line in flux. Still a weak
    correlation, not a strong one -- reported plainly rather than
    oversold. Teams with fewer than five O-line snap-takers that season
    come back NaN rather than a share computed against a denominator that
    doesn't mean what it means everywhere else.
    """
    cols = ["team", "continuity_raw"]
    prior = ol_snaps[ol_snaps["season"] == season]
    if prior.empty:
        return pd.DataFrame(columns=cols)
    totals = prior.groupby(["team", "gsis_id"])["offense_snaps"].sum().reset_index()
    team_totals = totals.groupby("team")["offense_snaps"].sum().rename("team_total")
    counts = totals.groupby("team")["gsis_id"].nunique().rename("n_linemen")
    top5 = totals.sort_values("offense_snaps", ascending=False).groupby("team").head(5)
    top5_totals = top5.groupby("team")["offense_snaps"].sum().rename("top5_total")
    out = pd.concat([team_totals, top5_totals, counts], axis=1).reset_index()
    out["continuity_raw"] = np.where(
        out["n_linemen"] >= 5, out["top5_total"] / out["team_total"].replace(0, np.nan), np.nan)
    return out[cols]


def primary_five(ol_snaps: pd.DataFrame, season: int) -> pd.DataFrame:
    """The five O-line gsis_ids who took the most offense snaps for each
    team in `season` -- last season's de facto starting five, used both to
    validate `_continuity` and to measure how much of it survived into the
    current depth chart (`_returning`)."""
    prior = ol_snaps[ol_snaps["season"] == season]
    if prior.empty:
        return pd.DataFrame(columns=["team", "gsis_id", "offense_snaps"])
    totals = prior.groupby(["team", "gsis_id"])["offense_snaps"].sum().reset_index()
    return totals.sort_values("offense_snaps", ascending=False).groupby("team").head(5)


def _returning(current_starters: pd.DataFrame, prior_five: pd.DataFrame) -> pd.DataFrame:
    """How much of last season's featured line is still penciled in.

    Distinct from `_continuity`: that measures stability *within* last
    season, this measures turnover *between* last season's featured five
    and the one the current depth chart shows. Divided by how many of this
    year's starters we actually have a `gsis_id` for, not by a flat 5 --
    a depth-chart row with a missing id is our gap, not evidence the team
    replaced that starter.
    """
    cols = ["team", "returning_raw"]
    if current_starters.empty:
        return pd.DataFrame(columns=cols)
    prior_sets = prior_five.groupby("team")["gsis_id"].apply(set)
    rows = []
    for team, grp in current_starters.groupby("team"):
        current_five = set(grp["gsis_id"].dropna())
        if not current_five:
            rows.append({"team": team, "returning_raw": np.nan})
            continue
        overlap = len(current_five & prior_sets.get(team, set()))
        rows.append({"team": team, "returning_raw": overlap / len(current_five)})
    return pd.DataFrame(rows, columns=cols)


def line_units(conn, season: int, *, snaps: pd.DataFrame | None = None,
               ol_snaps: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per team, the five projected starters for `season`: position, most
    recent snap share, prior-season games played versus possible, career
    availability, and seasons in the league. The detail a manager reads
    before a pick; `line_quality` is the same numbers rolled up to a score.

    `snaps` and `ol_snaps` are the two frames `line_quality` derives on its
    own account anyway; see its docstring for why they are threaded through.
    Passing neither reads them here, which is what every existing caller
    does.
    """
    starters = _current_starters(conn)
    if starters.empty:
        return pd.DataFrame(columns=LINE_UNITS_COLUMNS)

    if snaps is None:
        snaps = read_table(conn, "snap_counts")
    if ol_snaps is None:
        ol_snaps = _ol_snaps_with_gsis(conn, snaps=snaps)
    team_games = _team_games(snaps)
    avail = _availability(ol_snaps, team_games, season)
    share = _last_season_snap_share(ol_snaps, season)
    exp = _experience(conn, season)

    out = starters.merge(avail, on="gsis_id", how="left")
    out = out.merge(share, on="gsis_id", how="left")
    out = out.merge(exp, on="gsis_id", how="left")
    return out[LINE_UNITS_COLUMNS]


def line_quality(conn, season: int, *,
                 snaps: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per team, a 0-100 O-line rating built from continuity, availability,
    experience and returning-starter share, normalized within position the
    same way `scoring/factors.py` normalizes player factors
    (`normalize_within_position`: percentile rank within the group, missing
    -> 50). There is only one "position" here -- a team's O-line unit --
    so the normalization is a league-wide percentile rank, but it is the
    same function and the same missing-is-neutral convention the rest of
    the codebase already commits to, not a bespoke one.

    Every column this returns besides `line_quality` itself is a component
    part (`*_raw` before normalization, `*_n` after), so the score is
    decomposable rather than a single opaque number.

    ONE READ OF `snap_counts`, ONE CROSSWALK BUILD. This used to read the
    table three times (twice through `_ol_snaps_with_gsis`, once through
    `_team_games`) and rebuild the pfr->gsis crosswalk twice, which was
    invisible while nothing but a test called it and is not invisible now
    that scoring/profile.py does: it cost 1.54s of a profile's cold cache
    against 0.76s for the same answer off one read. The frames are threaded
    down through `_ol_snaps_with_gsis`/`line_units` rather than memoized on
    the module, so nothing here holds a 144 MB frame alive between calls.
    `snaps` lets a caller that has already read the table (profile_cache
    has) skip even the one read; the result is unchanged either way --
    verified frame-for-frame against the previous implementation for
    `line_quality`, `line_units` and `reconcile_pfr_to_gsis` across 2024,
    2025 and 2026 on data/nfl.duckdb.
    """
    if snaps is None:
        snaps = read_table(conn, "snap_counts")
    ol_snaps = _ol_snaps_with_gsis(conn, snaps=snaps)
    units = line_units(conn, season, snaps=snaps, ol_snaps=ol_snaps)

    cont = _continuity(ol_snaps, season - 1)
    prior_five = primary_five(ol_snaps, season - 1)
    returning = _returning(units[["team", "gsis_id"]], prior_five)

    avail_team = units.groupby("team")["availability"].mean().rename(
        "availability_raw").reset_index()
    exp_team = units.groupby("team")["seasons_in_league"].mean().rename(
        "experience_raw").reset_index()

    teams = sorted(set(units["team"]) | set(cont["team"]) | set(returning["team"]))
    # dtype="object" even when `teams` is empty -- an empty Python list
    # infers float64, and merging that against the other frames' string
    # "team" columns raises rather than silently coercing.
    out = pd.DataFrame({"team": pd.Series(teams, dtype="object")})
    out = out.merge(cont, on="team", how="left")
    out = out.merge(returning, on="team", how="left")
    out = out.merge(avail_team, on="team", how="left")
    out = out.merge(exp_team, on="team", how="left")

    # normalize_within_position groups by "position"; every row is the same
    # pseudo-position "OL" so the league of teams ranks against itself.
    out["position"] = "OL"
    for raw, norm in (("continuity_raw", "continuity_n"), ("returning_raw", "returning_n"),
                       ("availability_raw", "availability_n"),
                       ("experience_raw", "experience_n")):
        out = normalize_within_position(out, raw, norm)

    out["line_quality"] = (
        out["continuity_n"] * WEIGHTS["continuity"]
        + out["availability_n"] * WEIGHTS["availability"]
        + out["returning_n"] * WEIGHTS["returning"]
        + out["experience_n"] * WEIGHTS["experience"])
    out = out.sort_values("line_quality", ascending=False).reset_index(drop=True)
    return out[LINE_QUALITY_COLUMNS]
