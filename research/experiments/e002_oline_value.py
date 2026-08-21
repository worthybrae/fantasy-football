"""E002 -- does the offensive line show up in the numbers at all?

The board carries an O-line input, and the first thing anyone does with one
is ask whether it predicts a running back's fantasy points. It does not, and
that answer got written down as "the line doesn't matter."

The answer is wrong because the outcome is wrong. Fantasy points are volume
and touchdowns; a back who gets 250 carries scores like a back who gets 250
carries. What a line can plausibly move is what happens on each of those
carries. So this asks the same question twice -- once against points, once
against yards per carry -- and reports both, because the pair is the finding.

The rating itself is not a grade. There is no grade in this database: no
PFF, no pressure rate, no run-block win rate. What there is, is what teams
PAID. `apy_cap_pct` is a contract's annual value as a share of that season's
salary cap, which is a market's own estimate of a lineman's worth, already
inflation-adjusted and already comparable across eras.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.lab import Experiment

OL_CONTRACT_POSITIONS = ("LT", "RT", "LG", "RG", "C")
FIRST_MODEL_SEASON = 2017   # the earliest season with a rated line BEHIND it
MIN_GAMES = 8               # a player-season worth counting, as in e001
MIN_CARRIES = 50            # a real workload: roughly three totes a game
STRICT_CARRIES = 100        # the same test at a harsher cut, as a check


def _priced_seasons(contracts: pd.DataFrame, seasons) -> pd.DataFrame:
    """One row per (lineman, season he was under a given contract).

    Over The Cap stores a contract, not a season: signing year, length, and
    annual value. A deal signed in 2021 for four years is in force for 2021
    through 2024, so it is expanded across those seasons rather than counted
    once at signing -- otherwise three quarters of the league would be
    unpriced in any given year.

    Extensions and re-signings overlap, so a player can land under two live
    contracts at once. The most recently signed one wins: it is the deal
    that actually pays him, and the older one is the thing it replaced.
    """
    lo, hi = min(seasons), max(seasons)
    ol = contracts[contracts["position"].isin(OL_CONTRACT_POSITIONS)]
    ol = ol.dropna(subset=["gsis_id", "apy_cap_pct", "years"])
    # year_signed carries a handful of zeros and other junk; `years > 0`
    # drops rows whose span would otherwise be empty or run backwards.
    ol = ol[(ol["year_signed"] > 1990) & (ol["years"] > 0)]

    rows = []
    for gsis, signed, years, apy in zip(ol["gsis_id"], ol["year_signed"],
                                        ol["years"], ol["apy_cap_pct"]):
        start, end = int(signed), int(signed) + int(years) - 1
        for season in range(max(start, lo), min(end, hi) + 1):
            rows.append((gsis, season, start, float(apy)))
    priced = pd.DataFrame(rows, columns=["gsis_id", "season", "year_signed",
                                         "apy_cap_pct"])
    priced = priced.sort_values("year_signed")
    return priced.groupby(["gsis_id", "season"], as_index=False).last()


def _line_spend(conn, contracts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per team-season: the mean cap share of the five linemen who started.

    "Started" means took the most offensive snaps that season -- realised
    snaps, not a depth chart. `scoring.oline._current_starters` would be the
    wrong tool here and quietly so: `depth_charts` is a single live scrape
    with no season column, so scoring a 2019 line with it would price 2019's
    snaps against today's roster.

    Returns the priced-starter rows too, so coverage can be reported rather
    than assumed -- the mean of whoever happened to have a contract on file
    is only meaningful next to how many of the five that was.
    """
    from scoring.oline import _ol_snaps_with_gsis, primary_five

    snaps = _ol_snaps_with_gsis(conn)
    seasons = sorted(int(s) for s in snaps["season"].unique())
    five = pd.concat([primary_five(snaps, s).assign(season=s) for s in seasons])

    priced = _priced_seasons(contracts, seasons)
    starters = five.merge(priced[["gsis_id", "season", "apy_cap_pct"]],
                          on=["gsis_id", "season"], how="left")

    line = (starters.dropna(subset=["apy_cap_pct"])
            .groupby(["season", "team"])
            .agg(spend=("apy_cap_pct", "mean"), priced=("apy_cap_pct", "size"))
            .reset_index())
    # Within season, because the cap share of a whole line drifts with the
    # cap itself and with how the league happens to be paying linemen that
    # year. A coefficient "per standard deviation" then means per standard
    # deviation of the league a manager is actually drafting out of.
    line["z"] = line.groupby("season")["spend"].transform(
        lambda s: (s - s.mean()) / s.std())
    return line, starters


def _rb_seasons(conn, rules) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Running back production two ways: per player-season, and per team.

    A midseason trade splits a player across two `recent_team` values. For
    the PLAYER frame the team he played the most games for takes the season
    -- the alternative is throwing away every traded back, and the line he
    spent most of the year behind is the line the season mostly measures.
    The TEAM frame keeps both halves where they were actually run, so a
    team's yards per carry counts the carries it actually handed out.

    Backs only, on both sides. Quarterback runs are a large share of the
    modern rushing total and the five are not blocking a scramble the way
    they block a handoff; folding them in halves the coefficient measured
    below (0.033 against 0.072 in the otherwise identical test), which is a
    statement about scrambles rather than about lines.
    """
    from scoring.ppr import compute_ppr_points

    wk = conn.execute("select * from weekly where position = 'RB'").df()
    wk["pts"] = compute_ppr_points(wk, rules)
    by_team = wk.groupby(["season", "player_id", "recent_team"]).agg(
        pts=("pts", "sum"), g=("week", "nunique"),
        carries=("carries", "sum"), rush_yards=("rushing_yards", "sum")).reset_index()
    rb = (by_team.sort_values(["g", "carries"], ascending=False)
          .groupby(["season", "player_id"], as_index=False).first())
    rb["ppg"] = rb["pts"] / rb["g"]
    rb["ypc"] = rb["rush_yards"] / rb["carries"].replace(0, np.nan)

    team = by_team.groupby(["season", "recent_team"]).agg(
        carries=("carries", "sum"), rush_yards=("rush_yards", "sum")).reset_index()
    team = team[team.carries > 0].copy()
    team["ypc"] = team["rush_yards"] / team["carries"]
    return (rb.rename(columns={"recent_team": "team"}),
            team.rename(columns={"recent_team": "team"}))


def _r2(df: pd.DataFrame, y: str, cols: list[str]) -> tuple[float, np.ndarray]:
    """Ordinary least squares, R2 and coefficients. Same shape as e001's."""
    X = np.column_stack([np.ones(len(df))] + [df[c].values for c in cols])
    beta, *_ = np.linalg.lstsq(X, df[y].values, rcond=None)
    resid = df[y].values - X @ beta
    return 1 - (resid ** 2).sum() / ((df[y] - df[y].mean()) ** 2).sum(), beta


def _lag(df: pd.DataFrame, keys: list[str], cols: dict) -> pd.DataFrame:
    """The same measurements, shifted one season forward, ready to merge.

    `cols` maps a column to the name it takes once it means "last year's."
    """
    prev = df[keys + list(cols)].copy()
    prev["season"] += 1
    return prev.rename(columns=cols)


# `odds` is unused here -- E002 needs no betting lines. It stays in the
# signature so every experiment is called the same way by run.py; a lab
# where each experiment invents its own arguments is a lab with a dispatch
# table in it.
def run(conn, odds: pd.DataFrame, rules) -> dict:
    import nfl_data_py as nfl

    line, starters = _line_spend(conn, nfl.import_contracts())
    seasons = sorted(int(s) for s in line["season"].unique())

    # --- 1. coverage: how much of each line we can actually price ---------
    cov = (starters.groupby("season")["apy_cap_pct"]
           .apply(lambda s: s.notna().mean() * 100))
    span = max(seasons) - min(seasons)
    coverage_curve = [[round((s - min(seasons)) / span, 3), round(float(cov[s]), 1)]
                      for s in seasons]

    # --- 2. face validity: does the rating name the right teams? ---------
    recent = line[line.season == max(seasons)].sort_values("spend", ascending=False)
    top, bottom = recent.head(5), recent.tail(3)

    # --- 3. the naive test: fantasy points ------------------------------
    rb, team = _rb_seasons(conn, rules)
    played = rb[rb.g >= MIN_GAMES]
    ppg = played.merge(_lag(played, ["season", "player_id"], {"ppg": "prev_ppg"}),
                       on=["season", "player_id"])
    ppg = ppg.merge(line[["season", "team", "z"]], on=["season", "team"])
    ppg = ppg[ppg.season >= FIRST_MODEL_SEASON]
    r2_ppg_base, _ = _r2(ppg, "ppg", ["prev_ppg"])
    r2_ppg_line, ppg_beta = _r2(ppg, "ppg", ["prev_ppg", "z"])

    # --- 4a. the same test against efficiency ---------------------------
    # Same shape of test, same predictor, one word different in the outcome.
    ypc = team.merge(_lag(team, ["season", "team"], {"ypc": "prev_ypc"}),
                     on=["season", "team"])
    ypc = ypc.merge(line[["season", "team", "z"]], on=["season", "team"])
    ypc = ypc[ypc.season >= FIRST_MODEL_SEASON]
    r2_ypc_base, _ = _r2(ypc, "ypc", ["prev_ypc"])
    r2_ypc_line, ypc_beta = _r2(ypc, "ypc", ["prev_ypc", "z"])

    # How much of the line the lagged term already knows about. A team's
    # spending is over half correlated with its own spending last year, so
    # "last year's yards per carry" is partly a line measurement itself --
    # which is what holds the coefficient above down, and why the movers
    # below see a bigger one.
    lz = line[["season", "team", "z"]]
    auto = lz.merge(_lag(lz, ["season", "team"], {"z": "prev_z"}), on=["season", "team"])
    line_autocorr = float(auto["z"].corr(auto["prev_z"]))

    # --- 4b. the same effect, from backs who changed lines ---------------
    # An independent route to the same effect: hold the player fixed and
    # change the line under him. Nothing is shared with the team regression
    # above except the rating -- different unit, different outcome, different
    # sample. The carry floor applies to BOTH seasons, because a back with
    # thirty totes has a yards-per-carry that is mostly one long run either
    # way, and differencing two of those measures nothing but noise.
    worked = rb[rb.carries >= MIN_CARRIES]
    moves = worked.merge(
        _lag(worked, ["season", "player_id"],
             {"ypc": "prev_ypc", "team": "prev_team", "carries": "prev_carries"}),
        on=["season", "player_id"])
    moves = moves.merge(line[["season", "team", "z"]], on=["season", "team"])
    moves = moves.merge(
        _lag(line, ["season", "team"], {"z": "prev_z"}).rename(
            columns={"team": "prev_team"}), on=["season", "prev_team"])
    moves = moves[moves.team != moves.prev_team].copy()
    moves["d_line"] = moves["z"] - moves["prev_z"]
    moves["d_ypc"] = moves["ypc"] - moves["prev_ypc"]
    r_move = float(moves["d_line"].corr(moves["d_ypc"]))
    move_slope = float(np.polyfit(moves["d_line"], moves["d_ypc"], 1)[0])

    # Same test at double the carry floor. Not a better estimate -- a smaller
    # one -- but a threshold-shopped result would not survive it.
    strict = moves[(moves.carries >= STRICT_CARRIES)
                   & (moves.prev_carries >= STRICT_CARRIES)]
    r_move_strict = float(strict["d_line"].corr(strict["d_ypc"]))

    # --- what a standard deviation is worth in the only unit that pays ---
    # Yards per carry is not a fantasy stat. At the league's own rushing-yard
    # rate, over a starter's own carry count, it is worth this many points
    # across a season -- which is the honest reason test 3 finds nothing.
    # `rules` is None for a league that scores exactly full PPR, which is what
    # normalize_rules returns rather than a copy of the defaults.
    from scoring.ppr import DEFAULT_RULES
    per_yard = float((DEFAULT_RULES if rules is None else rules)
                     .get("rushing_yards", 0.0))
    starter_carries = float(played[played.carries >= MIN_CARRIES]["carries"].median())

    return {
        "seasons": f"{min(seasons)}-{max(seasons)}",
        "first_season": min(seasons),
        "last_season": max(seasons),
        "n_team_seasons": int(len(line)),
        "coverage_min": round(float(cov.min()), 1),
        "coverage_max": round(float(cov.max()), 1),
        "coverage_curve": coverage_curve,
        "recent_season": int(max(seasons)),
        "top_lines": ", ".join(top["team"]),
        "bottom_lines": ", ".join(bottom.sort_values("spend")["team"]),
        "spend_top": round(float(top["spend"].iloc[0]) * 100, 2),
        "spend_median": round(float(recent["spend"].median()) * 100, 2),
        "spend_bottom": round(float(bottom["spend"].min()) * 100, 2),
        "spend_ratio": round(float(top["spend"].iloc[0] / bottom["spend"].min()), 1),
        "n_rb_seasons": int(len(ppg)),
        "r2_ppg_base": round(float(r2_ppg_base), 3),
        "r2_ppg_line": round(float(r2_ppg_line), 3),
        "r2_ppg_gain": round(float(r2_ppg_line - r2_ppg_base), 4),
        "ppg_per_sd": round(float(ppg_beta[-1]), 2),
        "n_team_ypc": int(len(ypc)),
        "r2_ypc_base": round(float(r2_ypc_base), 3),
        "r2_ypc_line": round(float(r2_ypc_line), 3),
        "r2_ypc_gain": round(float(r2_ypc_line - r2_ypc_base), 4),
        "ypc_per_sd": round(float(ypc_beta[-1]), 3),
        "line_autocorr": round(line_autocorr, 2),
        "n_moves": int(len(moves)),
        "min_carries": MIN_CARRIES,
        "r_move": round(r_move, 3),
        "ypc_per_sd_move": round(move_slope, 3),
        "n_moves_strict": int(len(strict)),
        "strict_carries": STRICT_CARRIES,
        "r_move_strict": round(r_move_strict, 3),
        "starter_carries": int(starter_carries),
        "season_points_per_sd": round(float(ypc_beta[-1]) * starter_carries * per_yard, 1),
        "season_points_per_sd_move": round(move_slope * starter_carries * per_yard, 1),
        "_notes": [
            "Cap share measures what a team PAID, not how the five played. It "
            "underrates any line built on rookie contracts -- five good young "
            "starters on rookie deals price out near the bottom of the league, "
            "and that is a real error in the rating, not noise.",
            "The five starters are read off snaps the season actually produced, "
            "which is hindsight. At a draft you have a projected line, and the "
            "projection is wrong often enough (injury, camp battle, a guard "
            "kicked out to tackle) that the live version of this rating is "
            "weaker than the one measured here.",
            "Contracts come from Over The Cap via nfl_data_py at run time and "
            "are joined on gsis_id; linemen with no gsis_id on their contract "
            "row are simply unpriced, which is most of what the coverage "
            "number is measuring.",
        ],
    }


EXPERIMENT = Experiment(
    id="e002",
    title="The offensive line pays in yards, not points",
    question="Does offensive line quality affect player production -- and if it "
             "does, which measurement of production shows it?",
    run=run,
    tags=("oline", "contracts", "efficiency", "composite"),
    overturns="The project's first pass tested the O-line rating against running "
              "back fantasy points, found nothing, and concluded the line does not "
              "matter. It measured the wrong outcome: the effect is on yards per "
              "carry, and fantasy points bury it under volume and touchdowns.",
)
