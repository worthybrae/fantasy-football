"""E003 -- are the parts of a projection more predictable than the whole?

Every serious projection system is built bottom-up: project team volume,
project a player's share of it, project games played, multiply. The stated
reason is almost always that the PARTS are more stable year to year than the
fantasy points they combine into, so a model built on them starts from firmer
ground.

That is a claim about correlations, and this data has the correlations. It
asks three things in order:

  1. how persistent is each part -- team volume, player share, games played?
  2. how persistent are the points those parts are supposed to explain?
  3. if the parts lose, is there anywhere decomposition still wins?

The first two have the same answer for every position tested, and it is not
the one the rationale predicts. The third is where the case for decomposition
actually lives.
"""
from __future__ import annotations

import pandas as pd

from research.lab import Experiment

MIN_GAMES = 8      # a player-season worth counting, as in e001
MOVE_MIN_GAMES = 6 # looser, because the same/moved split needs both seasons
SKILL = ("QB", "RB", "WR", "TE")

# Volume columns that a team-level projection would have to forecast first.
TEAM_VOLUME = ("attempts", "passing_yards", "carries", "rushing_yards")


def _r(df: pd.DataFrame, a: str, b: str) -> float:
    return round(float(df[a].corr(df[b])), 3)


def _pair(df: pd.DataFrame, keys: list[str], cols: list[str]) -> pd.DataFrame:
    """Join each row to the same key one season earlier.

    Shifting the season forward on a copy and merging is the whole of
    "year over year" here: no row survives without both seasons, which is
    exactly the population a persistence correlation is about.
    """
    prev = df[keys + cols].copy()
    prev["season"] += 1
    prev = prev.rename(columns={c: f"prev_{c}" for c in cols})
    return df.merge(prev, on=keys)


def _player_seasons(weekly: pd.DataFrame, rules) -> pd.DataFrame:
    """One row per player-season: points, games, usage, share, team.

    A player traded mid-season is credited to the team he played the most
    games for and keeps ALL his production, so his share is measured against
    one team's volume for the whole year. Wrong for a few dozen rows out of
    several thousand, and the alternative -- splitting a player into two
    part-seasons -- would break the year-over-year pairing this is for.
    """
    from scoring.ppr import compute_ppr_points
    wk = weekly.copy()
    wk["pts"] = compute_ppr_points(wk, rules)

    team = wk.groupby(["season", "team"])[list(TEAM_VOLUME)].sum().reset_index()
    team = team.rename(columns={"attempts": "team_att", "carries": "team_car"})

    by_team = (wk.groupby(["season", "player_id", "team"])["week"].nunique()
                 .rename("gp_team").reset_index())
    main = (by_team.sort_values("gp_team")
                   .groupby(["season", "player_id"]).tail(1)[["season", "player_id", "team"]])

    pl = wk.groupby(["season", "player_id"]).agg(
        pts=("pts", "sum"), gp=("week", "nunique"),
        targets=("targets", "sum"), carries=("carries", "sum")).reset_index()
    pos = (wk.sort_values("week").groupby(["season", "player_id"])["position"]
             .last().reset_index())
    pl = (pl.merge(pos, on=["season", "player_id"])
            .merge(main, on=["season", "player_id"])
            .merge(team[["season", "team", "team_att", "team_car"]], on=["season", "team"]))

    pl["ppg"] = pl.pts / pl.gp
    # Target share against team PASS ATTEMPTS, not team targets: attempts is
    # what a volume projection actually forecasts, and the two differ only by
    # throwaways.
    pl["share"] = pl.targets / pl.team_att
    # Backs are the same question asked of the run game. Denominator includes
    # quarterback scrambles, which is what "team carries" means everywhere else.
    rb = pl.position == "RB"
    pl.loc[rb, "share"] = pl.loc[rb, "carries"] / pl.loc[rb, "team_car"]
    return pl


def run(conn, odds, rules) -> dict:
    # `odds` is unused here -- no part of this touches betting markets. It is
    # in the signature anyway so every experiment is called the same way by
    # run.py, rather than each one declaring its own arity.
    weekly = conn.execute("select * from weekly").df().rename(
        columns={"recent_team": "team"})
    pl = _player_seasons(weekly, rules)

    # --- 1. team volume: the bottom layer of any decomposition ------------
    team = weekly.groupby(["season", "team"])[list(TEAM_VOLUME)].sum().reset_index()
    tp = _pair(team, ["season", "team"], list(TEAM_VOLUME))
    team_r = {c: _r(tp, c, f"prev_{c}") for c in TEAM_VOLUME}

    # --- 2. share vs the points it is supposed to explain ------------------
    # Same players, same seasons, same threshold on both sides: if the share
    # were measured on a wider population than the points, the comparison
    # would be between two different samples and would mean nothing.
    strong = pl[pl.gp >= MIN_GAMES]
    share_r, ppg_r, total_r, n_pos = {}, {}, {}, {}
    for pos in ("WR", "RB", "TE"):
        d = strong[strong.position == pos]
        p = _pair(d, ["season", "player_id"], ["share", "ppg", "pts"])
        n_pos[pos] = int(len(p))
        share_r[pos] = _r(p, "share", "prev_share")
        ppg_r[pos] = _r(p, "ppg", "prev_ppg")
        total_r[pos] = _r(p, "pts", "prev_pts")

    # --- 3. games played, the third factor --------------------------------
    # Reported at a 6-game floor, and at two others, because this one number
    # moves with the floor more than any other in the experiment: a tighter
    # filter throws away the healthy-then-hurt seasons that carry the signal.
    def games_r(floor: int) -> tuple[float, int]:
        d = pl[(pl.gp >= floor) & (pl.position.isin(SKILL))]
        p = _pair(d, ["season", "player_id"], ["gp"])
        return _r(p, "gp", "prev_gp"), int(len(p))

    games, n_games = games_r(MOVE_MIN_GAMES)
    games_open, _ = games_r(1)
    games_tight, _ = games_r(MIN_GAMES)

    # --- 4. where decomposition still earns its keep ----------------------
    # Split the pairs by whether the player is on the same team as last year.
    # A top-down model cannot see the difference; a decomposed one can, because
    # a new team is a new denominator and a new share.
    def move_split(d: pd.DataFrame) -> dict:
        p = _pair(d, ["season", "player_id"], ["team", "ppg", "pts"])
        p["same"] = p.team == p.prev_team
        out = {}
        for flag, key in ((True, "same"), (False, "moved")):
            s = p[p.same == flag]
            out[f"{key}_ppg"] = _r(s, "ppg", "prev_ppg")
            out[f"{key}_total"] = _r(s, "pts", "prev_pts")
            out[f"n_{key}"] = int(len(s))
        return out

    loose = pl[pl.gp >= MOVE_MIN_GAMES]
    within = {pos: move_split(loose[loose.position == pos]) for pos in ("WR", "RB", "TE")}
    # The pooled version of the same split. Kept because it is the figure that
    # gets quoted, and reported next to the within-position ones so it cannot
    # be mistaken for them: pooling every position mixes quarterbacks with
    # kickers -- who score nothing under this league's rules -- and a wider
    # spread of true talent raises every correlation in it.
    pooled = move_split(loose)

    return {
        "seasons": f"{int(team.season.min())}-{int(team.season.max())}",
        "n_team_pairs": int(len(tp)),
        "r_team_attempts": team_r["attempts"],
        "r_team_pass_yards": team_r["passing_yards"],
        "r_team_carries": team_r["carries"],
        "r_team_rush_yards": team_r["rushing_yards"],

        "min_games": MIN_GAMES,
        "n_wr": n_pos["WR"], "n_rb": n_pos["RB"], "n_te": n_pos["TE"],
        "r_share_wr": share_r["WR"], "r_ppg_wr": ppg_r["WR"], "r_total_wr": total_r["WR"],
        "r_share_rb": share_r["RB"], "r_ppg_rb": ppg_r["RB"], "r_total_rb": total_r["RB"],
        "r_share_te": share_r["TE"], "r_ppg_te": ppg_r["TE"], "r_total_te": total_r["TE"],
        "share_gap_wr": round(ppg_r["WR"] - share_r["WR"], 3),
        "share_gap_rb": round(ppg_r["RB"] - share_r["RB"], 3),
        "share_gap_te": round(ppg_r["TE"] - share_r["TE"], 3),

        "move_min_games": MOVE_MIN_GAMES,
        "r_games": games, "n_games": n_games,
        "r_games_no_floor": games_open, "r_games_tight": games_tight,

        "wr_same_ppg": within["WR"]["same_ppg"], "wr_moved_ppg": within["WR"]["moved_ppg"],
        "wr_same_total": within["WR"]["same_total"], "wr_moved_total": within["WR"]["moved_total"],
        "rb_same_ppg": within["RB"]["same_ppg"], "rb_moved_ppg": within["RB"]["moved_ppg"],
        "rb_same_total": within["RB"]["same_total"], "rb_moved_total": within["RB"]["moved_total"],
        "te_same_ppg": within["TE"]["same_ppg"], "te_moved_ppg": within["TE"]["moved_ppg"],
        "te_same_total": within["TE"]["same_total"], "te_moved_total": within["TE"]["moved_total"],
        "n_wr_moved": within["WR"]["n_moved"], "n_rb_moved": within["RB"]["n_moved"],
        "n_te_moved": within["TE"]["n_moved"],

        "pooled_same_ppg": pooled["same_ppg"], "pooled_moved_ppg": pooled["moved_ppg"],
        "pooled_same_total": pooled["same_total"], "pooled_moved_total": pooled["moved_total"],
        "n_pooled_same": pooled["n_same"], "n_pooled_moved": pooled["n_moved"],

        "_notes": [
            "Persistence is not usefulness, and this experiment measures only "
            "persistence. Decomposition's real value is that it can price a KNOWN "
            "change -- a new team, a promoted role, a departed teammate's vacated "
            "targets, a rookie with no history at all -- which a top-down 'last "
            "year's points' model structurally cannot represent. No correlation "
            "here captures that, and none of them argue against it.",
            "The pooled same-team/changed-team correlations are INFLATED relative "
            "to every within-position number in this experiment. Pooling puts "
            "quarterbacks and kickers in one sample -- kickers score nothing under "
            "this league's full-PPR rules -- so the spread of true talent is far "
            "wider than inside any one position, and a wider spread raises "
            "correlation on its own. They are meaningful ONLY as same-versus-moved "
            "against each other, never against the per-position figures.",
            "Games-played persistence moves with the minimum-games filter more "
            "than anything else measured here: it is the reported figure at a "
            "six-game floor, higher with no floor at all, and lower at eight. "
            "Tightening the filter removes the healthy-then-injured seasons that "
            "the correlation is about.",
            "A player who misses an ENTIRE season has no rows in `weekly`, so he "
            "never enters a year-over-year pair. Every durability number here is "
            "therefore optimistic -- the worst outcome is invisible to it.",
            "A share correlating well is not the same as a decomposed projection "
            "working well. A real one multiplies three estimates and compounds "
            "three errors; this measures each input on its own, which is precisely "
            "what the standard rationale claims about them.",
        ],
    }


EXPERIMENT = Experiment(
    id="e003",
    title="The parts are not more predictable than the whole",
    question="Should we project fantasy points directly, or decompose into team "
             "volume, player share and games played -- are the parts really more "
             "predictable than the points they combine into?",
    run=run,
    tags=("projections", "persistence", "usage"),
    overturns="Bottom-up projection is justified by the claim that team volume, "
              "player share and games played are each more stable year to year "
              "than fantasy points. On this data they are not: shares are "
              "consistently LESS persistent than the per-game points they are "
              "meant to explain, and team volume is far less persistent than "
              "either.",
)
