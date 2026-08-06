"""Assemble the player-profile payload: history, outlook, similar seasons."""
import math
import numpy as np
import pandas as pd
from pipeline.db import read_table
from scoring import factors
from scoring.board import build_board, _norm_name, _adapt_depth_charts
from scoring.config import RECENCY_WEIGHTS
from scoring.ppr import compute_ppr_points
from scoring.similarity import (player_season_features, find_twins,
                                value_neighbors)

_KDST_POSITIONS = {"K", "DST"}


def _scrub(v):
    if isinstance(v, dict):
        return {k: _scrub(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_scrub(x) for x in v]
    if v is None or (pd.api.types.is_scalar(v) and pd.isna(v)):
        # Catches pd.NA/NaT (e.g. an empty adp table leaves the whole `adp`
        # column as pd.NA) in addition to float NaN, which the isinstance
        # checks below would otherwise miss.
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if math.isnan(f) else f
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _num(row, col):
    v = row.get(col, 0)
    return 0 if v is None or (isinstance(v, float) and math.isnan(v)) else v


def _round_or_none(v, ndigits):
    if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v):
        return None
    return round(float(v), ndigits)


def _stat_line(row, position):
    if position == "QB":
        comp = int(_num(row, "completions"))
        att = int(_num(row, "attempts"))
        pyds = int(_num(row, "passing_yards"))
        ptd = int(_num(row, "passing_tds"))
        ints = int(_num(row, "passing_interceptions"))
        line = f"{comp}/{att}, {pyds} yds, {ptd} TD, {ints} INT"
        car = _num(row, "carries")
        if car > 0:
            line += f" · {int(car)} car, {int(_num(row, 'rushing_yards'))} yds"
        return line
    if position == "RB":
        car = int(_num(row, "carries"))
        ryds = int(_num(row, "rushing_yards"))
        rtd = int(_num(row, "rushing_tds"))
        line = f"{car} car, {ryds} yds, {rtd} TD"
        tgt = _num(row, "targets")
        if tgt > 0:
            line += (f" · {int(_num(row, 'receptions'))} rec, "
                      f"{int(_num(row, 'receiving_yards'))} yds")
        return line
    # WR/TE (and fallback for K/other positions)
    tgt = int(_num(row, "targets"))
    rec = int(_num(row, "receptions"))
    yds = int(_num(row, "receiving_yards"))
    rtd = int(_num(row, "receiving_tds"))
    line = f"{tgt} tgt, {rec} rec, {yds} yds, {rtd} TD"
    car = _num(row, "carries")
    if car > 0:
        line += f" · {int(car)} car, {int(_num(row, 'rushing_yards'))} yds"
    return line


# Payload key -> weekly-table column. Every game_log row carries all 12 keys
# (zero-filled when the column is absent) so the frontend's position-aware
# column configs can index into a uniform shape.
_GAME_STAT_COLS = {
    "completions": "completions", "attempts": "attempts",
    "pass_yards": "passing_yards", "pass_tds": "passing_tds",
    "interceptions": "passing_interceptions",
    "carries": "carries", "rush_yards": "rushing_yards",
    "rush_tds": "rushing_tds", "targets": "targets",
    "receptions": "receptions", "rec_yards": "receiving_yards",
    "rec_tds": "receiving_tds",
}


def _game_stats(row):
    return {k: int(_num(row, col)) for k, col in _GAME_STAT_COLS.items()}


def season_summaries(weekly: pd.DataFrame, snaps: pd.DataFrame, player_id: str) -> list[dict]:
    if weekly.empty:
        return []
    feats = player_season_features(weekly)
    # Positional finish by total season points (the standard "finished RB12"
    # framing), ranked across every player in the weekly table.
    feats["pos_finish"] = (feats.groupby(["season", "position"])["points"]
                           .rank(ascending=False, method="min"))
    mine = feats[feats["player_id"] == player_id].copy()
    if mine.empty:
        return []

    # team per player-season, needed for the snap-share join (not present on
    # player_season_features output)
    team_by_season = (weekly[weekly["player_id"] == player_id]
                       .groupby("season")["recent_team"].last())
    mine["team"] = mine["season"].map(team_by_season)
    mine["_norm_name"] = mine["name"].map(_norm_name)

    if snaps is not None and not snaps.empty:
        s = snaps.copy()
        s["_norm_name"] = s["player"].map(_norm_name)
        share = (s.groupby(["_norm_name", "team", "season"], as_index=False)
                  ["offense_pct"].mean())
        mine = mine.merge(share, on=["_norm_name", "team", "season"], how="left")
        mine = mine.rename(columns={"offense_pct": "snap_share"})
    else:
        mine["snap_share"] = np.nan

    # Passing aggregates aren't part of player_season_features (that frame
    # feeds twin matching in similarity.py and must not change) -- aggregate
    # them here from the raw weekly rows instead.
    pass_cols = {k: _GAME_STAT_COLS[k] for k in
                 ("completions", "attempts", "pass_yards", "pass_tds", "interceptions")}
    wk_mine = weekly[weekly["player_id"] == player_id].copy()
    for out, col in pass_cols.items():
        wk_mine[out] = (pd.to_numeric(wk_mine[col], errors="coerce").fillna(0)
                        if col in wk_mine.columns else 0.0)
    passing = wk_mine.groupby("season", as_index=False)[list(pass_cols)].sum()
    mine = mine.merge(passing, on="season", how="left")
    for out in pass_cols:
        mine[out] = mine[out].fillna(0)

    # Week-to-week volatility for the consistency chart. Weekly rows are
    # games played by definition (dnp zero-fill exists only in game_log), so
    # no exclusion is needed; sample std is NaN -> None for 1-game seasons.
    wk_mine["_ppr"] = compute_ppr_points(wk_mine)
    mine["ppg_std"] = mine["season"].map(wk_mine.groupby("season")["_ppr"].std())

    mine = mine.sort_values("season", ascending=False)
    rows = []
    for _, r in mine.iterrows():
        rows.append({
            "season": int(r["season"]),
            "games": int(r["games"]),
            "ppg": _round_or_none(r["ppg"], 1),
            "ppg_std": _round_or_none(r["ppg_std"], 2),
            "pos_finish": int(r["pos_finish"]),
            "completions": int(r["completions"]),
            "attempts": int(r["attempts"]),
            "pass_yards": int(r["pass_yards"]),
            "pass_tds": int(r["pass_tds"]),
            "interceptions": int(r["interceptions"]),
            "targets": int(r["targets"]),
            "target_share": _round_or_none(r["target_share"], 3),
            "carries": int(r["carries"]),
            "rec_yards": int(r["rec_yards"]),
            "rush_yards": int(r["rush_yards"]),
            "tds": int(r["tds"]),
            "receptions": int(r["receptions"]),
            "yards_per_opp": _round_or_none(r["yards_per_opp"], 1),
            "snap_share": _round_or_none(r["snap_share"], 3),
        })
    return rows


_SUMMARY_STATS = ["completions", "attempts", "pass_yards", "pass_tds",
                  "interceptions", "carries", "rush_yards", "targets",
                  "receptions", "rec_yards", "tds"]


def career_summary(seasons: list[dict]) -> dict:
    """Recency-weighted per-game averages over the scoring window.

    Same weights as the production factor (RECENCY_WEIGHTS), renormalized
    over the seasons the player actually has; older seasons carry weight 0
    and drop out entirely.
    """
    weighted = [(RECENCY_WEIGHTS.get(s["season"], 0.0), s) for s in seasons]
    weighted = [(w, s) for w, s in weighted if w > 0 and s["games"] > 0]
    total = sum(w for w, _ in weighted)
    if total == 0:
        return {"w_ppg": None, "w_stats": {}}
    w_ppg = sum(w * s["ppg"] for w, s in weighted) / total
    stats = {k: round(sum(w * s[k] / s["games"] for w, s in weighted) / total, 1)
             for k in _SUMMARY_STATS}
    return {"w_ppg": round(w_ppg, 1), "w_stats": stats}


def _espn_projection(conn, player_id: str, name: str, position: str) -> float | None:
    """Projected season points from the ESPN table, or None.

    gsis<->espn crosswalk first (same as scoring/market.py), then a
    name+position fallback. Missing espn_proj column (a table written
    before the projection pull existed) degrades to None.
    """
    espn = read_table(conn, "espn_adp")
    if espn.empty or "espn_proj" not in espn.columns:
        return None
    e = espn.dropna(subset=["espn_proj"])
    e = e[e["espn_proj"] > 0]
    if e.empty:
        return None
    sleeper = read_table(conn, "sleeper_ids")
    if not sleeper.empty:
        xwalk = sleeper[["gsis_id", "espn_id"]].dropna().drop_duplicates("espn_id")
        hit = e.merge(xwalk, on="espn_id")
        hit = hit[hit["gsis_id"] == player_id]
        if not hit.empty:
            return float(hit.iloc[0]["espn_proj"])
    hit = e[(e["position"] == position)
            & (e["espn_name"].map(_norm_name) == _norm_name(name))]
    if not hit.empty:
        return float(hit.iloc[0]["espn_proj"])
    return None


def game_log(weekly: pd.DataFrame, player_id: str) -> list[dict]:
    if weekly.empty:
        return []
    wk = weekly[weekly["player_id"] == player_id].copy()
    if wk.empty:
        return []
    wk["ppr_points"] = compute_ppr_points(wk)
    # League-wide last week per season tells how long the season ran (17 vs
    # 18-week eras included); weeks the player has no row for -- injury,
    # bye, healthy scratch -- become zeroed `dnp` rows so the log shows the
    # games missed, not just the games played. Consumers computing per-game
    # averages must exclude dnp rows.
    season_len = weekly.groupby("season")["week"].max()
    by_week = {(int(r["season"]), int(r["week"])): r for _, r in wk.iterrows()}
    position = wk["position"].iloc[-1] if "position" in wk.columns else None
    zero = pd.Series(0.0, index=wk.columns)
    rows = []
    for season in sorted(wk["season"].unique(), reverse=True):
        for week in range(int(season_len[season]), 0, -1):
            r = by_week.get((int(season), week))
            dnp = r is None
            if dnp:
                r = zero
            rows.append({
                "season": int(season),
                "week": week,
                "opponent": None if dnp else r.get("opponent_team"),
                "stat_line": "Did not play" if dnp else _stat_line(r, r.get("position")),
                "stats": _game_stats(r),
                "ppr_points": 0.0 if dnp else round(float(r["ppr_points"]), 1),
                "dnp": dnp,
            })
    return rows


_DEPTH_POSITIONS = ["QB", "RB", "WR", "TE"]


def team_depth_chart(depth: pd.DataFrame, team: str, player_id: str) -> list[dict]:
    """Latest-snapshot offensive depth chart for the player's team.

    Works on the RAW depth_charts schema (dt/team/pos_abb/pos_rank); older
    or unknown schemas degrade to an empty list and the UI hides the card.
    """
    need = {"gsis_id", "team", "pos_abb", "pos_rank", "player_name"}
    if depth.empty or not need.issubset(depth.columns):
        return []
    d = depth[depth["team"] == team].copy()
    if d.empty:
        return []
    if "dt" in d.columns:
        d = d[d["dt"] == d["dt"].max()]
    d["pos_rank"] = pd.to_numeric(d["pos_rank"], errors="coerce")
    out = []
    for pos in _DEPTH_POSITIONS:
        rows = (d[d["pos_abb"] == pos].dropna(subset=["pos_rank"])
                .sort_values("pos_rank").drop_duplicates("gsis_id").head(4))
        players = [{"name": r["player_name"], "rank": int(r["pos_rank"]),
                    "is_me": r["gsis_id"] == player_id}
                   for _, r in rows.iterrows()]
        if players:
            out.append({"position": pos, "players": players})
    return out


def weekly_difficulty(schedules: pd.DataFrame, prior_weekly: pd.DataFrame,
                      team: str, position: str) -> list[dict]:
    """Week-by-week matchup difficulty for the player's team and position.

    fpa_pg = opponent's prior-season PPR points allowed per game to this
    position; pct = its percentile among all teams (high = allows a lot =
    soft matchup). Weeks without a game (bye) carry a null opponent. K/DST
    have no meaningful positional FPA -- empty list, card hidden.
    """
    if schedules.empty or prior_weekly.empty or position not in _DEPTH_POSITIONS:
        return []
    wk = prior_weekly.copy()
    wk["ppr_points"] = compute_ppr_points(wk)
    def_games = wk.groupby("opponent_team")["week"].nunique()
    allowed = (wk[wk["position"] == position]
               .groupby("opponent_team")["ppr_points"].sum() / def_games).dropna()
    if allowed.empty:
        return []
    pct = allowed.rank(pct=True) * 100
    games = {}
    mine = schedules[(schedules["home_team"] == team) | (schedules["away_team"] == team)]
    for _, g in mine.iterrows():
        week = int(g["week"])
        if week > 18:
            continue
        home = g["home_team"] == team
        games[week] = (g["away_team"] if home else g["home_team"], home)
    rows = []
    for week in range(1, 19):
        opp, home = games.get(week, (None, None))
        rows.append({
            "week": week, "opponent": opp, "home": home,
            "fpa_pg": round(float(allowed[opp]), 1) if opp in allowed.index else None,
            "pct": round(float(pct[opp])) if opp in pct.index else None,
        })
    return rows


def _outlook(weekly: pd.DataFrame, depth: pd.DataFrame, schedules: pd.DataFrame,
             player_row: dict) -> dict:
    team = player_row.get("team")
    position = player_row.get("position")
    player_id = player_row.get("player_id")

    depth_slot = None
    adapted = _adapt_depth_charts(depth)
    if not adapted.empty and "gsis_id" in adapted.columns:
        mine = adapted[adapted["gsis_id"] == player_id]
        if not mine.empty:
            ranks = pd.to_numeric(mine["depth_team"], errors="coerce").dropna()
            if not ranks.empty:
                depth_slot = int(ranks.min())

    implied_points = None
    sos_raw = None
    sos_pct = None
    bye = None

    if not schedules.empty:
        env = factors.environment_factor(schedules)
        erow = env[env["team"] == team]
        if not erow.empty:
            implied_points = _round_or_none(erow.iloc[0]["env_raw"], 1)

        if not weekly.empty:
            prior = weekly[weekly["season"] == weekly["season"].max()]
            sos = factors.schedule_factor(prior, schedules)
            pos_sos = sos[sos["position"] == position]
            srow = pos_sos[pos_sos["team"] == team]
            if not srow.empty:
                sos_raw = _round_or_none(srow.iloc[0]["sos_raw"], 1)
                pct = pos_sos["sos_raw"].rank(pct=True) * 100
                sos_pct = _round_or_none(pct.loc[srow.index[0]], 1)

        byes = factors.bye_weeks(schedules)
        brow = byes[byes["team"] == team]
        if not brow.empty:
            b = brow.iloc[0]["bye"]
            bye = None if pd.isna(b) else int(b)

    return {"depth_slot": depth_slot, "implied_points": implied_points,
            "sos_raw": sos_raw, "sos_pct": sos_pct, "bye": bye}


def _enrich_twins(twins: dict, board: pd.DataFrame) -> dict:
    board_idx = board.set_index("player_id")
    for p in twins["players"]:
        if p["player_id"] in board_idx.index:
            r = board_idx.loc[p["player_id"]]
            p["rank"] = int(r["rank"])
            market_rank = r["market_rank"]
            p["market_rank"] = None if pd.isna(market_rank) else float(market_rank)
        else:
            p["rank"] = None
            p["market_rank"] = None
    return twins


def build_profile(conn, player_id: str, weights: dict | None = None) -> dict | None:
    board = build_board(conn, weights)
    match = board[board["player_id"] == player_id]
    if match.empty:
        return None
    row = match.iloc[0]
    header = row.to_dict()
    factors_out = {k: header[k] for k in
                   ("production", "durability", "role", "environment", "schedule")}

    weekly = read_table(conn, "weekly")
    snaps = read_table(conn, "snap_counts")
    schedules = read_table(conn, "schedules")
    depth = read_table(conn, "depth_charts")

    if header["position"] == "K":
        # Kickers have weekly rows, but the PPR formula doesn't score kicking
        # stats, so every one of those rows nets 0 points -- a "history" of
        # zeros is misleading, not informative. Collapse it the same way a
        # rookie's genuinely-empty history collapses (DST never has weekly
        # rows at all, so it already returns empty here).
        seasons = []
        logs = []
    else:
        seasons = season_summaries(weekly, snaps, player_id)
        logs = game_log(weekly, player_id)
    outlook_out = _outlook(weekly, depth, schedules, header)

    if header["position"] in _KDST_POSITIONS:
        similar = value_neighbors(board, player_id)
    else:
        players = read_table(conn, "players")
        twins = (find_twins(weekly, player_id, players=players)
                 if not weekly.empty else None)
        similar = (_enrich_twins(twins, board) if twins is not None
                   else value_neighbors(board, player_id))

    summary = career_summary(seasons)
    proj_total = _espn_projection(conn, player_id, header["name"], header["position"])
    summary["proj_ppg"] = round(proj_total / 17, 1) if proj_total else None
    summary["proj_delta"] = (round(summary["proj_ppg"] - summary["w_ppg"], 1)
                             if summary["proj_ppg"] is not None and summary["w_ppg"] is not None
                             else None)

    prior = weekly[weekly["season"] == weekly["season"].max()] if not weekly.empty else weekly
    payload = {
        "header": header,
        "factors": factors_out,
        "summary": summary,
        "seasons": seasons,
        "game_log": logs,
        "outlook": outlook_out,
        "depth_chart": team_depth_chart(depth, header["team"], player_id),
        "schedule": weekly_difficulty(schedules, prior, header["team"], header["position"]),
        "similar": similar,
    }
    return _scrub(payload)
