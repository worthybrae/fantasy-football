"""Assemble the player-profile payload: history, outlook, similar seasons."""
import math
import numpy as np
import pandas as pd
from pipeline.db import read_table
from scoring import factors
from scoring.board import build_board, _norm_name, _adapt_depth_charts
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

    mine = mine.sort_values("season", ascending=False)
    rows = []
    for _, r in mine.iterrows():
        rows.append({
            "season": int(r["season"]),
            "games": int(r["games"]),
            "ppg": _round_or_none(r["ppg"], 1),
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


def game_log(weekly: pd.DataFrame, player_id: str) -> list[dict]:
    if weekly.empty:
        return []
    wk = weekly[weekly["player_id"] == player_id].copy()
    if wk.empty:
        return []
    wk["ppr_points"] = compute_ppr_points(wk)
    wk = wk.sort_values(["season", "week"], ascending=[False, False])
    rows = []
    for _, r in wk.iterrows():
        rows.append({
            "season": int(r["season"]),
            "week": int(r["week"]),
            "opponent": r.get("opponent_team"),
            "stat_line": _stat_line(r, r.get("position")),
            "stats": _game_stats(r),
            "ppr_points": round(float(r["ppr_points"]), 1),
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
            adp = r["adp"]
            p["adp"] = None if pd.isna(adp) else float(adp)
        else:
            p["rank"] = None
            p["adp"] = None
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
        twins = find_twins(weekly, player_id) if not weekly.empty else None
        similar = (_enrich_twins(twins, board) if twins is not None
                   else value_neighbors(board, player_id))

    payload = {
        "header": header,
        "factors": factors_out,
        "seasons": seasons,
        "game_log": logs,
        "outlook": outlook_out,
        "similar": similar,
    }
    return _scrub(payload)
