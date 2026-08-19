"""Assemble the player-profile payload: history, outlook, similar seasons."""
import math
import numpy as np
import pandas as pd
from pipeline.db import read_table
from scoring import factors, league
from scoring.board import _norm_name, _adapt_depth_charts
from scoring.board_cache import cached_build_board
from scoring.config import RECENCY_WEIGHTS
from scoring.profile_cache import cached_profile_frames, snap_share_by_season
from scoring.ppr import compute_ppr_points, normalize_rules
from scoring.similarity import (player_season_features, find_twins,
                                value_neighbors)

_KDST_POSITIONS = {"K", "DST"}

# Distinguishable from an explicit `snap_share=None`, which means "there is
# no snap data" -- a real, different answer from "work it out yourself".
_UNSET = object()


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


def season_summaries(weekly: pd.DataFrame, snaps: pd.DataFrame | None, player_id: str,
                     *, season_features: pd.DataFrame | None = None,
                     snap_share=_UNSET, rules: dict | None = None) -> list[dict]:
    """Per-season rows for one player.

    Only ONE thing here is league-wide: `pos_finish`, which ranks this
    player against every other player-season in `season_features`.
    Everything else reads `weekly[weekly.player_id == player_id]` and
    `snaps` for this player's name/team/season. So a caller that already
    holds the two league-wide aggregates -- the features frame and the
    snap-share aggregate -- may pass them in and hand `weekly` nothing but
    this player's rows, and `snaps` nothing at all. That is what
    scoring/profile.py's `build_profile` does now: recomputing them per
    click cost 0.211s and 0.343s respectively on data/nfl.duckdb (see
    scoring/profile_cache.py). Pass neither and the behaviour is what it
    always was, which is what every test here does.

    `rules` is the league's `settings.scoring`; None is full PPR. It prices
    `ppg_std` here directly, and -- on the path that computes them -- `ppg`,
    `points` and `pos_finish` through `player_season_features`. A caller
    supplying `season_features` MUST have priced that frame under the same
    rules, or a row would carry a half-PPR volatility beside a full-PPR
    average: `build_profile` gets both from `cached_profile_frames(conn,
    rules)`, which keys on them (scoring/profile_cache.py).
    """
    if weekly.empty:
        return []
    rules = normalize_rules(rules)
    feats = (player_season_features(weekly, rules) if season_features is None
             else season_features)
    # Positional finish by total season points (the standard "finished RB12"
    # framing), ranked across every player in the weekly table.
    # `.assign` rather than `feats["pos_finish"] = ...`: `feats` may now be
    # a frame owned by the caller (profile_cache's cached aggregate), and
    # writing a column into it would leave `pos_finish` stuck on the cached
    # object for every later click. profile_cache hands out copies too --
    # this is the belt to that braces, and it costs 1.2ms.
    feats = feats.assign(pos_finish=feats.groupby(["season", "position"])["points"]
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

    share = snap_share_by_season(snaps) if snap_share is _UNSET else snap_share
    if share is not None:
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
    wk_mine["_ppr"] = compute_ppr_points(wk_mine, rules)
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


def game_log(weekly: pd.DataFrame, player_id: str, *,
             season_len: pd.Series | None = None,
             rules: dict | None = None) -> list[dict]:
    """Week-by-week rows, each week's points scored under `rules`.

    The payload key stays `ppr_points` -- it is what the client reads and
    what the consistency chart is keyed on -- but the number in it is this
    league's points, not full PPR, whenever `rules` says so. None is full
    PPR, so every existing caller is unchanged.
    """
    if weekly.empty:
        return []
    wk = weekly[weekly["player_id"] == player_id].copy()
    if wk.empty:
        return []
    wk["ppr_points"] = compute_ppr_points(wk, normalize_rules(rules))
    # League-wide last week per season tells how long the season ran (17 vs
    # 18-week eras included); weeks the player has no row for -- injury,
    # bye, healthy scratch -- become zeroed `dnp` rows so the log shows the
    # games missed, not just the games played. Consumers computing per-game
    # averages must exclude dnp rows.
    # This is the one thing here that is NOT per-player, so a caller passing
    # only this player's rows in `weekly` MUST supply it -- derived from
    # their own rows it would be the last week they played, and every week
    # they missed at the end of a season would vanish from the log instead
    # of showing up as a dnp.
    if season_len is None:
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
                      team: str, position: str,
                      rules: dict | None = None) -> list[dict]:
    """Week-by-week matchup difficulty for the player's team and position.

    fpa_pg = opponent's prior-season points allowed per game to this
    position, scored under `rules` (None = full PPR). Points allowed is as
    format-dependent as points scored -- a defense that concedes catches
    gives up a third less in a standard league -- and this number sits on the
    same page as the board's `schedule` factor, which `build_board` already
    computes under the league's rules via `factors.schedule_factor`. Leaving
    it full PPR put two differently-priced versions of one quantity in front
    of the same reader.

    pct = its percentile among all teams (high = allows a lot = soft
    matchup). Weeks without a game (bye) carry a null opponent. K/DST
    have no meaningful positional FPA -- empty list, card hidden.
    """
    if schedules.empty or prior_weekly.empty or position not in _DEPTH_POSITIONS:
        return []
    wk = prior_weekly.copy()
    wk["ppr_points"] = compute_ppr_points(wk, normalize_rules(rules))
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
             player_row: dict, rules: dict | None = None) -> dict:
    """Depth slot, implied points, strength of schedule and bye week.

    `weekly` is only ever read as `weekly[weekly.season == max(season)]`
    below, so handing this the latest-season slice instead of the whole
    table produces the identical frame (the re-slice becomes a no-op that
    keeps every row, index and order intact) -- which is what
    `build_profile` does, off profile_cache's `prior_weekly`.

    `depth` likewise is only read as `adapted[adapted.gsis_id == player_id]`,
    so a slice already narrowed to this player is enough.
    """
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
            # The same `rules` build_board hands schedule_factor, so the
            # profile's `sos_raw` is the raw number behind the board's
            # `schedule` percentile rather than a second, PPR-priced version
            # of it. `environment_factor` and `bye_weeks` above and below read
            # only betting lines and the schedule grid, so no scoring rule
            # touches them.
            sos = factors.schedule_factor(prior, schedules, normalize_rules(rules))
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


def _table_columns(conn, name: str) -> set[str]:
    """Column names of `name`, or an empty set if the table doesn't exist.

    Same existence check `read_table` (pipeline/db.py) makes, one query
    later: the filtered reads below have to know a column is there before
    they can put it in a WHERE clause, and a table that predates a column
    must fall back to reading the lot rather than raising.
    """
    return {r[0] for r in conn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = ?",
        [name]).fetchall()}


def _player_weekly(conn, player_id: str) -> pd.DataFrame:
    """This player's weekly rows only.

    `read_table(conn, "weekly")` pulled all 174,373 rows x 145 columns
    (0.247s) so that `season_summaries` and `game_log` could immediately
    throw away everything but the ~100 rows belonging to one player. The
    same filter in SQL costs 36ms.

    VERIFIED, not assumed, because both callers are order-sensitive
    (`groupby("season")["recent_team"].last()` and `wk["position"].iloc[-1]`
    both depend on row order, and a differently-ordered scan would quietly
    change a team or a position): for all 249 board players this returns a
    frame `assert_frame_equal`-identical -- values, order AND dtypes -- to
    `read_table(conn, "weekly")[weekly.player_id == player_id]`. The
    mechanism is DuckDB's `preserve_insertion_order`, which defaults to
    true and makes a filtered scan yield storage order just as a full scan
    does. If that setting is ever turned off for this connection, this is
    the line to revisit.
    """
    cols = _table_columns(conn, "weekly")
    if not cols:
        return pd.DataFrame()
    if "player_id" not in cols:
        return read_table(conn, "weekly")
    return conn.execute("SELECT * FROM weekly WHERE player_id = ?", [player_id]).df()


def _depth_slice(conn, team, player_id: str) -> pd.DataFrame:
    """The depth-chart rows for this player's team, plus this player's own.

    depth_charts is the biggest table the profile touched -- 416,885 rows
    over 142 daily snapshots, 0.233s to read, 275 MB in memory -- and the
    two consumers between them want one team's rows (`team_depth_chart`)
    and one player's rows (`_outlook`). The union of the two filters is a
    single query, 18ms, and both consumers then narrow it exactly as they
    did before: `team_depth_chart` takes `depth.team == team` and the
    latest `dt` WITHIN that team, which the OR-clause cannot disturb
    because it never removes a row of that team.

    Verified identical (`assert_frame_equal`, dtypes included) to
    `read_table(conn, "depth_charts")[(team match) | (gsis_id match)]` for
    all 249 board players. A schema without both columns -- the fixture in
    tests/test_profile.py writes depth_charts with no `team` column at all
    -- falls back to the full read, which is what those rows are sized for.
    """
    cols = _table_columns(conn, "depth_charts")
    if not cols:
        return pd.DataFrame()
    if not {"team", "gsis_id"}.issubset(cols):
        return read_table(conn, "depth_charts")
    # A NaN/None team must match nothing, which is what `= NULL` does and
    # what pandas' `depth["team"] == nan` did.
    team = team if isinstance(team, str) else None
    return conn.execute(
        "SELECT * FROM depth_charts WHERE team = ? OR gsis_id = ?",
        [team, player_id]).df()


def build_profile(conn, player_id: str, weights: dict | None = None,
                  settings: "league.LeagueSettings | None" = None) -> dict | None:
    # `settings` (the league's roster shape AND its scoring rules) is loaded
    # here the same way `build_board` loads it, and for the same reason: every
    # derived number below is priced in the league's points, not in full PPR.
    # Before this it was never loaded at all, so a half-PPR league's profile
    # showed PPR ppg, PPR volatility, a PPR game log, PPR stat twins matched
    # on PPR features with a PPR "next year" forecast, and a PPR points-
    # allowed schedule -- every one of them beside a header row the board had
    # already priced correctly.
    #
    # Passed to `cached_build_board` too, not left to default: without it the
    # two would load `league.load(conn)` separately, which is the same answer
    # today but would silently diverge the first time a caller (api/live.py's
    # connect flow already does this for the board) hands in settings fetched
    # live from ESPN instead of the database's.
    settings = settings or league.load(conn)
    rules = settings.scoring
    # Was `build_board(conn, weights)` -- every profile click rebuilt the
    # whole 249-row board (all factors, composite, VOR, tiers, a five-source
    # market consensus) just to read one row back out, measured at ~3.1s
    # end to end. cached_build_board (scoring/board_cache.py) reuses the same
    # board `GET /api/players` just built, or built for a prior profile
    # click with the same weights/settings/drafted state, and only
    # recomputes when one of those actually changed -- see that module's
    # docstring for the exact key and the staleness failure it guards
    # against (a profile showing a just-picked player as still available).
    board = cached_build_board(conn, weights, settings)
    match = board[board["player_id"] == player_id]
    if match.empty:
        return None
    row = match.iloc[0]
    header = row.to_dict()
    factors_out = {k: header[k] for k in
                   ("production", "durability", "role", "environment", "schedule")}

    # Was four full `read_table` calls -- weekly (174,373 rows), snap_counts
    # (253,106), depth_charts (416,885) and schedules -- 0.614s of reading
    # 844k rows per click, before a single one of them was filtered down to
    # one player. cached_profile_frames (scoring/profile_cache.py) holds the
    # league-wide aggregates that survive the filtering (the season-features
    # frame, the snap-share aggregate, the latest season's weekly rows,
    # season lengths, schedules, players) and re-derives them only when
    # `meta` says a pipeline refresh has happened -- the same signal
    # scoring/board_cache.py keys on, imported from it rather than
    # re-invented. The two genuinely per-player reads have their filters
    # pushed into SQL instead.
    frames = cached_profile_frames(conn, rules)
    schedules = frames.schedules
    wk_mine = _player_weekly(conn, player_id)
    depth = _depth_slice(conn, header["team"], player_id)

    if header["position"] == "K":
        # Kickers have weekly rows, but the PPR formula doesn't score kicking
        # stats, so every one of those rows nets 0 points -- a "history" of
        # zeros is misleading, not informative. Collapse it the same way a
        # rookie's genuinely-empty history collapses (DST never has weekly
        # rows at all, so it already returns empty here).
        seasons = []
        logs = []
    else:
        seasons = season_summaries(wk_mine, None, player_id,
                                   season_features=frames.season_features,
                                   snap_share=frames.snap_share, rules=rules)
        logs = game_log(wk_mine, player_id, season_len=frames.season_len,
                        rules=rules)
    outlook_out = _outlook(frames.prior_weekly, depth, schedules, header, rules)

    if header["position"] in _KDST_POSITIONS:
        similar = value_neighbors(board, player_id)
    else:
        players = frames.players
        # `wk_mine` is ignored: find_twins' only use of its `weekly`
        # argument is `player_season_features`, which is what
        # `season_features` supplies. The guard stays on the FULL table
        # being empty, which is what it always tested -- `frames`
        # carries that flag for exactly this line.
        # No `rules` argument: `season_features` is supplied, and
        # find_twins ignores `rules` entirely on that path (it is only used
        # to price the frame it is not being asked to build). The frame IS
        # priced under this league's rules -- cached_profile_frames was given
        # them above and keys its cache on them -- so both the distance the
        # twins are matched on and the `next_ppg` they are reported with are
        # this league's points.
        twins = (find_twins(wk_mine, player_id, players=players,
                            season_features=frames.season_features)
                 if not frames.weekly_empty else None)
        similar = (_enrich_twins(twins, board) if twins is not None
                   else value_neighbors(board, player_id))

    summary = career_summary(seasons)
    proj_total = _espn_projection(conn, player_id, header["name"], header["position"])
    # Re-priced by the board's own `proj_scale` (scoring/board.projection_scale)
    # rather than shown raw. `proj_total` is ESPN's PPR-shaped season number;
    # `w_ppg` two lines down is this league's points; `proj_delta` subtracts
    # one from the other. Differencing two currencies produced a number that
    # was not wrong so much as meaningless -- in a standard league it read a
    # high-reception WR as a huge positive "projected improvement" that was
    # nothing but the missing reception points. The board row carries the
    # factor so this page and the draft room cannot disagree about it.
    proj_scale = header.get("proj_scale")
    if proj_total and proj_scale is not None and not pd.isna(proj_scale):
        proj_total = proj_total * float(proj_scale)
    summary["proj_ppg"] = round(proj_total / 17, 1) if proj_total else None
    summary["proj_delta"] = (round(summary["proj_ppg"] - summary["w_ppg"], 1)
                             if summary["proj_ppg"] is not None and summary["w_ppg"] is not None
                             else None)

    # Identical frame to the old `weekly[weekly.season == weekly.season.max()]`
    # -- profile_cache slices it off the full table with that exact
    # expression, so rows, index and order all match; an empty `weekly`
    # still yields the empty frame this used to fall back to.
    prior = frames.prior_weekly
    payload = {
        "header": header,
        "factors": factors_out,
        "summary": summary,
        "seasons": seasons,
        "game_log": logs,
        "outlook": outlook_out,
        "depth_chart": team_depth_chart(depth, header["team"], player_id),
        "schedule": weekly_difficulty(schedules, prior, header["team"],
                                      header["position"], rules),
        "similar": similar,
    }
    return _scrub(payload)
