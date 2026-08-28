import json
import math
import threading
import uuid
from contextlib import asynccontextmanager

# FIRST, before anything reads a variable. `api.billing` and
# `pipeline.credentials` both decide what they are at import time from the
# environment, so a `.env` loaded after them is a `.env` that did nothing.
# No-op without the file, which is every deployment -- see api/env.py.
#
# AND NEVER UNDER PYTEST. A suite whose behaviour depends on whether the
# person running it happens to keep a Stripe key in a file is a suite that
# passes on one machine and fails on another; with a real key present, every
# test that touches the lobby would start writing a billing database into the
# repository. The tests that exercise this loader call it directly with their
# own path.
import os
import sys

from api.env import load_env_file

# Two guards, because the first is not enough on its own: PYTEST_CURRENT_TEST
# is set per test, so a test module that imports this one at COLLECTION
# time (tests/test_api.py does) saw no such variable and loaded the owner's
# `.env` -- Stripe key, Supabase DSN -- into every test that followed.
# `pytest` in sys.modules is true from the moment the runner starts.
if "PYTEST_CURRENT_TEST" not in os.environ and "pytest" not in sys.modules:
    load_env_file()

import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query, Request, Response
from api import http_cache
from pipeline.db import get_conn, read_table, write_table, DEFAULT_PATH
from scoring import league
from scoring.board import build_board  # noqa: F401 -- kept importable so
# test_landing_status_does_not_build_the_board can monkeypatch
# "api.main.build_board" to assert /api/landing/status never reaches it.
# Every real board build below goes through cached_build_board instead.
from scoring.board_cache import cached_build_board
from scoring.config import DEFAULT_WEIGHTS
from scoring.draft_model import SUMMARY_FEATURES
from scoring.draft_sim import DEFAULT_ROLLOUTS, run_sim
from scoring.game_points import cached_game_points
from scoring.profile import build_profile

def _int_or_none(value):
    return None if value is None or pd.isna(value) else int(value)


def _float_or_none(value):
    """JSON has no infinity, and `logloss` is genuinely inf when there is
    nothing to score -- FastAPI's encoder rejects it outright."""
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _bool_or_none(value):
    """None, not False, for a missing/NA cell -- same rule as the numeric
    fields beside it. `bool(None)` is False, and the rail renders False as
    the positive claim "the fitted model does not beat the ADP baseline",
    so a `model_backtest` row written before this column existed would
    accuse the model of losing a comparison nobody ran."""
    return None if value is None or pd.isna(value) else bool(value)


def _seasons_or_none(value):
    """`model_backtest.seasons` is a JSON-encoded list of ints (see
    `draft_model.write_backtest`) -- decode it back into a real list for the
    response. None for a missing/NA cell, including a `model_backtest` row
    written before this column existed (a pre-Task-6 backtest table)."""
    if value is None or pd.isna(value):
        return None
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


# The only board columns /api/landing/preview returns. Named here rather than
# sliced inline so the response cannot quietly grow back toward /api/players'
# full 27 columns (185KB) as the board gains more.
PREVIEW_COLUMNS = ("rank", "name", "position", "team", "tier", "vor",
                   "market_rank", "edge")


# Positions the history shape row always reports, zero-filled -- matches the
# web's SHAPE_POSITIONS order (ManagerForecast.tsx) so a manager who has
# never taken a position at all still shows a badge for it, not a gap.
_HISTORY_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")


def _tendency_payload(rows: pd.DataFrame) -> dict | None:
    """Reshape one manager's slice of the long `manager_tendencies` table
    (see scoring.draft_model.manager_tendencies for what each metric means)
    into the nested object the card reads.

    None -- not an object of empty lists -- when the manager has no rows at
    all, which is what a database predating `make fit-managers` writing this
    table looks like. The card then renders exactly what it rendered before
    the table existed, rather than a row of blanks.
    """
    if rows.empty:
        return None

    def sorted_rows(metric: str, by_value_desc: bool):
        sub = rows[rows["metric"] == metric]
        return sub.sort_values("value", ascending=not by_value_desc)

    first_pick = [{"position": r["key"], "drafts": int(r["value"]),
                   "of": int(r["n"])}
                  for _, r in sorted_rows("first_pick", True).iterrows()]
    overall = rows[rows["metric"] == "reach_overall"]
    by_bucket = {r["key"]: r for _, r in rows[rows["metric"] == "reach_bucket"].iterrows()}
    # Fixed early/mid/late order, not whatever order the table came back in:
    # the card prints these as a sequence of rounds and a shuffled one reads
    # as nonsense.
    reach_by_bucket = [
        {"bucket": b, "mean_gap": float(by_bucket[b]["value"]),
         "n": int(by_bucket[b]["n"])}
        for b in ("early", "mid", "late") if b in by_bucket]
    # Strongest reach first (most positive mean_gap) -- the position they
    # jump the board for is the actionable one.
    reach_by_position = [
        {"position": r["key"], "mean_gap": float(r["value"]), "n": int(r["n"])}
        for _, r in sorted_rows("reach_position", True).iterrows()]
    # Earliest first, so "takes a QB in round 4" leads and "gets round to a
    # kicker in round 15" trails.
    first_at_position = [
        {"position": r["key"], "mean_round": float(r["value"]),
         "drafts": int(r["n"])}
        for _, r in sorted_rows("first_at_position", False).iterrows()]

    return {
        "first_pick": first_pick,
        "reach": None if overall.empty else {
            "mean_gap": float(overall.iloc[0]["value"]),
            "n": int(overall.iloc[0]["n"])},
        "reach_by_bucket": reach_by_bucket,
        "reach_by_position": reach_by_position,
        "first_at_position": first_at_position,
    }


def _history_round_bucket(round_no: int) -> str:
    """Same "early" cutoff (round <= 3) scoring.draft_model.EARLY_ROUNDS
    trains against, and the same mid/late split _round_bucket there uses --
    kept as a local constant rather than an import so this endpoint has no
    dependency on the model module while it's under separate active edit."""
    if round_no <= 3:
        return "early"
    return "mid" if round_no <= 8 else "late"


# How many requests may be inside threadpool-backed handlers at once.
THREADPOOL_TOKENS = 200


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Startup and shutdown, in the one place the pinned FastAPI wants them.

    `@app.on_event` is deprecated in this version and warns on every app
    built, which in a test run is hundreds of lines. Same two jobs it did.

    ROOM FOR THE POLLS, before the first request. Every route here is a
    plain `def`, so Starlette runs each request on a thread from anyio's
    default pool -- forty threads, which was plenty for one draft room and
    is not for two hundred polling every 2.5 s alongside their SSE streams
    and a handful of connects blocked in a 30 s build. Set here rather than
    in create_app because the limiter belongs to the running event loop,
    and the number is a ceiling on concurrency, not a pool created up front.
    """
    import anyio
    anyio.to_thread.current_default_thread_limiter().total_tokens = THREADPOOL_TOKENS
    try:
        yield
    finally:
        # The build workers (api/live_build.py) are child processes; a
        # SIGTERM'd parent that does not tell them to stop leaves them to
        # the platform's grace period. No wait: uvicorn is already closing.
        from api import live_build
        live_build.shutdown()


def create_app(db_path: str = DEFAULT_PATH) -> FastAPI:
    app = FastAPI(title="Draft Board API", lifespan=_lifespan)
    # Compression. First, before a single route exists. The other half of
    # this -- the `private, no-store` default -- goes on at the very END of
    # this function, because it has to be the outermost middleware to see
    # the responses that never reach a route. See api/http_cache.py.
    http_cache.install(app)
    conn = get_conn(db_path)
    # In-process only: run status lives here, not in the database, so a
    # status poll survives only as long as this app instance does (the same
    # lifetime as `conn` and every other piece of in-memory server state).
    # A poll for a run_id from a previous process lifetime finds nothing here
    # and 404s -- see sim_status -- rather than crashing.
    _sim_runs: dict[str, dict] = {}

    # THE SEAM BETWEEN THIS MODULE AND THE LIVE DRAFT'S SETTINGS.
    #
    # Declared here as a stub and replaced by `register_live_routes` at the
    # bottom of this function, which owns the live session and its lock. The
    # stub is what the name holds only during construction; every handler
    # below runs after create_app has returned, so none of them can observe
    # it. A missing seam is therefore a wiring bug that shows up as "no live
    # session, ever", which is why the default is written here explicitly
    # rather than left to a getattr fallback at the call site.
    #
    # WHY AN ACCESSOR AND NOT THE STATE ITSELF: `api/live.py`'s `state` dict
    # is guarded by a lock that lives in the same closure, and this module
    # must not learn that discipline. `live_settings()` takes the lock, reads
    # the one slot, and returns a frozen LeagueSettings off a frozen
    # DraftSession (both are replaced wholesale via dataclasses.replace, never
    # mutated in place -- see DraftSession), so what comes back cannot be a
    # torn read and cannot change underneath the request that got it.
    #
    # WHY NOT PERSIST THE LIVE SETTINGS INTO THE `league` TABLE and let
    # `league.load` keep answering for everybody -- the obvious alternative,
    # and it was rejected for three reasons. (1) That table is one row PER
    # SEASON of IMPORTED DRAFT HISTORY, and each row's `pick_order` is that
    # season's real order; `_slot_from_pick_order` in api/live.py exists
    # precisely because the newest stored row's order is last year's and
    # answers confidently and wrongly. Writing a live-fetched row would make
    # `load()` return it forever after, including long after the draft ends.
    # (2) A connect to a non-default league builds against a DIFFERENT
    # database file (`state["league_conn"]`), so a row written there would
    # never reach the connection these endpoints hold anyway. (3) A session
    # is transient by design -- it dies with the process unless the small
    # session record is restored -- and a database row is not, so persisting
    # would outlive the fact it describes.
    app.state.live_settings = lambda request=None: None

    def _league_settings(cur, request=None):
        """The league every price on the board and the profile is computed in.

        The live session's real ESPN settings while a draft is connected,
        the stored `league` row otherwise. Those are two different leagues on
        the owner's own machine: the stored row is the 2025 import and prices
        NO KICKING (6 of the 11 kicking stat ids scoring/league.py can map sit
        in its `unmapped_scoring`, because the row was written before that map
        understood them), while a connect fetches the current season's roster
        and scoring live from ESPN. Before this, the draft room was ranking on
        one and the profile beside it was priced under the other -- a kicker's
        card came back empty next to a room that had his points.

        The board cache keys on `league.to_json(settings)` and the profile
        cache on the rules dict, so the two leagues get two entries and
        neither can ever be served the other's frame. Neither thrashes: a
        session's settings object is fixed for its whole life (frozen, and
        only ever replaced by dataclasses.replace on unrelated fields), so
        every request during one draft produces the identical key.
        """
        # No request means an anonymous, shared answer (the landing
        # endpoints): the stored league, never anybody's live room.
        if request is None:
            return league.load(cur)
        live = app.state.live_settings(request)
        return live if live is not None else league.load(cur)

    @app.get("/api/players")
    def players(request: Request,
                w_production: float = Query(DEFAULT_WEIGHTS["production"], ge=0),
                w_role: float = Query(DEFAULT_WEIGHTS["role"], ge=0),
                w_environment: float = Query(DEFAULT_WEIGHTS["environment"], ge=0),
                w_schedule: float = Query(DEFAULT_WEIGHTS["schedule"], ge=0),
                w_durability: float = Query(DEFAULT_WEIGHTS["durability"], ge=0)):
        cur = conn.cursor()
        try:
            weights = {"production": w_production, "role": w_role,
                       "environment": w_environment, "schedule": w_schedule,
                       "durability": w_durability}
            settings = _league_settings(cur, request)
            try:
                # cached_build_board (scoring/board_cache.py): this endpoint
                # alone cost ~1.6-1.9s per request rebuilding the same board
                # from scratch. The cache key covers weights/settings/drafted/
                # data-freshness, so a slider change or a pick still produces
                # a fresh board -- see that module's docstring.
                board = cached_build_board(cur, weights, settings)
            except ValueError as e:
                # compute_composite raises when weights sum <= 0 -- reachable
                # from the UI if every slider is dragged to 0.
                raise HTTPException(status_code=422, detail=str(e))
            # Last completed season's points, week by week, for the available
            # table's inline bar chart (AvailableList.tsx). Priced under the
            # SAME settings the board was, so the bars and the row's own
            # `stats`/`proj_points` are in one currency -- and built once per
            # refresh rather than per request or per pick, which is the whole
            # argument in scoring/game_points.py. `.map` over a dict leaves
            # NaN for a player with no rows in that season (a rookie, a
            # defense, or a kicker in a league that prices no kicking), and
            # the NaN->None pass below turns that into JSON null: an explicit
            # "no games last season" the table renders as an empty state,
            # never as a row of zero-height bars.
            games = cached_game_points(cur, settings.scoring)
            board = board.assign(
                game_points=board["player_id"].map(games.by_player))
            # astype(object) first, else float columns silently revert None -> NaN
            # and FastAPI's JSON encoder rejects NaN
            board = board.astype(object).where(board.notna(), None)
            return {"players": board.to_dict(orient="records")}
        finally:
            cur.close()

    @app.post("/api/drafted/{player_id}")
    def draft(player_id: str):
        cur = conn.cursor()
        try:
            next_pick = cur.execute(
                "SELECT coalesce(max(pick_no), 0) + 1 FROM drafted").fetchone()[0]
            cur.execute("INSERT OR IGNORE INTO drafted VALUES (?, ?)",
                        [player_id, next_pick])
            # INSERT OR IGNORE silently no-ops for an already-drafted
            # player, leaving its original pick_no in place -- report that
            # stored value, not the freshly-computed next_pick, so a re-POST
            # can never claim a pick_no that disagrees with the actual row.
            stored = cur.execute(
                "SELECT pick_no FROM drafted WHERE player_id = ?", [player_id]
            ).fetchone()[0]
            return {"drafted": True, "pick_no": stored}
        finally:
            cur.close()

    @app.delete("/api/drafted/{player_id}")
    def undraft(player_id: str):
        cur = conn.cursor()
        try:
            cur.execute("DELETE FROM drafted WHERE player_id = ?", [player_id])
            return {"drafted": False}
        finally:
            cur.close()

    def _sources(cur):
        """Per-source freshness rows, shaped for JSON.

        Shared by /api/meta and the landing page's readiness strip so the two
        cannot disagree about what "never refreshed" looks like.
        """
        m = read_table(cur, "meta")
        if m.empty:
            return []
        # astype(object).where(notna, None) first to convert NaN -> None,
        # then stringify non-null values to avoid "NaT" in JSON
        m = m.astype(object).where(m.notna(), None)
        m["refreshed_at"] = m["refreshed_at"].map(
            lambda v: None if v is None else str(v)
        )
        return m.to_dict(orient="records")

    @app.get("/api/meta")
    def meta():
        cur = conn.cursor()
        try:
            return {"sources": _sources(cur)}
        finally:
            cur.close()

    @app.get("/api/landing/status")
    def landing_status(response: Response):
        """Is this machine ready for draft night?

        Raw table reads only -- deliberately never build_board, which costs
        seconds. That is the whole reason this is split from
        /api/landing/preview: the landing page paints readiness on the first
        frame and lets the board preview arrive behind a skeleton. A test
        pins the no-board-build property, since it is the kind of thing a
        later "simplification" would merge away.
        """
        # The same answer for every visitor, and the first request the
        # landing page makes. Fifteen seconds is short enough that a refresh
        # after `make refresh` shows the new counts almost at once, and long
        # enough that a burst of arrivals reads one set of table reads.
        http_cache.public(response, 15)
        cur = conn.cursor()
        try:
            settings = league.load(cur)
            picks = read_table(cur, "draft_picks")
            teams = read_table(cur, "draft_teams")
            profiles = read_table(cur, "manager_profiles")
            sim = read_table(cur, "sim_results")

            # manager_profiles is long -- one row per model coefficient -- so
            # counting rows would report the number of terms as the number of
            # managers.
            per_manager = profiles.drop_duplicates("manager") if not profiles.empty else profiles

            return {
                "sources": _sources(cur),
                "league": {"season": settings.season, "teams": settings.teams,
                           "rounds": settings.rounds,
                           # False means "the built-in default shape", not
                           # "no league" -- the strip says which, rather than
                           # implying a configured league that isn't there.
                           "derived": not read_table(cur, "league").empty},
                "history": {
                    "picks": int(len(picks)),
                    "seasons": (sorted(int(s) for s in picks["season"].unique())
                                if not picks.empty else []),
                    "teams": (int(teams["team_id"].nunique())
                              if not teams.empty else 0),
                },
                "managers": {
                    "fitted": int(len(per_manager)),
                    # The subset whose own fitted model beat the pooled one.
                    "personal": (int(per_manager["uses_personal"].sum())
                                 if not per_manager.empty else 0),
                },
                "sim": (None if sim.empty or "created_at" not in sim.columns else {
                    "run_id": str(sim.iloc[0]["run_id"]),
                    "my_slot": _int_or_none(sim.iloc[0].get("my_slot")),
                    "created_at": str(sim.iloc[0]["created_at"]),
                }),
            }
        finally:
            cur.close()

    @app.get("/api/landing/preview")
    def landing_preview(response: Response, limit: int = 12):
        """The top of the board, for the landing page's preview panel.

        `limit` is clamped, not validated: this feeds a panel whose job is to
        show the tool works, and rendering an error there in answer to
        ?limit=0 would defeat the point. Fifty is the ceiling because nothing
        on that page scrolls past it.
        """
        # A minute, because the board underneath this costs seconds to build
        # and the top twelve of it do not move between builds. Anonymous and
        # identical for everybody, so one build can answer the whole window.
        http_cache.public(response, 60)
        cur = conn.cursor()
        try:
            # Same board GET /api/players just built (same weights, same
            # cache key) -- see cached_build_board's docstring. `settings`
            # has to come from the same place /api/players gets it, or a
            # connected draft would put two boards in the cache and pay the
            # 1.6-1.9s build twice to show the same twelve rows.
            # Anonymous by design: this answer is shared by everybody
            # (http_cache.public above), so it must not vary with the
            # caller's own live room. The stored league prices it.
            board = cached_build_board(cur, DEFAULT_WEIGHTS,
                                       _league_settings(cur, None))
            n = max(1, min(int(limit), 50))
            top = board.sort_values("rank").head(n)[list(PREVIEW_COLUMNS)]
            top = top.astype(object).where(top.notna(), None)
            return {"pool": int(len(board)),
                    "players": top.to_dict(orient="records")}
        finally:
            cur.close()

    @app.get("/api/players/{player_id}/profile")
    def player_profile(player_id: str, request: Request,
                        w_production: float = Query(DEFAULT_WEIGHTS["production"], ge=0),
                        w_role: float = Query(DEFAULT_WEIGHTS["role"], ge=0),
                        w_environment: float = Query(DEFAULT_WEIGHTS["environment"], ge=0),
                        w_schedule: float = Query(DEFAULT_WEIGHTS["schedule"], ge=0),
                        w_durability: float = Query(DEFAULT_WEIGHTS["durability"], ge=0)):
        cur = conn.cursor()
        try:
            weights = {"production": w_production, "role": w_role,
                       "environment": w_environment, "schedule": w_schedule,
                       "durability": w_durability}
            try:
                # `settings` passed rather than left to build_profile's own
                # `league.load` fallback: with a draft connected, this is the
                # league ESPN says the owner is IN, and every figure on the
                # card -- points per game, the game log, the volatility rank,
                # the stat twins and their forecast -- is priced under it.
                # The visible failure this fixes: the owner's stored league
                # row prices no kicking, so a kicker's card came back with an
                # empty history while the room next to it, built on the live
                # settings, was ranking him on real points.
                payload = build_profile(cur, player_id, weights,
                                        _league_settings(cur, request))
            except ValueError as e:
                # Same as /api/players -- compute_composite raises when
                # weights sum <= 0, reachable if every slider is at 0.
                raise HTTPException(status_code=422, detail=str(e))
            if payload is None:
                raise HTTPException(status_code=404, detail="unknown player_id")
            return payload
        finally:
            cur.close()

    @app.get("/api/league")
    def league_info():
        cur = conn.cursor()
        try:
            derived = not read_table(cur, "league").empty
            s = league.load(cur)
            return {"season": s.season, "teams": s.teams,
                    "starters": s.starters, "flex_slots": s.flex_slots,
                    "bench": s.bench, "rounds": s.rounds, "derived": derived,
                    "unmapped_scoring": list(s.unmapped_scoring)}
        finally:
            cur.close()

    @app.get("/api/managers")
    def managers():
        cur = conn.cursor()
        try:
            profiles = read_table(cur, "manager_profiles")
            if profiles.empty:
                return {"managers": []}
            out = []
            for manager, grp in profiles.groupby("manager"):
                head = grp.iloc[0]
                gain = head["heldout_gain"]
                out.append({
                    "manager": manager,
                    "summary": head["summary"],
                    "n_picks": int(head["n_picks"]),
                    "uses_personal": bool(head["uses_personal"]),
                    "heldout_gain": None if pd.isna(gain) else float(gain),
                    # `shown` is `draft_model.SUMMARY_FEATURES`, sent per
                    # coefficient so the rail's card filters on a flag from
                    # the model rather than on its own copy of the feature
                    # list -- a copy that went stale as soon as the model
                    # grew a feature. Every coefficient still ships; the
                    # client decides only what to draw.
                    "coefficients": [
                        {"feature": r["feature"], "value": float(r["value"]),
                         "pooled_value": float(r["pooled_value"]),
                         "shown": r["feature"] in SUMMARY_FEATURES}
                        for _, r in grp.iterrows()],
                })
            return {"managers": out}
        finally:
            cur.close()

    @app.get("/api/managers/history")
    def managers_history():
        """Every manager's real draft picks, shaped for the forecast cards.

        Right now every manager falls back to the league-average model (see
        /api/model -- n_managers with uses_personal is 0), so this raw
        history is genuinely more informative about a specific manager than
        the fitted coefficients are. One read of each table plus a pandas
        groupby -- not a query per manager.

        `tendencies` carries the measured statistics `make fit-managers`
        precomputes into `manager_tendencies`: what they open with, how far
        ahead of the board they take players and in which rounds, and when
        they get to their first QB/TE/K/DST. Null for any manager the table
        doesn't cover, including every manager if it was never written.
        """
        cur = conn.cursor()
        try:
            picks = read_table(cur, "draft_picks")
            teams = read_table(cur, "draft_teams")
            tendencies = read_table(cur, "manager_tendencies")
            if picks.empty or teams.empty:
                return {"managers": []}
            merged = picks.merge(
                teams[["season", "team_id", "manager"]],
                on=["season", "team_id"], how="inner")

            out = []
            for manager, grp in merged.groupby("manager"):
                n_seasons = int(
                    teams.loc[teams["manager"] == manager, "season"].nunique())

                # One entry per season: the round-1 pick. A manager who held
                # two team_ids in the same season (a mid-draft trade) could
                # in principle produce two round-1 rows for that season --
                # keep the earliest overall_pick so the list stays one entry
                # per season, most recent season first.
                firsts = (grp[grp["round"] == 1]
                          .sort_values("overall_pick")
                          .drop_duplicates("season", keep="first")
                          .sort_values("season", ascending=False))
                # player_name/position/nfl_team are null when a pick's
                # espn_player_id was missing from that season's ESPN player
                # directory (import_seasons' left join) -- passed through as
                # JSON null rather than papered over, so the frontend decides
                # how to render an unidentified pick instead of this endpoint
                # guessing a label for it.
                first_rounders = [
                    {"season": int(r["season"]),
                     "player_name": None if pd.isna(r["player_name"]) else r["player_name"],
                     "position": None if pd.isna(r["position"]) else r["position"],
                     "nfl_team": None if pd.isna(r["nfl_team"]) else r["nfl_team"],
                     "keeper": bool(r["keeper"])}
                    for _, r in firsts.iterrows()]

                # Positional shape by round bucket, across every season on
                # record. Picks with no matched player have no position to
                # bucket and are excluded here (though they still count
                # toward total_picks below) -- there is nothing dishonest
                # about that: they are absent from the shape, not silently
                # folded into some position they weren't.
                positioned = grp.dropna(subset=["position"])
                bucket_counts = (
                    positioned.assign(bucket=positioned["round"].map(_history_round_bucket))
                    .groupby(["bucket", "position"]).size())
                shape = {b: {p: int(bucket_counts.get((b, p), 0))
                             for p in _HISTORY_POSITIONS}
                         for b in ("early", "mid", "late")}

                mine = (tendencies[tendencies["manager"] == manager]
                        if not tendencies.empty and "manager" in tendencies.columns
                        else pd.DataFrame())

                out.append({
                    "manager": manager,
                    "seasons": n_seasons,
                    "total_picks": int(len(grp)),
                    "first_rounders": first_rounders,
                    "shape": shape,
                    "tendencies": _tendency_payload(mine),
                })
            return {"managers": out}
        finally:
            cur.close()

    @app.get("/api/draft-order")
    def draft_order():
        cur = conn.cursor()
        try:
            saved = read_table(cur, "draft_order")
            if not saved.empty:
                saved = saved.sort_values("slot")
                me = saved[saved["is_me"]]
                return {"order": [{"slot": int(r["slot"]), "manager": r["manager"]}
                                  for _, r in saved.iterrows()],
                        "my_slot": int(me.iloc[0]["slot"]) if not me.empty else None,
                        "source": "manual"}
            teams = read_table(cur, "draft_teams")
            if teams.empty:
                return {"order": [], "my_slot": None, "source": "none"}
            newest = teams[teams["season"] == teams["season"].max()]
            ordered = newest.dropna(subset=["slot"]).sort_values("slot")
            if not ordered.empty:
                return {"order": [{"slot": int(r["slot"]), "manager": r["manager"]}
                                  for _, r in ordered.iterrows()],
                        "my_slot": None, "source": "espn"}
            # ESPN leaves `draftDayPickOrder` null until it publishes an
            # order, which is the normal state for the season you are
            # preparing for. Returning an empty list here left the rail with
            # no rows to edit, so there was no way to enter an order at all --
            # and entering one is the whole pre-draft workflow. Seed the
            # managers into slots so they can be rearranged, and say the
            # order is a placeholder rather than ESPN's.
            seeded = newest.sort_values("team_id")
            return {"order": [{"slot": i, "manager": r["manager"]}
                              for i, (_, r) in enumerate(seeded.iterrows(), start=1)],
                    "my_slot": None, "source": "unpublished"}
        finally:
            cur.close()

    @app.put("/api/draft-order")
    def set_draft_order(payload: dict = Body(...)):
        entries = payload.get("order") or []
        if not entries:
            raise HTTPException(status_code=422, detail="order must not be empty")
        slots = [int(e["slot"]) for e in entries]
        if len(slots) != len(set(slots)):
            raise HTTPException(status_code=422, detail="order contains duplicate slots")
        my_slot = payload.get("my_slot")
        my_slot = int(my_slot) if my_slot is not None else None
        if my_slot is not None and my_slot not in slots:
            raise HTTPException(status_code=422,
                                 detail="my_slot must be one of the submitted slots")
        rows = pd.DataFrame([{"slot": int(e["slot"]), "manager": e["manager"],
                              "is_me": int(e["slot"]) == my_slot}
                             for e in entries])
        cur = conn.cursor()
        try:
            write_table(cur, "draft_order", rows)
            return {"saved": len(rows)}
        finally:
            cur.close()

    @app.post("/api/sim")
    def start_sim(payload: dict = Body(...)):
        my_slot = int(payload.get("my_slot") or 0)
        if my_slot < 1:
            raise HTTPException(status_code=422, detail="my_slot is required")
        rollouts = int(payload.get("rollouts") or DEFAULT_ROLLOUTS)
        order = draft_order()
        slot_managers = {e["slot"]: e["manager"] for e in order["order"]}
        run_id = uuid.uuid4().hex[:12]
        _sim_runs[run_id] = {"status": "running", "detail": None}

        def worker():
            # A dedicated cursor, not the shared `conn`, so this background
            # thread's long-running writes (sim_results, sim_survival, and
            # fit_all's manager_profiles) don't share a live statement/result
            # state with whatever cursor a concurrent request handler is
            # using at the same moment -- see api/main.py's other handlers,
            # every one of which already opens conn.cursor() per call for
            # the same reason (see test_concurrent_requests).
            cur = conn.cursor()
            try:
                run_sim(cur, my_slot, slot_managers, n_rollouts=rollouts)
                _sim_runs[run_id] = {"status": "done", "detail": None}
            except Exception as e:
                _sim_runs[run_id] = {"status": "error", "detail": str(e)}
            finally:
                cur.close()

        threading.Thread(target=worker, daemon=True).start()
        return {"run_id": run_id, "status": "running"}

    # Declared BEFORE /api/sim/{run_id}: FastAPI matches routes in
    # registration order, so the path parameter would otherwise swallow
    # "latest" and 404 it as an unknown run_id.
    @app.get("/api/sim/latest")
    def sim_latest():
        """Provenance for the sim currently merged into the board.

        sim_results/sim_survival are replaced wholesale by each run and
        merged into every board unconditionally, so a reloaded page is
        showing some run -- this says which slot, which pick, and when.
        """
        cur = conn.cursor()
        try:
            res = read_table(cur, "sim_results")
            if res.empty or "created_at" not in res.columns:
                return {"run": None}
            head = res.iloc[0]
            return {"run": {"run_id": str(head["run_id"]),
                            "my_slot": _int_or_none(head.get("my_slot")),
                            "pick_no": _int_or_none(head.get("pick_no")),
                            "created_at": str(head["created_at"])}}
        finally:
            cur.close()

    @app.get("/api/sim/board")
    def sim_board():
        cur = conn.cursor()
        try:
            settings = league.load(cur)
            order = draft_order()["order"]
            cells = read_table(cur, "sim_board")
            # Delegated rather than re-derived: sim_results was created
            # without my_slot/pick_no/created_at (they arrived eleven commits
            # later), so a run that predates that migration has no such
            # columns and reading them raises. sim_latest already guards for
            # exactly that, and inheriting its guard means the two endpoints
            # can never disagree about what "the last run" is.
            run = sim_latest()["run"]
            if cells.empty:
                return {"run": run, "teams": settings.teams,
                        "rounds": settings.rounds, "order": order, "cells": []}
            # cached_build_board, not build_board directly -- same board as
            # /api/players when weights/settings/drafted state agree, no
            # separate 1.6-1.9s rebuild for this grid.
            board = cached_build_board(cur, settings=settings)
            # drop_duplicates because board ids are not unique: see
            # build_board's own comment -- _add_adp_only_players synthesizes
            # `player_id = "adp_" + norm` with no position in the key, so one
            # normalized name at two positions in the ADP feed produces two
            # rows sharing an id. Without this, `names.loc[pid]` is a
            # DataFrame, `row["name"]` is a Series, and the encoder recurses
            # until it dies -- one bad row killing all 120 cells.
            # astype(object) for the same reason /api/players does it: a
            # missing `team` reaches here as np.nan (json.dumps rejects it,
            # HTTP 500) or pd.NA (encoded as `{}`, which React refuses to
            # render as a child), and SimBoardCell.team is `string | null`.
            # `market_spread` rides along so the grid can mark a placement
            # the sources fight about. A cell shows one player at one pick and
            # reads as a flat claim; when ESPN has a player 21st and FFC 13th
            # and CBS 9th, the claim is a good deal softer than it looks, and
            # the number that says so is already computed here.
            names = board[["player_id", "name", "position", "team",
                           "market_spread"]] \
                .drop_duplicates("player_id").set_index("player_id")
            names = names.astype(object).where(names.notna(), None)
            out = []
            for _, c in cells.iterrows():
                pid = c["player_id"]
                # A cell can name a player the board no longer carries (a
                # refresh between runs). Render the id rather than dropping
                # the cell, so the grid never silently loses a pick.
                if pid in names.index:
                    row = names.loc[pid]
                    name, position, team = row["name"], row["position"], row["team"]
                    spread = row["market_spread"]
                else:
                    name, position, team, spread = pid, None, None, None
                out.append({"overall_pick": int(c["overall_pick"]),
                            "round": int(c["round"]),
                            "round_pick": int(c["round_pick"]),
                            "slot": int(c["slot"]),
                            "alt_rank": int(c["alt_rank"]),
                            "player_id": pid, "name": name,
                            "position": position, "team": team,
                            "prob": float(c["prob"]),
                            "market_spread": None if spread is None else float(spread),
                            "certain": bool(c["certain"])})
            return {"run": run, "teams": settings.teams,
                    "rounds": settings.rounds, "order": order, "cells": out}
        finally:
            cur.close()

    @app.get("/api/model")
    def model_status():
        """Whether opponent models exist at all, and whether they beat ADP.

        Both are guards the spec asked for and neither reached the board:
        with no fitted models every opponent picks uniformly at random, and
        the backtest was printed to stdout by `make fit-managers` and then
        dropped.
        """
        cur = conn.cursor()
        try:
            profiles = read_table(cur, "manager_profiles")
            has_managers = not profiles.empty and "manager" in profiles.columns
            n_managers = int(profiles["manager"].nunique()) if has_managers else 0
            bt = read_table(cur, "model_backtest")
            report = None
            if not bt.empty:
                row = bt.iloc[0]
                report = {"seasons": _seasons_or_none(row.get("seasons")),
                          "top1": _float_or_none(row.get("top1")),
                          "top5": _float_or_none(row.get("top5")),
                          "logloss": _float_or_none(row.get("logloss")),
                          "adp_top1": _float_or_none(row.get("adp_top1")),
                          "adp_logloss": _float_or_none(row.get("adp_logloss")),
                          "beats_adp": _bool_or_none(row.get("beats_adp"))}
            return {"fitted": n_managers > 0, "n_managers": n_managers,
                    "backtest": report}
        finally:
            cur.close()

    @app.get("/api/sim/{run_id}")
    def sim_status(run_id: str):
        state = _sim_runs.get(run_id)
        if state is None:
            raise HTTPException(status_code=404, detail="unknown run_id")
        return state

    from api.live import register_live_routes
    register_live_routes(app, conn, db_path)

    # Imported here rather than at module scope for the same reason the live
    # routes are: this is the seam where the app is assembled, and neither
    # module is importable as a plain dependency of this one -- `api.mocks`
    # imports `api.live`'s cell builder, so a top-level import here would fix
    # the order the three modules must load in.
    #
    # `conn` and nothing else: the mock pages read the DRAFT CORPUS, which is
    # a different file with a different lifetime, and they open it read-only
    # per request rather than holding a handle for the process lifetime --
    # the farm has to be able to take that file's write lock while the API is
    # up. `conn` is passed only so a mock board can name its players out of
    # the board this process already has.
    from api.mocks import register_mock_routes
    register_mock_routes(app, conn)

    # Imported here for the same reason as the two routers above: this is
    # the seam where the app is assembled. Unlike those two, `api.lobby`
    # reads nothing from `conn` or the corpus at all -- it proxies ESPN's
    # own public mock-lobby directory, cached in its own module -- so it
    # takes no arguments beyond `app`.
    from api.lobby import register_lobby_routes
    register_lobby_routes(app)

    # Credential custody: the two disconnect controls and the status probe
    # the connect screen reads. Imported here for the same assembly-seam
    # reason as the routers above, and given no arguments at all -- it shares
    # nothing with `conn`, because stored ESPN sessions live in their own
    # database file (see pipeline/credentials.DEFAULT_DB_PATH for why they
    # must not ride along in the one that gets copied to seed new leagues).
    #
    # Registering the routes does NOT open that file: the store is lazy (see
    # `default_store`), so a deployment nobody has connected an account to
    # never creates it, and the test suite never takes its lock.
    from api.custody import register_custody_routes
    register_custody_routes(app)

    # Billing. Mounted unconditionally: with no Stripe key in the environment
    # it answers "nothing is for sale here" and gates nothing, which is every
    # local checkout and the whole test suite. See api/billing.py.
    from api.billing import register_billing_routes
    register_billing_routes(app)

    # Per-account preferences -- today, the favourites the draft plan targets.
    # After billing because it reads through the same account id, and given
    # `conn` because the one thing it validates is that a favourite names a
    # player this board knows. See api/account.py.
    from api.account import register_account_routes
    register_account_routes(app, conn)

    # Upcoming drafts and server-side token minting. Registered after custody
    # because it reads through it: the session these routes act as is the one
    # `custody_for` resolves, and only in its total absence this machine's own
    # saved ESPN login (see api/drafts.py for why that order is fixed).
    from api.drafts import register_draft_routes
    register_draft_routes(app)

    # The landing page's hero: a real mock draft, live, off the farm's own
    # record. Read-only, anonymous, and given this app's connection so the
    # names and prices it shows are the board every other endpoint serves.
    from api.demo import register_demo_routes
    register_demo_routes(app, conn)

    # The draft archive: what hundreds of recorded drafts do from a given
    # seat. Signed in only, and read out of the corpus rather than this
    # database -- the board's connection is passed for names alone.
    from api.market import register_market_routes
    register_market_routes(app, conn)

    # The crawlable site: ADP pages rendered from the corpus as plain HTML,
    # and the sitemap. Before the SPA, whose fallback would otherwise answer
    # every one of these paths with index.html -- see api/seo.py.
    from api.seo import register_seo_routes
    register_seo_routes(app, conn)

    # League reports: public reads of a stored report, owner-only builds.
    # Before the SPA for the same reason the SEO pages are.
    from api.reports import register_report_routes
    register_report_routes(app)

    # A league's history and manager profiles, imported on first visit
    # with the visitor's own ESPN session (api/league_history.py). Before
    # the SPA catch-all, like everything under /api.
    from api.league_history import register_league_history_routes
    register_league_history_routes(app)

    # THE BUILT FRONTEND, LAST. Its fallback route matches every path there
    # is, so anything registered after it would be unreachable -- see
    # api/static.py, which also explains why the SPA is served from this
    # process at all rather than from its own domain (the custody cookie is
    # SameSite=Lax and would not survive the split).
    #
    # A checkout with no `web/dist` -- every test run, and the dev server,
    # where Vite serves the frontend itself -- mounts nothing and is not an
    # error.
    from api.static import register_spa
    register_spa(app)

    # Background work this instance does to itself: refreshing its own data,
    # and farming mock drafts into the corpus. Both off unless switched on,
    # so a local `make up` and every test import start nothing -- see
    # api/jobs.py, which also explains why these are threads in this process
    # rather than a second service (DuckDB's per-process file lock, and a
    # volume that attaches to exactly one service).
    #
    # LAST, and after the routes: `start_jobs` never blocks, but a first
    # refresh runs for minutes, and the healthcheck has to be answerable
    # throughout it.
    from api.jobs import start_jobs
    start_jobs(conn)

    # The threadpool ceiling and the build-worker shutdown live in
    # `_lifespan` above, which this app was constructed with.

    # `private, no-store` on every /api answer that named no policy of its
    # own. LAST LINE, so it is the outermost middleware and every response
    # passes through it -- including the 400s CredentialTransportGuard
    # produces without routing. See api/http_cache.py.
    http_cache.install_private_default(app)

    return app

app = create_app()
