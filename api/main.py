import json
import math
import threading
import uuid

import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query
from pipeline.db import get_conn, read_table, write_table, DEFAULT_PATH
from scoring import league
from scoring.board import build_board
from scoring.config import DEFAULT_WEIGHTS
from scoring.draft_model import SUMMARY_FEATURES
from scoring.draft_sim import DEFAULT_ROLLOUTS, run_sim
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


def create_app(db_path: str = DEFAULT_PATH) -> FastAPI:
    app = FastAPI(title="Draft Board API")
    conn = get_conn(db_path)
    # In-process only: run status lives here, not in the database, so a
    # status poll survives only as long as this app instance does (the same
    # lifetime as `conn` and every other piece of in-memory server state).
    # A poll for a run_id from a previous process lifetime finds nothing here
    # and 404s -- see sim_status -- rather than crashing.
    _sim_runs: dict[str, dict] = {}

    @app.get("/api/players")
    def players(w_production: float = Query(DEFAULT_WEIGHTS["production"], ge=0),
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
                board = build_board(cur, weights)
            except ValueError as e:
                # compute_composite raises when weights sum <= 0 -- reachable
                # from the UI if every slider is dragged to 0.
                raise HTTPException(status_code=422, detail=str(e))
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
    def landing_status():
        """Is this machine ready for draft night?

        Raw table reads only -- deliberately never build_board, which costs
        seconds. That is the whole reason this is split from
        /api/landing/preview: the landing page paints readiness on the first
        frame and lets the board preview arrive behind a skeleton. A test
        pins the no-board-build property, since it is the kind of thing a
        later "simplification" would merge away.
        """
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
    def landing_preview(limit: int = 12):
        """The top of the board, for the landing page's preview panel.

        `limit` is clamped, not validated: this feeds a panel whose job is to
        show the tool works, and rendering an error there in answer to
        ?limit=0 would defeat the point. Fifty is the ceiling because nothing
        on that page scrolls past it.
        """
        cur = conn.cursor()
        try:
            board = build_board(cur, DEFAULT_WEIGHTS)
            n = max(1, min(int(limit), 50))
            top = board.sort_values("rank").head(n)[list(PREVIEW_COLUMNS)]
            top = top.astype(object).where(top.notna(), None)
            return {"pool": int(len(board)),
                    "players": top.to_dict(orient="records")}
        finally:
            cur.close()

    @app.get("/api/players/{player_id}/profile")
    def player_profile(player_id: str,
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
                payload = build_profile(cur, player_id, weights)
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
            board = build_board(cur, settings=settings)
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

    return app

app = create_app()
