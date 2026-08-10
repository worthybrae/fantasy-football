import math
import threading
import uuid

import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query
from pipeline.db import get_conn, read_table, write_table, DEFAULT_PATH
from scoring import league
from scoring.board import build_board
from scoring.config import DEFAULT_WEIGHTS
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

    @app.get("/api/meta")
    def meta():
        cur = conn.cursor()
        try:
            m = read_table(cur, "meta")
            # astype(object).where(notna, None) first to convert NaN -> None,
            # then stringify non-null values to avoid "NaT" in JSON
            m = m.astype(object).where(m.notna(), None)
            m["refreshed_at"] = m["refreshed_at"].map(
                lambda v: None if v is None else str(v)
            )
            return {"sources": m.to_dict(orient="records")}
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
                    "coefficients": [
                        {"feature": r["feature"], "value": float(r["value"]),
                         "pooled_value": float(r["pooled_value"])}
                        for _, r in grp.iterrows()],
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
            names = board[["player_id", "name", "position", "team"]] \
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
                else:
                    name, position, team = pid, None, None
                out.append({"overall_pick": int(c["overall_pick"]),
                            "round": int(c["round"]),
                            "round_pick": int(c["round_pick"]),
                            "slot": int(c["slot"]),
                            "alt_rank": int(c["alt_rank"]),
                            "player_id": pid, "name": name,
                            "position": position, "team": team,
                            "prob": float(c["prob"]),
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
                report = {"holdout_season": _int_or_none(row.get("holdout_season")),
                          "top1": _float_or_none(row.get("top1")),
                          "top5": _float_or_none(row.get("top5")),
                          "logloss": _float_or_none(row.get("logloss")),
                          "adp_top1": _float_or_none(row.get("adp_top1")),
                          "adp_logloss": _float_or_none(row.get("adp_logloss")),
                          "beats_adp": bool(row.get("beats_adp"))}
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

    return app

app = create_app()
