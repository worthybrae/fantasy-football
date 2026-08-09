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
            newest = newest.dropna(subset=["slot"]).sort_values("slot")
            return {"order": [{"slot": int(r["slot"]), "manager": r["manager"]}
                              for _, r in newest.iterrows()],
                    "my_slot": None, "source": "espn"}
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

    @app.get("/api/sim/{run_id}")
    def sim_status(run_id: str):
        state = _sim_runs.get(run_id)
        if state is None:
            raise HTTPException(status_code=404, detail="unknown run_id")
        return state

    return app

app = create_app()
