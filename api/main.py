import pandas as pd
from fastapi import FastAPI
from pipeline.db import get_conn, read_table, DEFAULT_PATH
from scoring.board import build_board
from scoring.config import DEFAULT_WEIGHTS

def create_app(db_path: str = DEFAULT_PATH) -> FastAPI:
    app = FastAPI(title="Draft Board API")
    conn = get_conn(db_path)

    @app.get("/api/players")
    def players(w_production: float = DEFAULT_WEIGHTS["production"],
                w_role: float = DEFAULT_WEIGHTS["role"],
                w_environment: float = DEFAULT_WEIGHTS["environment"],
                w_schedule: float = DEFAULT_WEIGHTS["schedule"],
                w_durability: float = DEFAULT_WEIGHTS["durability"]):
        weights = {"production": w_production, "role": w_role,
                   "environment": w_environment, "schedule": w_schedule,
                   "durability": w_durability}
        board = build_board(conn, weights)
        # astype(object) first, else float columns silently revert None -> NaN
        # and FastAPI's JSON encoder rejects NaN
        board = board.astype(object).where(board.notna(), None)
        return {"players": board.to_dict(orient="records")}

    @app.post("/api/drafted/{player_id}")
    def draft(player_id: str):
        conn.execute("INSERT OR IGNORE INTO drafted VALUES (?)", [player_id])
        return {"drafted": True}

    @app.delete("/api/drafted/{player_id}")
    def undraft(player_id: str):
        conn.execute("DELETE FROM drafted WHERE player_id = ?", [player_id])
        return {"drafted": False}

    @app.get("/api/meta")
    def meta():
        m = read_table(conn, "meta")
        m["refreshed_at"] = m["refreshed_at"].astype(str)
        m = m.astype(object).where(m.notna(), None)
        return {"sources": m.to_dict(orient="records")}

    return app

app = create_app()
