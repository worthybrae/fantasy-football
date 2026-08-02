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
        cur = conn.cursor()
        try:
            weights = {"production": w_production, "role": w_role,
                       "environment": w_environment, "schedule": w_schedule,
                       "durability": w_durability}
            board = build_board(cur, weights)
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
            cur.execute("INSERT OR IGNORE INTO drafted VALUES (?)", [player_id])
            return {"drafted": True}
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

    return app

app = create_app()
