import pandas as pd
from pipeline.db import get_conn, write_table, read_table, record_freshness
from pipeline.sources import parse_adp

def test_write_and_read_roundtrip(tmp_path):
    conn = get_conn(str(tmp_path / "t.duckdb"))
    write_table(conn, "weekly", pd.DataFrame({"a": [1, 2]}))
    assert read_table(conn, "weekly")["a"].tolist() == [1, 2]

def test_read_missing_table_returns_empty(tmp_path):
    conn = get_conn(str(tmp_path / "t.duckdb"))
    assert read_table(conn, "nope").empty

def test_freshness_upsert(tmp_path):
    conn = get_conn(str(tmp_path / "t.duckdb"))
    record_freshness(conn, "weekly", True, 100)
    record_freshness(conn, "weekly", True, 200)
    meta = read_table(conn, "meta")
    assert len(meta) == 1 and meta.iloc[0]["rows"] == 200

def test_parse_adp():
    payload = {"players": [
        {"player_id": 1, "name": "Justin Jefferson", "position": "WR", "team": "MIN", "adp": 3.2},
        {"player_id": 2, "name": "49ers Defense", "position": "DEF", "team": "SF", "adp": 140.1},
    ]}
    df = parse_adp(payload)
    assert list(df.columns) == ["adp_name", "position", "team", "adp"]
    assert df.iloc[1]["position"] == "DST"  # FFC "DEF" mapped to nflverse "DST"
