import pandas as pd
from pipeline.db import get_conn, write_table, read_table, record_freshness
from pipeline.sources import parse_adp, _normalize_weekly
from pipeline.sources import parse_espn, parse_fp_ecr, parse_sleeper

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

def test_normalize_weekly_renames_team_to_recent_team():
    df = pd.DataFrame({"player_id": [1, 2], "team": ["MIN", "SF"]})
    out = _normalize_weekly(df)
    assert "recent_team" in out.columns
    assert "team" not in out.columns
    assert out["recent_team"].tolist() == ["MIN", "SF"]

def test_normalize_weekly_leaves_existing_recent_team_untouched():
    df = pd.DataFrame({"player_id": [1, 2], "recent_team": ["MIN", "SF"], "team": ["XXX", "YYY"]})
    out = _normalize_weekly(df)
    # recent_team already present: no rename should occur, both columns survive as-is
    assert out["recent_team"].tolist() == ["MIN", "SF"]
    assert out["team"].tolist() == ["XXX", "YYY"]

def test_normalize_weekly_no_season_type_column_no_filter():
    df = pd.DataFrame({"player_id": [1, 2], "team": ["MIN", "SF"]})
    out = _normalize_weekly(df)
    assert len(out) == 2

def test_normalize_weekly_filters_to_regular_season():
    df = pd.DataFrame({
        "player_id": [1, 2, 3],
        "team": ["MIN", "SF", "KC"],
        "season_type": ["REG", "POST", "REG"],
    })
    out = _normalize_weekly(df)
    assert out["player_id"].tolist() == [1, 3]
    assert (out["season_type"] == "REG").all()

def test_parse_espn():
    payload = {"players": [
        {"player": {"id": 4429795, "fullName": "Jahmyr Gibbs", "defaultPositionId": 2,
                    "ownership": {"averageDraftPosition": 1.77},
                    "draftRanksByRankType": {"PPR": {"rank": 1}}}},
        {"player": {"id": 1, "fullName": "Some Lineman", "defaultPositionId": 9}},
    ]}
    df = parse_espn(payload)
    assert len(df) == 1
    r = df.iloc[0]
    assert r["espn_id"] == 4429795 and r["position"] == "RB"
    assert r["espn_adp"] == 1.77 and r["espn_ppr_rank"] == 1

def test_parse_fp_ecr():
    html = ('<script>var x = 1; var ecrData = {"players": [{"player_name": "JaMarr Chase",'
            '"player_team_id": "CIN", "player_position_id": "WR", "rank_ecr": 1,'
            '"rank_ave": "1.77", "rank_std": "1.2", "tier": 1}]};</script>')
    df = parse_fp_ecr(html)
    assert df.iloc[0]["rank_ecr"] == 1 and df.iloc[0]["position"] == "WR"
    assert parse_fp_ecr("<html>no data</html>").empty

def test_parse_sleeper():
    payload = {
        "a": {"gsis_id": " 00-0038543", "espn_id": 4429795, "full_name": "Jahmyr Gibbs",
              "position": "RB", "team": "DET", "active": True},
        "b": {"gsis_id": None, "espn_id": 99, "full_name": "No Gsis", "position": "WR"},
        "c": {"gsis_id": "00-1", "espn_id": 5, "full_name": "A Lineman", "position": "OT"},
    }
    df = parse_sleeper(payload)
    assert len(df) == 1 and df.iloc[0]["gsis_id"] == "00-0038543"
