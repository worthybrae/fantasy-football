import pandas as pd
import pytest
from pipeline import sources
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

class _StubResponse:
    """Minimal stand-in for requests.Response -- just enough surface for
    fetch_espn_adp/fetch_fp_ecr (.raise_for_status(), .json(), .text)."""
    def __init__(self, *, json_data=None, text=""):
        self._json = json_data
        self.text = text
    def raise_for_status(self):
        pass
    def json(self):
        return self._json

def test_fetch_espn_adp_raises_on_empty_parse(monkeypatch):
    """A schema-drift response that parses to zero rows must fail loud (so
    refresh records it as FAIL/red-dot) instead of silently writing an
    empty table that reports OK."""
    monkeypatch.setattr(sources.requests, "get",
                         lambda *a, **k: _StubResponse(json_data={"players": []}))
    with pytest.raises(ValueError):
        sources.fetch_espn_adp(2026)

def test_fetch_fp_ecr_raises_on_empty_parse(monkeypatch):
    monkeypatch.setattr(sources.requests, "get",
                         lambda *a, **k: _StubResponse(text="<html>no data</html>"))
    with pytest.raises(ValueError):
        sources.fetch_fp_ecr()

def test_parse_espn_extracts_season_projection():
    # The kona payload carries many stats entries; the season projection is
    # statSourceId=1 (projection), statSplitTypeId=0 (season), seasonId=year.
    payload = {"players": [
        {"player": {"id": 4429795, "fullName": "Jahmyr Gibbs", "defaultPositionId": 2,
                    "ownership": {"averageDraftPosition": 1.77},
                    "draftRanksByRankType": {"PPR": {"rank": 1}},
                    "stats": [
                        {"statSourceId": 0, "statSplitTypeId": 0, "seasonId": 2025, "appliedTotal": 366.9},
                        {"statSourceId": 1, "statSplitTypeId": 1, "seasonId": 2026, "appliedTotal": 22.0},
                        {"statSourceId": 1, "statSplitTypeId": 0, "seasonId": 2026, "appliedTotal": 365.5},
                    ]}},
        {"player": {"id": 2, "fullName": "No Stats Guy", "defaultPositionId": 3}},
    ]}
    df = parse_espn(payload, year=2026)
    assert df.iloc[0]["espn_proj"] == 365.5
    assert pd.isna(df.iloc[1]["espn_proj"])

def test_parse_mfl():
    from pipeline.sources import parse_mfl
    adp = {"adp": {"player": [
        {"id": "100", "averagePick": "2.85", "rank": "1"},
        {"id": "200", "averagePick": "3.90", "rank": "2"},
        {"id": "300", "averagePick": "5.00", "rank": "3"},   # a kicker (PK)
        {"id": "400", "averagePick": "9.00", "rank": "4"},   # team defense -> dropped
        {"id": "999", "averagePick": "9.50", "rank": "5"},   # not in directory -> dropped
    ]}}
    players = {"players": {"player": [
        {"id": "100", "name": "Gibbs, Jahmyr", "position": "RB", "team": "DET"},
        {"id": "200", "name": "Chase, Ja'Marr", "position": "WR", "team": "CIN"},
        {"id": "300", "name": "Aubrey, Brandon", "position": "PK", "team": "DAL"},
        {"id": "400", "name": "Ravens, Baltimore", "position": "Def", "team": "BAL"},
    ]}}
    df = parse_mfl(adp, players)
    assert df["mfl_name"].tolist() == ["Jahmyr Gibbs", "Ja'Marr Chase", "Brandon Aubrey"]
    assert df["position"].tolist() == ["RB", "WR", "K"]
    assert df["mfl_rank"].tolist() == [1, 2, 3]

def test_parse_cbs():
    from pipeline.sources import parse_cbs
    html = (
        '<div class="player-row first"><div class="rank">1</div>'
        '<a href="/nfl/players/3162723/jahmyr-gibbs/fantasy/"><span class="player-name">J. Gibbs</span></a>'
        '<span class="team position">RB $34</span></div>'
        '<div class="player-row"><div class="rank">2</div>'
        '<a href="/nfl/players/28drt/jamarr-chase/fantasy/"><span class="player-name">J. Chase</span></a>'
        '<span class="team position">WR $38</span></div>'
    )
    df = parse_cbs(html)
    assert df["cbs_name"].tolist() == ["jahmyr gibbs", "jamarr chase"]
    assert df["position"].tolist() == ["RB", "WR"]
    assert df["cbs_rank"].tolist() == [1, 2]
    assert parse_cbs("<html>nothing</html>").empty

def test_parse_cbs_row_without_player_href_does_not_bleed():
    from pipeline.sources import parse_cbs
    # A DST-style row (team link, no /nfl/players/ href) sits between two
    # players; its rank must not be attributed to the next player's name.
    html = (
        '<div class="player-row"><div class="rank">1</div>'
        '<a href="/nfl/players/1/jahmyr-gibbs/fantasy/"><span class="player-name">J. Gibbs</span></a>'
        '<span class="team position">RB $34</span></div>'
        '<div class="player-row"><div class="rank">2</div>'
        '<a href="/nfl/teams/BAL/baltimore-ravens/"><span class="player-name">Ravens DST</span></a>'
        '<span class="team position">DST</span></div>'
        '<div class="player-row"><div class="rank">3</div>'
        '<a href="/nfl/players/2/jamarr-chase/fantasy/"><span class="player-name">J. Chase</span></a>'
        '<span class="team position">WR $38</span></div>'
    )
    df = parse_cbs(html)
    assert df["cbs_rank"].tolist() == [1, 3]
    assert df["cbs_name"].tolist() == ["jahmyr gibbs", "jamarr chase"]
