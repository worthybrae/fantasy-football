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

def test_get_conn_migrates_old_drafted_table_without_dropping_rows(tmp_path):
    """Task 13 adds a pick_no column to `drafted` so POST /api/drafted can
    report pick order. A real data/nfl.duckdb from before this change only
    has the single-column (player_id) drafted table -- get_conn's ALTER
    TABLE migration must add pick_no in place, not drop and recreate the
    table (which would silently wipe a real user's drafted list)."""
    import duckdb
    path = str(tmp_path / "old.duckdb")
    old = duckdb.connect(path)
    old.execute("CREATE TABLE drafted (player_id VARCHAR PRIMARY KEY)")
    old.execute("INSERT INTO drafted VALUES ('p1')")
    old.close()

    conn = get_conn(path)
    columns = {r[1] for r in conn.execute("PRAGMA table_info('drafted')").fetchall()}
    assert "pick_no" in columns
    assert conn.execute("SELECT player_id, pick_no FROM drafted").fetchall() == [("p1", None)]

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
    # `format` is additive: same four columns as before, plus the format tag.
    # Default fmt='ppr' keeps a caller that doesn't pass fmt getting exactly
    # today's shape (see the multi-scoring-format ADP ingestion spec).
    assert list(df.columns) == ["adp_name", "position", "team", "adp", "format"]
    assert (df["format"] == "ppr").all()
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

def test_espn_adp_is_usable_rejects_a_constant_column():
    # ESPN returns 170.0 for every player in some completed seasons -- a
    # reset value, not a ranking. Verified against the live 2025 season.
    from pipeline.sources import espn_adp_is_usable
    constant = pd.DataFrame({"espn_adp": [170.0, 170.0, 170.0]})
    assert not espn_adp_is_usable(constant)
    assert not espn_adp_is_usable(pd.DataFrame({"espn_adp": [None, None]}))
    assert espn_adp_is_usable(pd.DataFrame({"espn_adp": [1.7, 2.6, 3.8]}))


def test_espn_adp_is_usable_on_a_missing_column():
    from pipeline.sources import espn_adp_is_usable
    assert not espn_adp_is_usable(pd.DataFrame({"other": [1, 2]}))


# --- ESPN preseason cheat sheet (PPR300) ------------------------------------
# The reference `draft_model._enrich_pool` fits on and `draft_sim.build_pool`
# ranks the live board on. Text is extracted from a PDF laid out in four
# columns, so one physical line carries four entries at ranks n, n+80, n+160
# and n+240 -- a per-line parse would only ever see the first.

_CHEATSHEET_TEXT = (
    "1. (RB1) Jahmyr Gibbs, DET $57 6 81. (WR40) Brian Thomas Jr., JAC $4 7 "
    "161. (WR67) Jalen Nailor, LV $0 13 241. (RB70) LeQuint Allen, JAC $0 7\n"
    "2. (RB2) Bijan Robinson, ATL $56 11 82. (QB8) Trevor Lawrence, JAC $4 7 "
    "162. (DST1) Broncos D/ST, DEN $0 13 242. (K1) Brandon Aubrey, DAL $0 11\n"
)


def test_parse_cheatsheet_reads_all_four_columns_of_a_line():
    df = sources.parse_espn_cheatsheet_text(_CHEATSHEET_TEXT, 2026)
    assert df["cs_rank"].tolist() == [1, 2, 81, 82, 161, 162, 241, 242]
    assert df.set_index("cs_rank").loc[1, "cs_name"] == "Jahmyr Gibbs"
    assert df.set_index("cs_rank").loc[241, "cs_name"] == "LeQuint Allen"


def test_parse_cheatsheet_drops_the_duplicate_rank_one_header():
    """Every real sheet yields 301 matches for 300 players -- rank 1 appears
    twice as a header artifact. Deduping on rank is what makes "one row per
    rank" hold rather than trusting the match count."""
    doubled = "1. (RB1) Jahmyr Gibbs, DET $57 6\n" + _CHEATSHEET_TEXT
    df = sources.parse_espn_cheatsheet_text(doubled, 2026)
    assert df["cs_rank"].is_unique
    assert (df["cs_rank"] == 1).sum() == 1


def test_parse_cheatsheet_normalizes_dst_and_keeps_auction_value():
    df = sources.parse_espn_cheatsheet_text(_CHEATSHEET_TEXT, 2026).set_index("cs_rank")
    # ESPN's position tag is "DST" but the name carries "D/ST"; the rest of
    # the codebase says DST, and `adp_match_key` keys a defense on its team.
    assert df.loc[162, "position"] == "DST" and df.loc[162, "team"] == "DEN"
    assert df.loc[242, "position"] == "K"
    # Nothing consumes the auction value yet; it is a cardinal price signal
    # that rank cannot express, captured so a later feature can use it.
    assert df.loc[1, "auction_value"] == 57.0
    assert df.loc[161, "auction_value"] == 0.0
    assert df["season"].unique().tolist() == [2026]


def test_parse_cheatsheet_reads_a_row_with_no_space_before_the_price():
    """A long name pushes the auction column flush against the team
    ("CLE$0"). Requiring whitespace there silently dropped exactly one
    player from each of the 2021 and 2022 sheets."""
    text = "274. (WR100) Donovan Peoples-Jones, CLE$0 13\n"
    df = sources.parse_espn_cheatsheet_text(text, 2021)
    assert df["cs_rank"].tolist() == [274]
    assert df.iloc[0]["cs_name"] == "Donovan Peoples-Jones"
    assert df.iloc[0]["team"] == "CLE" and df.iloc[0]["auction_value"] == 0.0


def test_cheatsheet_url_switches_naming_convention_in_2023():
    assert sources.cheatsheet_url(2021).endswith("/21/NFLDK2021_CS_PPR300.pdf")
    assert sources.cheatsheet_url(2022).endswith("/22/NFLDK2022_CS_PPR300.pdf")
    assert sources.cheatsheet_url(2023).endswith("/23/NFL23_CS_PPR300.pdf")
    assert sources.cheatsheet_url(2026).endswith("/26/NFL26_CS_PPR300.pdf")


# --- Multi-scoring-format ADP ingestion -------------------------------------
# FFC/FantasyPros/CBS/MFL each gain a `fmt` argument ('ppr' | 'half' | 'std',
# default 'ppr') so the board's consensus ADP can be built per league scoring
# instead of hardcoded to PPR. Every fetcher tags its rows with the format it
# fetched; ESPN stays untouched (its rank is explicitly PPR, no other format
# exists to ask for). See pipeline/sources.py's per-source comments for how
# each site's format support was verified against a live fetch.

class _URLCapture:
    """Stands in for `requests.get`: records every URL it was called with
    (so a test can assert which URL a `fmt` value maps to) and returns a
    canned response instead of hitting the network."""
    def __init__(self, response):
        self.urls = []
        self._response = response
    def __call__(self, url, *a, **k):
        self.urls.append(url)
        return self._response

def _no_request_allowed(*a, **k):
    raise AssertionError("requests.get should not have been called")


def test_fetch_adp_std_hits_the_standard_url_and_tags_rows(monkeypatch):
    payload = {"players": [{"name": "Saquon Barkley", "position": "RB",
                            "team": "PHI", "adp": 1.0}]}
    capture = _URLCapture(_StubResponse(json_data=payload))
    monkeypatch.setattr(sources.requests, "get", capture)
    df = sources.fetch_adp(2026, fmt="std")
    assert capture.urls == ["https://fantasyfootballcalculator.com/api/v1/adp/standard"]
    assert (df["format"] == "std").all()


def test_fetch_adp_default_fmt_is_ppr_and_reproduces_todays_rows(monkeypatch):
    """The default fmt='ppr' path must hit exactly the URL it always did and
    produce exactly the same rows -- just carrying a format='ppr' tag now.
    This is the byte-identical-PPR-pass guarantee the board's existing
    consumers depend on."""
    payload = {"players": [
        {"name": "Justin Jefferson", "position": "WR", "team": "MIN", "adp": 3.2}]}
    capture = _URLCapture(_StubResponse(json_data=payload))
    monkeypatch.setattr(sources.requests, "get", capture)
    df = sources.fetch_adp(2026)
    assert capture.urls == ["https://fantasyfootballcalculator.com/api/v1/adp/ppr"]
    assert df.to_dict("records") == [
        {"adp_name": "Justin Jefferson", "position": "WR", "team": "MIN",
         "adp": 3.2, "format": "ppr"}]


def test_fetch_adp_unsupported_format_raises_before_any_request(monkeypatch):
    monkeypatch.setattr(sources.requests, "get", _no_request_allowed)
    with pytest.raises(ValueError):
        sources.fetch_adp(2026, fmt="bogus")


def test_fetch_fp_ecr_std_hits_the_standard_cheatsheets_url(monkeypatch):
    html = ('<script>var ecrData = {"players": [{"player_name": "Test Guy",'
            '"player_team_id": "KC", "player_position_id": "RB", "rank_ecr": 1,'
            '"rank_ave": "1.0", "rank_std": "0.1", "tier": 1}]};</script>')
    capture = _URLCapture(_StubResponse(text=html))
    monkeypatch.setattr(sources.requests, "get", capture)
    df = sources.fetch_fp_ecr(fmt="std")
    assert capture.urls == ["https://www.fantasypros.com/nfl/rankings/standard-cheatsheets.php"]
    assert (df["format"] == "std").all()


def test_fetch_fp_ecr_default_fmt_is_ppr_and_hits_the_ppr_url(monkeypatch):
    html = ('<script>var ecrData = {"players": [{"player_name": "JaMarr Chase",'
            '"player_team_id": "CIN", "player_position_id": "WR", "rank_ecr": 1,'
            '"rank_ave": "1.77", "rank_std": "1.2", "tier": 1}]};</script>')
    capture = _URLCapture(_StubResponse(text=html))
    monkeypatch.setattr(sources.requests, "get", capture)
    df = sources.fetch_fp_ecr()
    assert capture.urls == ["https://www.fantasypros.com/nfl/rankings/ppr-cheatsheets.php"]
    assert (df["format"] == "ppr").all()


def test_fetch_cbs_half_ppr_is_not_a_supported_format(monkeypatch):
    """CBS's half-ppr top200 page 404s (verified against the live site) --
    fetch_cbs raises before making a request rather than surfacing a raw
    404, and refresh's per-format loop (pipeline/refresh.py's
    _fetch_multi_format) treats that as "this source contributes no half
    rows", not a crash."""
    monkeypatch.setattr(sources.requests, "get", _no_request_allowed)
    with pytest.raises(ValueError):
        sources.fetch_cbs(fmt="half")


def test_fetch_cbs_std_hits_the_standard_url(monkeypatch):
    html = ('<div class="player-row first"><div class="rank">1</div>'
            '<a href="/nfl/players/1/saquon-barkley/fantasy/">'
            '<span class="player-name">S. Barkley</span></a>'
            '<span class="team position">RB $34</span></div>')
    capture = _URLCapture(_StubResponse(text=html))
    monkeypatch.setattr(sources.requests, "get", capture)
    df = sources.fetch_cbs(fmt="std")
    assert capture.urls == ["https://www.cbssports.com/fantasy/football/rankings/standard/top200/"]
    assert (df["format"] == "std").all()


def test_fetch_mfl_adp_std_requests_is_ppr_0(monkeypatch):
    adp_payload = {"adp": {"player": [{"id": "1", "averagePick": "2.0", "rank": "1"}]}}
    players_payload = {"players": {"player": [
        {"id": "1", "name": "Barkley, Saquon", "position": "RB", "team": "PHI"}]}}
    responses = iter([_StubResponse(json_data=adp_payload),
                      _StubResponse(json_data=players_payload)])
    capture = _URLCapture(None)
    def fake_get(url, *a, **k):
        capture.urls.append(url)
        return next(responses)
    monkeypatch.setattr(sources.requests, "get", fake_get)
    df = sources.fetch_mfl_adp(2026, fmt="std")
    assert "IS_PPR=0" in capture.urls[0]
    assert (df["format"] == "std").all()


def test_fetch_mfl_adp_default_fmt_requests_is_ppr_1(monkeypatch):
    adp_payload = {"adp": {"player": [{"id": "1", "averagePick": "2.0", "rank": "1"}]}}
    players_payload = {"players": {"player": [
        {"id": "1", "name": "Chase, JaMarr", "position": "WR", "team": "CIN"}]}}
    responses = iter([_StubResponse(json_data=adp_payload),
                      _StubResponse(json_data=players_payload)])
    capture = _URLCapture(None)
    def fake_get(url, *a, **k):
        capture.urls.append(url)
        return next(responses)
    monkeypatch.setattr(sources.requests, "get", fake_get)
    df = sources.fetch_mfl_adp(2026)
    assert "IS_PPR=1" in capture.urls[0]
    assert (df["format"] == "ppr").all()


def test_fetch_mfl_adp_has_no_half_format(monkeypatch):
    """MFL's ADP export is a binary IS_PPR flag -- there is no half-PPR
    value to request at all, unlike CBS's 404. Both must be skippable by
    refresh's per-format loop without erroring, which is what matters; the
    ValueError here is what makes that possible."""
    monkeypatch.setattr(sources.requests, "get", _no_request_allowed)
    with pytest.raises(ValueError):
        sources.fetch_mfl_adp(2026, fmt="half")


def test_parse_mfl_tags_rows_with_the_given_format():
    from pipeline.sources import parse_mfl
    adp = {"adp": {"player": [{"id": "100", "averagePick": "2.85", "rank": "1"}]}}
    players = {"players": {"player": [
        {"id": "100", "name": "Gibbs, Jahmyr", "position": "RB", "team": "DET"}]}}
    df = parse_mfl(adp, players, fmt="std")
    assert df["format"].tolist() == ["std"]


def test_parse_cbs_tags_rows_with_the_given_format():
    from pipeline.sources import parse_cbs
    html = ('<div class="player-row first"><div class="rank">1</div>'
            '<a href="/nfl/players/1/jahmyr-gibbs/fantasy/">'
            '<span class="player-name">J. Gibbs</span></a>'
            '<span class="team position">RB $34</span></div>')
    df = parse_cbs(html, fmt="half")
    assert df["format"].tolist() == ["half"]


def test_refresh_fetch_multi_format_skips_an_unsupported_format_without_erroring():
    """Mirrors CBS/MFL in production: a fetch_fn that raises for one format
    token (the source doesn't support it) must not abort the whole source --
    the other formats' rows still come back, correctly tagged, and the
    caller never sees an exception."""
    from pipeline.refresh import _fetch_multi_format

    def fake_fetch(year, fmt):
        if fmt == "half":
            raise ValueError("no half endpoint")
        return pd.DataFrame({"name": ["A"], "format": [fmt]})

    df = _fetch_multi_format(fake_fetch, ("ppr", "half", "std"), 2026)
    assert sorted(df["format"].tolist()) == ["ppr", "std"]


def test_refresh_fetch_multi_format_raises_if_every_format_fails():
    from pipeline.refresh import _fetch_multi_format

    def fake_fetch(fmt):
        raise ValueError("upstream down")

    with pytest.raises(ValueError):
        _fetch_multi_format(fake_fetch, ("ppr", "std"))


def test_refresh_formats_by_source_matches_verified_support():
    """Locks in exactly what was verified against the live sites: FFC and
    FantasyPros serve all three formats, CBS and MFL only ppr/std (CBS's
    half-ppr page 404s, MFL's IS_PPR flag has no half value)."""
    from pipeline.refresh import FORMATS_BY_SOURCE
    assert set(FORMATS_BY_SOURCE["adp"]) == {"ppr", "half", "std"}
    assert set(FORMATS_BY_SOURCE["fp_ecr"]) == {"ppr", "half", "std"}
    assert set(FORMATS_BY_SOURCE["cbs_ranks"]) == {"ppr", "std"}
    assert set(FORMATS_BY_SOURCE["mfl_adp"]) == {"ppr", "std"}


# ------------------------------------------------- ESPN team-defense stats

def _dst_fixture():
    """Three real defenses, weeks 1-4 of 2025, taken verbatim from ESPN's
    `leaguedefaults/3?view=kona_player_info` response for slot 16 -- plus the
    season-aggregate and projection blocks the parser has to drop."""
    import json
    from pathlib import Path
    path = Path(__file__).parent / "fixtures" / "espn_dst_2025.json"
    return json.loads(path.read_text())


def test_parse_espn_dst_keeps_the_scored_weeks_and_drops_the_rest():
    from pipeline.sources import parse_espn_dst
    df = parse_espn_dst(_dst_fixture())
    # 3 defenses x weeks 1-4. The season aggregate (scoringPeriodId 0) would
    # double every total if kept beside the weeks; the projection blocks
    # (statSourceId 1) are a different question entirely.
    assert len(df) == 12
    assert sorted(df["week"].unique()) == [1, 2, 3, 4]
    assert set(df["season"]) == {2025}
    assert set(df["team"]) == {"DEN", "HOU", "SEA"}
    assert set(df["espn_id"]) == {-16007, -16034, -16026}


def test_parse_espn_dst_reads_the_season_off_the_stat_block():
    """Not off the URL year, which routinely disagrees: asking ESPN for
    season 2026 in August 2026 returns 2025's completed weeks."""
    from pipeline.sources import parse_espn_dst
    payload = {"players": [{"player": {
        "id": -16007, "proTeamId": 7, "fullName": "Broncos D/ST", "stats": [
            {"statSourceId": 0, "scoringPeriodId": 4, "seasonId": 2024,
             "appliedTotal": 3.0, "stats": {"99": 3.0}}]}}]}
    df = parse_espn_dst(payload)
    assert df["season"].tolist() == [2024]


def test_scoring_a_defense_reproduces_espns_own_total_exactly():
    """The whole basis for scoring defenses without identifying a single
    stat: ESPN hands over the values and the point values keyed by the same
    ids, so `sum(stat x points)` IS ESPN's arithmetic. Checked here on real
    weeks; checked at scale (3744 defense-weeks, 0 mismatches) in the note on
    scoring.ppr.DEFAULT_DST_RULES.

    `applied_total` is scored under leaguedefaults/3's rules, and
    DEFAULT_DST_RULES is a live read of exactly those rules -- so the two
    must agree to the bit, not merely to a tolerance.
    """
    from pipeline.sources import parse_espn_dst
    from scoring.ppr import DEFAULT_DST_RULES, compute_dst_points
    df = parse_espn_dst(_dst_fixture())
    assert (compute_dst_points(df, DEFAULT_DST_RULES) == df["applied_total"]).all()
    # None means "not specified" and resolves to the same default.
    assert (compute_dst_points(df) == df["applied_total"]).all()
    # {} means "this league scores no defense" -- a different answer.
    assert (compute_dst_points(df, {}) == 0.0).all()


def test_a_tiered_rule_is_an_ordinary_column_times_points():
    """The thing that used to make defenses unscorable: points allowed is a
    nine-way step function. ESPN emits the bucket already decided, one-hot
    per game, so it is a `column x points` rule like any other.

    Texans week 2 conceded 20 points, which lands in the 18-21 band (statId
    121) -- worth 0 in this scoring, which is why 121 is NOT in
    DEFAULT_DST_RULES at all. Texans week 4 conceded 0, which is statId 89,
    worth 5. Broncos week 2 conceded 29, statId 123, worth -1.
    """
    from pipeline.sources import parse_espn_dst
    from scoring.ppr import DEFAULT_DST_RULES
    df = parse_espn_dst(_dst_fixture()).set_index(["team", "week"])
    tiers = ["89", "90", "91", "92", "121", "122", "123", "124", "125"]
    # A tier no defense landed in has no column at all: ESPN omits a stat id
    # from a week's line when it is zero, so absent means zero here.
    assert "stat_125" not in df.columns
    for key in df.index:
        hot = [t for t in tiers
               if f"stat_{t}" in df.columns and df.loc[key, f"stat_{t}"] == 1.0]
        assert len(hot) == 1, f"{key} is not one-hot across the tier family"
    assert df.loc[("HOU", 2), "stat_120"] == 20.0
    assert df.loc[("HOU", 2), "stat_121"] == 1.0
    assert "121" not in DEFAULT_DST_RULES
    assert df.loc[("HOU", 4), "stat_120"] == 0.0
    assert df.loc[("HOU", 4), "stat_89"] == 1.0
    assert DEFAULT_DST_RULES["89"] == 5.0
    assert df.loc[("DEN", 2), "stat_120"] == 29.0
    assert df.loc[("DEN", 2), "stat_123"] == 1.0
    assert DEFAULT_DST_RULES["123"] == -1.0


def test_fetch_espn_dst_asks_for_the_defense_slot_and_raises_on_empty(monkeypatch):
    """Without `filterSlotIds` the endpoint answers 200 with the 50 most-owned
    PLAYERS and not one defense -- a healthy-looking fetch that silently
    empties the table. Both halves of the guard are pinned."""
    import json
    from pipeline import sources
    seen = {}

    class _Resp:
        def __init__(self, payload):
            self._payload = payload
        def raise_for_status(self):
            pass
        def json(self):
            return self._payload

    def fake_get(url, headers=None, timeout=None):
        seen["url"] = url
        seen["filter"] = json.loads(headers["X-Fantasy-Filter"])
        return _Resp(_dst_fixture())

    monkeypatch.setattr(sources.requests, "get", fake_get)
    df = sources.fetch_espn_dst([2025])
    assert len(df) == 12
    assert seen["filter"]["players"]["filterSlotIds"]["value"] == [16]
    assert "leaguedefaults/3" in seen["url"] and "2025" in seen["url"]

    monkeypatch.setattr(sources.requests, "get",
                        lambda *a, **k: _Resp({"players": []}))
    with pytest.raises(ValueError):
        sources.fetch_espn_dst([2025])


def test_dst_weekly_is_registered_as_a_universal_table():
    """A table in neither set is treated as universal by provisioning anyway,
    so the classification has to be explicit. What a defense did in week 4 is
    the same fact in every league; the LEAGUE-specific half is the price list
    in `league.settings_json`."""
    from pipeline.db import LEAGUE_TABLES, UNIVERSAL_TABLES
    assert "dst_weekly" in UNIVERSAL_TABLES
    assert "dst_weekly" not in LEAGUE_TABLES
