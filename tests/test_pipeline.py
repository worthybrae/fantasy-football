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
