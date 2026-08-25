import numpy as np
import pandas as pd
import pytest

from pipeline.db import get_conn, write_table
from scoring import oline

DT = "2026-08-09T00:00:00Z"
STALE_DT = "2026-03-01T00:00:00Z"
SEASON = 2025  # the draft this data supports; only 2023/2024 history exists


def _dc_row(team, pos_name, pos_rank, gsis_id, name, dt=DT):
    return {"dt": dt, "team": team, "player_name": name, "gsis_id": gsis_id,
            "pos_name": pos_name, "pos_rank": pos_rank}


def _snap_rows(pfr_id, name, team, season, weeks, snaps, pct, position="T"):
    return [{"season": season, "week": w, "team": team, "player": name,
              "pfr_player_id": pfr_id, "position": position, "game_type": "REG",
              "offense_snaps": snaps, "offense_pct": pct} for w in weeks]


def _seed(tmp_path):
    """Three teams that isolate one signal each:

    AAA -- the same five veterans start every game of 2023 and 2024, and
    the same five are penciled in for 2025. Continuity, availability,
    returning-starters and experience should all read as strong.

    BBB -- three iron-man starters, one (B4) hurt after week 10 of 2024
    and replaced by B5 for the rest of the season, plus B6 spelling the
    tackles a little every week. 2025's depth chart keeps B1/B2/B3/B5 and
    adds a brand new signee (BN) with no snap history anywhere. Exercises
    a lower (not zero) continuity, a below-1.0 team availability driven by
    B5's short track record, and 4/5 returning starters.

    CCC -- an entirely new five with zero snap_counts history anywhere
    (a synthetic stand-in for "the data doesn't cover this team's past").
    Exercises the missing-history -> neutral-50 convention, as distinct
    from a genuinely-measured worst case (0 returning starters, 0 seasons
    of experience), which must NOT be neutral-filled.
    """
    conn = get_conn(str(tmp_path / "t.duckdb"))

    players = []
    snaps = []
    depth = []

    # --- AAA: stable veteran line, 2023 + 2024, all 17 games both years ---
    aaa_rookie = {"A1": 2016, "A2": 2017, "A3": 2018, "A4": 2019, "A5": 2020}
    for gsis, rookie in aaa_rookie.items():
        name = f"Player {gsis}"
        players.append({"gsis_id": gsis, "display_name": name,
                         "birth_date": "1995-01-01", "rookie_season": rookie})
        for season in (2023, 2024):
            snaps += _snap_rows(f"pfr{gsis}", name, "AAA", season, range(1, 18), 60, 1.0)
        depth.append(_dc_row("AAA", "Left Tackle" if gsis == "A1" else
                              "Left Guard" if gsis == "A2" else
                              "Center" if gsis == "A3" else
                              "Right Guard" if gsis == "A4" else "Right Tackle",
                              1, gsis, name))
    # A backup behind A1, to prove pos_rank (not row order) picks the starter.
    players.append({"gsis_id": "A1B", "display_name": "Player A1B",
                     "birth_date": "1999-01-01", "rookie_season": 2023})
    depth.append(_dc_row("AAA", "Left Tackle", 2, "A1B", "Player A1B"))

    # --- BBB: churn + one in-season injury, 2024 only ---
    bbb = {"B1": 2019, "B2": 2020, "B3": 2021, "B4": 2018, "B5": 2022, "B6": 2023}
    for gsis, rookie in bbb.items():
        name = f"Player {gsis}"
        players.append({"gsis_id": gsis, "display_name": name,
                         "birth_date": "1995-01-01", "rookie_season": rookie})
    players.append({"gsis_id": "BN", "display_name": "Player BN",
                     "birth_date": "2003-01-01", "rookie_season": 2025})
    for gsis in ("B1", "B2", "B3"):
        snaps += _snap_rows(f"pfr{gsis}", f"Player {gsis}", "BBB", 2024, range(1, 18), 60, 1.0)
    snaps += _snap_rows("pfrB4", "Player B4", "BBB", 2024, range(1, 11), 60, 1.0)   # hurt wk 11
    snaps += _snap_rows("pfrB5", "Player B5", "BBB", 2024, range(11, 18), 60, 1.0)  # replacement
    snaps += _snap_rows("pfrB6", "Player B6", "BBB", 2024, range(1, 18), 9, 0.15)   # swing depth
    for gsis, slot in (("B1", "Left Tackle"), ("B2", "Left Guard"), ("B3", "Center"),
                       ("B5", "Right Guard"), ("BN", "Right Tackle")):
        depth.append(_dc_row("BBB", slot, 1, gsis, f"Player {gsis}"))
    # A stale snapshot naming a different RT -- must be ignored (not the max dt).
    depth.append(_dc_row("BBB", "Right Tackle", 1, "B4", "Player B4", dt=STALE_DT))

    # --- CCC: brand new five, no snap_counts history anywhere ---
    for i in range(1, 6):
        gsis = f"C{i}"
        players.append({"gsis_id": gsis, "display_name": f"Player {gsis}",
                         "birth_date": "2003-01-01", "rookie_season": 2025})
    for gsis, slot in zip(("C1", "C2", "C3", "C4", "C5"),
                          ("Left Tackle", "Left Guard", "Center", "Right Guard", "Right Tackle")):
        depth.append(_dc_row("CCC", slot, 1, gsis, f"Player {gsis}"))

    write_table(conn, "players", pd.DataFrame(players))
    write_table(conn, "snap_counts", pd.DataFrame(snaps))
    write_table(conn, "depth_charts", pd.DataFrame(depth))
    return conn


def _seed_reconciliation(tmp_path):
    """Isolates id-reconciliation from the scoring logic above.

    Two "John Smith"s in `players`: one whose rookie_season rules him out
    for a 2022-debut pfr id (leaving that id genuinely ambiguous, since a
    veteran could plausibly still be playing), and one whose rookie_season
    rules him out for a 2016-debut pfr id (leaving that one uniquely
    resolved). A third pfr id, "Zachary Tomlin", has no matching name at
    all -- `players` only has "Zach Tomlin" -- the nickname-mismatch case
    seen in the real data (e.g. "Michael Onwenu" vs "Mike Onwenu").
    """
    conn = get_conn(str(tmp_path / "recon.duckdb"))
    write_table(conn, "players", pd.DataFrame([
        {"gsis_id": "g1", "display_name": "John Smith",
         "birth_date": "1990-01-01", "rookie_season": 2015},
        {"gsis_id": "g2", "display_name": "John Smith",
         "birth_date": "1998-01-01", "rookie_season": 2021},
        {"gsis_id": "g3", "display_name": "Zach Tomlin",
         "birth_date": "1994-01-01", "rookie_season": 2019},
    ]))
    rows = (
        _snap_rows("SmitJo01", "John Smith", "GB", 2016, [1, 2, 3], 60, 1.0)  # -> g1 only
        + _snap_rows("SmitJo02", "John Smith", "DET", 2022, [1, 2], 60, 1.0)  # ambiguous
        + _snap_rows("TomlZa00", "Zachary Tomlin", "NYJ", 2020, [1], 60, 1.0)  # no match
    )
    write_table(conn, "snap_counts", pd.DataFrame(rows))
    return conn


# ---------------------------------------------------------------- reconcile

def test_reconcile_resolves_unique_name_disambiguated_by_rookie_season(tmp_path):
    mapping = oline.reconcile_pfr_to_gsis(_seed_reconciliation(tmp_path))
    resolved = dict(zip(mapping["pfr_player_id"], mapping["gsis_id"]))
    assert resolved == {"SmitJo01": "g1"}


def test_reconcile_drops_genuinely_ambiguous_and_unmatched_ids(tmp_path):
    """SmitJo02 (2022 debut) can't rule out either John Smith -- a veteran
    from 2015 is still a plausible candidate for a 2022 snap row. TomlZa00
    matches no one by name at all. Neither should appear in the mapping."""
    mapping = oline.reconcile_pfr_to_gsis(_seed_reconciliation(tmp_path))
    assert "SmitJo02" not in set(mapping["pfr_player_id"])
    assert "TomlZa00" not in set(mapping["pfr_player_id"])
    assert len(mapping) == 1


def test_oline_id_coverage_reports_exact_resolution_rates(tmp_path):
    cov = oline.oline_id_coverage(_seed_reconciliation(tmp_path))
    assert cov["pfr_ids"] == 3
    assert cov["resolved_ids"] == 1
    assert cov["resolved_pct"] == pytest.approx(100 / 3, abs=0.05)
    # SmitJo01 contributes 3 of the 6 total OL rows in this fixture.
    assert cov["rows"] == 6
    assert cov["rows_resolved"] == 3
    assert cov["rows_resolved_pct"] == pytest.approx(50.0)


def test_oline_id_coverage_empty_db(tmp_path):
    conn = get_conn(str(tmp_path / "empty.duckdb"))
    cov = oline.oline_id_coverage(conn)
    assert cov == {"pfr_ids": 0, "resolved_ids": 0, "resolved_pct": 0.0,
                    "rows": 0, "rows_resolved": 0, "rows_resolved_pct": 0.0}


# -------------------------------------------------------------- line_units

def test_current_starters_use_the_latest_snapshot_and_top_depth_rank(tmp_path):
    """BBB has a stale row naming B4 at right tackle from an earlier `dt`,
    and AAA has a backup (A1B, pos_rank 2) behind its actual starter. Both
    must be ignored -- the max `dt` and pos_rank 1 are what define
    "starter," not row order or a player's mere presence in the table."""
    units = oline.line_units(_seed(tmp_path), SEASON)
    bbb = units[units["team"] == "BBB"].set_index("position")
    assert bbb.loc["Right Tackle", "gsis_id"] == "BN"
    aaa = units[units["team"] == "AAA"].set_index("position")
    assert aaa.loc["Left Tackle", "gsis_id"] == "A1"


def test_line_units_returns_five_positions_per_team(tmp_path):
    units = oline.line_units(_seed(tmp_path), SEASON)
    for team in ("AAA", "BBB", "CCC"):
        rows = units[units["team"] == team]
        assert sorted(rows["position"]) == sorted(oline.LINE_SLOTS)


def test_availability_reflects_an_injury_shortened_replacement(tmp_path):
    """B5 only has 7 of BBB's 17 games on record (he took over in week 11),
    so his personal availability is 7/17, not 1.0 like the iron-man
    starters next to him on the same line."""
    units = oline.line_units(_seed(tmp_path), SEASON).set_index("gsis_id")
    assert units.loc["B5", "availability"] == pytest.approx(7 / 17)
    assert units.loc["B1", "availability"] == pytest.approx(1.0)
    # BN has never taken an offensive snap on record -- unknown, not zero.
    assert pd.isna(units.loc["BN", "availability"])


def test_availability_accumulates_across_prior_seasons(tmp_path):
    """A1 played all 17 games in both 2023 and 2024 -- 34 of 34 possible,
    not just his most recent season."""
    units = oline.line_units(_seed(tmp_path), SEASON).set_index("gsis_id")
    assert units.loc["A1", "games_played"] == 34
    assert units.loc["A1", "team_games_possible"] == 34


def test_line_units_empty_without_a_depth_chart(tmp_path):
    conn = get_conn(str(tmp_path / "nodc.duckdb"))
    units = oline.line_units(conn, SEASON)
    assert units.empty
    assert list(units.columns) == oline.LINE_UNITS_COLUMNS


# ------------------------------------------------------------ line_quality

def test_continuity_is_lower_for_a_shuffled_line(tmp_path):
    """AAA: exactly five linemen played all of 2024, so the top-5 by snaps
    *are* the whole line -- continuity is exactly 1.0. BBB's swing lineman
    (B6) and in-season replacement (B5) mean a sixth person shared the
    snaps, so the top-5 share is less than the whole. Hand-computed:
    starters 1020*3 + B4's 600 + B5's 420 = 4080 of a 4233 total."""
    q = oline.line_quality(_seed(tmp_path), SEASON).set_index("team")
    assert q.loc["AAA", "continuity_raw"] == pytest.approx(1.0)
    assert q.loc["BBB", "continuity_raw"] == pytest.approx(4080 / 4233)
    assert q.loc["BBB", "continuity_raw"] < q.loc["AAA", "continuity_raw"]


def test_continuity_is_nan_with_fewer_than_five_linemen_on_record(tmp_path):
    """A team where only three players ever manned the line that season
    has no meaningful "top five share" -- reading it as 1.0 (a perfectly
    stable line) would be a fabrication, not a measurement."""
    conn = get_conn(str(tmp_path / "thin.duckdb"))
    players, snaps, depth = [], [], []
    for i, gsis in enumerate(("D1", "D2", "D3")):
        name = f"Player {gsis}"
        players.append({"gsis_id": gsis, "display_name": name,
                         "birth_date": "1995-01-01", "rookie_season": 2019})
        snaps += _snap_rows(f"pfr{gsis}", name, "DDD", 2024, range(1, 18), 60, 1.0)
    for gsis, slot in zip(("D1", "D2", "D3"), ("Left Tackle", "Left Guard", "Center")):
        depth.append(_dc_row("DDD", slot, 1, gsis, f"Player {gsis}"))
    write_table(conn, "players", pd.DataFrame(players))
    write_table(conn, "snap_counts", pd.DataFrame(snaps))
    write_table(conn, "depth_charts", pd.DataFrame(depth))

    q = oline.line_quality(conn, SEASON).set_index("team")
    assert pd.isna(q.loc["DDD", "continuity_raw"])
    # Missing, not measured -- normalize_within_position must neutral-fill it.
    assert q.loc["DDD", "continuity_n"] == pytest.approx(50.0)


def test_returning_counts_overlap_with_last_seasons_primary_five(tmp_path):
    """AAA's 2025 five are exactly last season's five (5/5). BBB kept
    B1/B2/B3/B5 and added a new player BN in place of B4 (4/5). CCC's five
    were never on this team's line before -- a real, measured zero, not a
    missing-data gap, since we know exactly who all five are."""
    q = oline.line_quality(_seed(tmp_path), SEASON).set_index("team")
    assert q.loc["AAA", "returning_raw"] == pytest.approx(1.0)
    assert q.loc["BBB", "returning_raw"] == pytest.approx(4 / 5)
    assert q.loc["CCC", "returning_raw"] == pytest.approx(0.0)
    assert not pd.isna(q.loc["CCC", "returning_raw"])


def test_experience_is_seasons_since_rookie_season(tmp_path):
    """AAA's five debuted 2016-2020, so as of the 2025 draft they average
    (9+8+7+6+5)/5 = 7 years in. CCC are all rookie_season 2025 -> 0."""
    q = oline.line_quality(_seed(tmp_path), SEASON).set_index("team")
    assert q.loc["AAA", "experience_raw"] == pytest.approx(7.0)
    assert q.loc["CCC", "experience_raw"] == pytest.approx(0.0)


def test_missing_history_is_neutral_but_measured_zero_is_not(tmp_path):
    """CCC has no continuity/availability signal at all (neutral 50), but
    it does have a measured, real 0.0 returning-starters figure and the
    lowest experience in the league -- those must rank low, not neutral."""
    q = oline.line_quality(_seed(tmp_path), SEASON).set_index("team")
    assert q.loc["CCC", "continuity_n"] == pytest.approx(50.0)
    assert q.loc["CCC", "availability_n"] == pytest.approx(50.0)
    assert q.loc["CCC", "returning_n"] < 50.0
    assert q.loc["CCC", "experience_n"] < 50.0


def test_line_quality_is_the_documented_weighted_sum(tmp_path):
    """The headline number must equal what its own component columns say
    it is -- decomposable, not a black box with a matching label."""
    q = oline.line_quality(_seed(tmp_path), SEASON)
    for _, row in q.iterrows():
        expected = (row["continuity_n"] * oline.WEIGHTS["continuity"]
                    + row["availability_n"] * oline.WEIGHTS["availability"]
                    + row["returning_n"] * oline.WEIGHTS["returning"]
                    + row["experience_n"] * oline.WEIGHTS["experience"])
        assert row["line_quality"] == pytest.approx(expected)


def test_line_quality_ranks_the_stable_veteran_line_highest(tmp_path):
    q = oline.line_quality(_seed(tmp_path), SEASON).set_index("team")
    assert q.loc["AAA", "line_quality"] > q.loc["BBB", "line_quality"]
    assert q.loc["BBB", "line_quality"] > q.loc["CCC", "line_quality"]


def test_no_lookahead_2025_data_does_not_change_a_2025_rating(tmp_path):
    """A rating for season 2025 may read 2023/2024 and 2025's preseason
    depth chart, never 2025's own results. Adding 2025 snap rows that
    would (if read) make BBB look like the most continuous, healthiest
    team in the league must not move its score at all."""
    conn = _seed(tmp_path)
    before = oline.line_quality(conn, SEASON)
    before_units = oline.line_units(conn, SEASON)

    sabotage = _snap_rows("pfrB1", "Player B1", "BBB", 2025, range(1, 18), 60, 1.0)
    sabotage += _snap_rows("pfrB2", "Player B2", "BBB", 2025, range(1, 18), 60, 1.0)
    sabotage += _snap_rows("pfrB3", "Player B3", "BBB", 2025, range(1, 18), 60, 1.0)
    sabotage += _snap_rows("pfrB5", "Player B5", "BBB", 2025, range(1, 18), 60, 1.0)
    existing = conn.execute("SELECT * FROM snap_counts").df()
    write_table(conn, "snap_counts", pd.concat(
        [existing, pd.DataFrame(sabotage)], ignore_index=True))

    after = oline.line_quality(conn, SEASON)
    after_units = oline.line_units(conn, SEASON)
    pd.testing.assert_frame_equal(before, after)
    pd.testing.assert_frame_equal(before_units, after_units)


def test_line_quality_empty_db_returns_correct_columns(tmp_path):
    conn = get_conn(str(tmp_path / "empty2.duckdb"))
    q = oline.line_quality(conn, SEASON)
    assert q.empty
    assert list(q.columns) == oline.LINE_QUALITY_COLUMNS


def _seed_a_father_and_a_son(tmp_path):
    """The collision that folding a suffix creates.

    `_norm_name` strips "Jr." so a son folds onto his father, and the
    plausibility test only rules out candidates whose careers began too LATE
    -- a 1996 rookie is never too old for a 2024 snap row. Both survive, the
    pair reads as ambiguous, and the son loses his per-game snap share.
    """
    conn = get_conn(str(tmp_path / "family.duckdb"))
    write_table(conn, "players", pd.DataFrame([
        {"gsis_id": "dad", "display_name": "Marvin Harrison",
         "birth_date": "1972-08-25", "rookie_season": 1996},
        {"gsis_id": "son", "display_name": "Marvin Harrison Jr.",
         "birth_date": "2002-08-07", "rookie_season": 2024},
    ]))
    write_table(conn, "snap_counts", pd.DataFrame(
        _snap_rows("HarrMa09", "Marvin Harrison Jr.", "ARI", 2024, [1, 2], 60, 1.0)))
    return conn


def test_reconcile_resolves_a_son_from_his_father(tmp_path):
    """Twenty-eight seasons apart is not two men who could both have taken
    these snaps. The one whose career actually explains them wins."""
    mapping = oline.reconcile_pfr_to_gsis(_seed_a_father_and_a_son(tmp_path),
                                          positions=None)
    assert dict(zip(mapping["pfr_player_id"], mapping["gsis_id"])) == {
        "HarrMa09": "son"}


def test_reconcile_still_refuses_a_close_call(tmp_path):
    """The margin rule only fires on daylight. `_seed_reconciliation`'s
    SmitJo02 sits 7 and 1 seasons from its two candidates -- a margin of 6,
    under `_ERA_GAP` -- and a veteran seven years in really could be the man
    on those snaps. It stays dropped, because a wrong per-game snap share is
    a lie told every week rather than a gap admitted once."""
    mapping = oline.reconcile_pfr_to_gsis(_seed_reconciliation(tmp_path))
    assert "SmitJo02" not in set(mapping["pfr_player_id"])


def test_experience_drops_an_implausible_career_rather_than_capping_it(tmp_path):
    """A career longer than any lineman has actually had is not a durable
    veteran, it is a join that landed on the wrong man: the depth chart
    carries a gsis_id, and for a rookie who shares his name with a player
    from the nineties it is sometimes that player's. Indianapolis' 2026
    right tackle read as 34 seasons and dragged his line's mean from under
    three to nine.

    NaN, not a clip: capping would put a plausible 22 on a man who has played
    none, which is the same lie in a quieter voice. `line_quality`'s
    normalization already treats a missing part as neutral.
    """
    from scoring.oline import _MAX_CAREER, _experience

    conn = get_conn(str(tmp_path / "exp.duckdb"))
    write_table(conn, "players", pd.DataFrame([
        {"gsis_id": "real", "rookie_season": 2020},
        {"gsis_id": "rookie", "rookie_season": 2026},
        {"gsis_id": "ghost", "rookie_season": 1992},
    ]))
    out = _experience(conn, 2026).set_index("gsis_id")["seasons_in_league"]
    assert out["real"] == 6
    assert out["rookie"] == 0
    assert pd.isna(out["ghost"]), "34 seasons is an identity error, not a career"
    assert _MAX_CAREER >= 21, "Jason Peters played 21; the guard must clear him"
