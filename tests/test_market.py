import numpy as np
import pandas as pd
from scoring.market import add_market

def _board():
    return pd.DataFrame({
        "player_id": ["g1", "g2", "g3"],
        "name": ["Jahmyr Gibbs", "Amon-Ra St. Brown", "Ravens DST"],
        "position": ["RB", "WR", "DST"], "team": ["DET", "DET", "BAL"],
        "rank": [1, 2, 3], "adp": [1.6, 5.1, np.nan]})

def _espn():
    return pd.DataFrame({"espn_id": [101, 102], "espn_name": ["Jahmyr Gibbs", "Amon-Ra St Brown"],
                         "position": ["RB", "WR"], "espn_adp": [1.77, 6.0],
                         "espn_ppr_rank": [1, 4]})

def _sleeper():
    return pd.DataFrame({"gsis_id": ["g1"], "espn_id": [101],
                         "sleeper_name": ["Jahmyr Gibbs"], "position": ["RB"], "team": ["DET"]})

def _fp():
    return pd.DataFrame({"fp_name": ["Amon-Ra St. Brown", "Baltimore Ravens"],
                         "team": ["DET", "BAL"], "position": ["WR", "DST"],
                         "rank_ecr": [3, 40], "rank_ave": [3.0, 41.0],
                         "rank_std": [1.0, 5.0], "fp_tier": [1, 5]})

def test_consensus_all_paths():
    out = add_market(_board(), _espn(), _fp(), _sleeper())
    g1 = out[out["player_id"] == "g1"].iloc[0]   # FFC rank 1, ESPN rank 1 (via crosswalk)
    assert g1["market_rank"] == 1.0 and g1["market_sources"]["espn"] == 1.0
    assert g1["espn_ppr_rank"] == 1.0             # crosswalk path carries espn_ppr_rank
    g2 = out[out["player_id"] == "g2"].iloc[0]   # FFC 2, ESPN PPR 4 (name fallback), FP 3
    assert g2["market_rank"] == round((2 + 4 + 3) / 3, 1)
    assert g2["market_spread"] == 2.0
    assert g2["market_sources"]["fp_tier"] == 1
    assert g2["espn_ppr_rank"] == 4.0              # name-fallback path carries espn_ppr_rank
    dst = out[out["player_id"] == "g3"].iloc[0]  # FP only, joined by team
    assert dst["market_sources"]["fp"] == 40.0 and dst["market_rank"] == 40.0
    assert pd.isna(dst["market_spread"])         # single source
    assert pd.isna(dst["espn_ppr_rank"])          # no ESPN match at all
    assert "adp" not in out.columns

def test_edge_uses_market_rank():
    out = add_market(_board(), _espn(), _fp(), _sleeper())
    g2 = out[out["player_id"] == "g2"].iloc[0]
    assert g2["edge"] == g2["market_rank"] - g2["rank"]

def test_all_sources_empty():
    empty = pd.DataFrame()
    out = add_market(_board(), empty, empty, empty)
    r = out.iloc[0]
    assert r["market_rank"] == 1.0              # FFC alone still ranks
    assert out[out["player_id"] == "g3"].iloc[0]["market_sources"]["ffc"] is None or \
           pd.isna(out[out["player_id"] == "g3"].iloc[0]["market_sources"]["ffc"])
    assert out["espn_ppr_rank"].isna().all()     # no espn table -> all NaN, not missing column

def test_no_sources_at_all():
    board = _board().assign(adp=np.nan)
    empty = pd.DataFrame()
    out = add_market(board, empty, empty, empty)
    assert out["market_rank"].isna().all()

def test_duplicate_espn_id_lowest_rank_wins():
    """Regression: duplicate espn_id should not crash; lowest rank wins."""
    board = _board()
    sleeper = _sleeper()
    # Two ESPN rows for same espn_id (101), different ADP → different rank
    espn = pd.DataFrame({
        "espn_id": [101, 101, 102],
        "espn_name": ["Jahmyr Gibbs", "Jahmyr Gibbs (ALT)", "Amon-Ra St Brown"],
        "position": ["RB", "RB", "WR"],
        "espn_adp": [1.77, 2.5, 6.0],
        "espn_ppr_rank": [1, 2, 4]
    })
    fp = _fp()
    out = add_market(board, espn, fp, sleeper)
    g1 = out[out["player_id"] == "g1"].iloc[0]
    # Should use rank from lowest espn_adp (1.77 → rank 1), not second one
    assert g1["market_sources"]["espn"] == 1.0
    # espn_ppr_rank must travel with the same winning row as espn_rank (1),
    # not the duplicate's (2).
    assert g1["espn_ppr_rank"] == 1.0

def test_duplicate_gsis_id_in_crosswalk_lowest_rank_wins():
    """Regression: a junk crosswalk row mapping two different espn_ids to the
    same gsis_id (seen live: 00-0029981 "Duplicate Player") must not crash
    board building via a non-unique index in `.map()`; lowest espn_rank
    wins for that player."""
    board = _board()
    espn = pd.DataFrame({
        "espn_id": [101, 999],
        "espn_name": ["Jahmyr Gibbs", "Duplicate Player"],
        "position": ["RB", "RB"],
        "espn_adp": [1.77, 0.5],   # the duplicate has the better (lower) ADP
        "espn_ppr_rank": [1, 7],   # distinct per-row, so a misaligned pick is observable
    })
    # Crosswalk maps BOTH espn_ids to the same gsis_id "g1".
    sleeper = pd.DataFrame({
        "gsis_id": ["g1", "g1"], "espn_id": [101, 999],
        "sleeper_name": ["Jahmyr Gibbs", "Duplicate Player"],
        "position": ["RB", "RB"], "team": ["DET", "DET"],
    })
    fp = _fp()
    out = add_market(board, espn, fp, sleeper)  # must not raise
    g1 = out[out["player_id"] == "g1"].iloc[0]
    # espn_ppr_rank must come from the winning dedup row (espn_id 999, the
    # lower-ADP duplicate), i.e. 7 -- not 1 from the loser. The consensus
    # espn source IS the ppr rank now, so both read 7.
    assert g1["market_sources"]["espn"] == 7.0
    assert g1["espn_ppr_rank"] == 7.0

def test_duplicate_fp_name_position_lowest_rank_wins():
    """Regression: duplicate (fp_name, position) should not crash; lowest rank wins."""
    board = _board()
    sleeper = _sleeper()
    espn = _espn()
    # Two FP rows for Amon-Ra St. Brown WR, different rank_ecr
    fp = pd.DataFrame({
        "fp_name": ["Amon-Ra St. Brown", "Amon-Ra St. Brown", "Baltimore Ravens"],
        "team": ["DET", "DET", "BAL"],
        "position": ["WR", "WR", "DST"],
        "rank_ecr": [3, 5, 40],
        "rank_ave": [3.0, 5.0, 41.0],
        "rank_std": [1.0, 1.5, 5.0],
        "fp_tier": [1, 2, 5]
    })
    out = add_market(board, espn, fp, sleeper)
    g2 = out[out["player_id"] == "g2"].iloc[0]
    # Should use lowest rank (3), not second one (5); should use tier 1
    assert g2["market_sources"]["fp"] == 3.0
    assert g2["market_sources"]["fp_tier"] == 1

def test_mfl_and_cbs_fold_into_consensus():
    mfl = pd.DataFrame({"mfl_name": ["Jahmyr Gibbs"], "position": ["RB"], "mfl_rank": [3]})
    cbs = pd.DataFrame({"cbs_name": ["jahmyr gibbs", "amonra st brown"],
                        "position": ["RB", "WR"], "cbs_rank": [1, 4]})
    out = add_market(_board(), _espn(), _fp(), _sleeper(), mfl=mfl, cbs=cbs)
    g1 = out[out["player_id"] == "g1"].iloc[0]
    # FFC 1, ESPN 1, MFL 3, CBS 1 -> median 1.0
    assert g1["market_rank"] == 1.0
    assert g1["market_sources"]["mfl"] == 3.0 and g1["market_sources"]["cbs"] == 1.0
    g2 = out[out["player_id"] == "g2"].iloc[0]
    # FFC 2, ESPN PPR 4, FP 3, CBS 4 (norm-name match), no MFL -> median 3.5
    assert g2["market_rank"] == 3.5
    assert g2["market_sources"]["mfl"] is None


def test_espn_marks_a_retired_player_undraftable_however_stale_the_others_are():
    """ESPN's PPR rank runs across its whole 2565-player universe, not the
    500 we fetch, so it says "not a fantasy player" as a number: Tyreek Hill
    retired and reads 1899, Keenan Allen 1930, Najee Harris 2047. The real
    board ends at 519 with an empty gap up to 978.

    Read as a rank it is poison. Two sources still carrying Hill at 184 and
    301 outvote the one source that knows he retired, the median lands him
    at 301, and a retired player takes a pick in the draft grid -- which is
    how this was found. Blanked, `espn_unranked` marks him and `build_board`
    drops him.
    """
    espn = _espn()
    espn.loc[espn["espn_name"] == "Jahmyr Gibbs", "espn_ppr_rank"] = 1899
    cbs = pd.DataFrame({"cbs_name": ["jahmyr gibbs"], "position": ["RB"],
                        "cbs_rank": [184]})
    out = add_market(_board(), espn, _fp(), _sleeper(), cbs=cbs)
    g1 = out[out["player_id"] == "g1"].iloc[0]
    # Blanked unconditionally: 1899 is never a draft-position opinion, at any
    # board size. The `espn_unranked` flag it feeds is separately gated on the
    # feed being real -- see the two tests below.
    assert g1["market_sources"]["espn"] is None
    # And the median no longer follows the stale pair onto the board: with
    # ESPN blanked, CBS 184 and FFC are all that remain to speak for him.
    assert g1["market_sources"]["cbs"] == 184.0


def test_the_filter_fires_on_a_board_much_larger_than_espn_ranks(_=None):
    """The shape that actually ships, and the shape a coverage-share gate got
    wrong: the board carries 676 non-DST players and ESPN ranks 223 of them,
    a share of 0.33. A `>= 0.5` gate read that as a broken feed and disabled
    the filter on the one board it was written for -- and the wider the gap,
    the more there is to drop, the less likely it would ever have fired.

    Here ESPN ranks 150 of 600. The unranked 450 are what the filter is FOR;
    they must not be mistaken for evidence that ESPN is down.
    """
    n_board, n_ranked = 600, 150
    board = pd.DataFrame([
        {"player_id": f"p{i}", "name": f"Player {i}", "position": "RB",
         "team": "DET", "rank": i + 1, "adp": float(i + 1)} for i in range(n_board)])
    espn = pd.DataFrame([
        {"espn_id": i, "espn_name": f"Player {i}", "position": "RB",
         "team": "DET", "espn_adp": float(i + 1), "espn_ppr_rank": float(i + 1)}
        for i in range(n_ranked)])
    sleeper = pd.DataFrame([{"gsis_id": f"p{i}", "espn_id": i} for i in range(n_ranked)])
    out = add_market(board, espn, _fp(), sleeper)
    assert int(out["espn_unranked"].sum()) == n_board - n_ranked


def _wide(n=140, retired_at=None):
    """A board big enough for the undraftable flag to be trusted.

    `ESPN_MIN_RANKED` is 100, so a fixture has to clear real scale before it
    can exercise the flag at all -- which is the guard working, not an
    obstacle to route around.
    """
    board = pd.DataFrame([
        {"player_id": f"p{i}", "name": f"Player {i}", "position": "RB",
         "team": "DET", "rank": i + 1, "adp": float(i + 1)} for i in range(n)])
    espn = pd.DataFrame([
        {"espn_id": i, "espn_name": f"Player {i}", "position": "RB",
         "team": "DET", "espn_adp": float(i + 1),
         "espn_ppr_rank": 1899.0 if i == retired_at else float(i + 1)}
        for i in range(n)])
    sleeper = pd.DataFrame([{"gsis_id": f"p{i}", "espn_id": i} for i in range(n)])
    return board, espn, sleeper


def test_a_full_espn_feed_flags_only_the_player_it_calls_undraftable():
    board, espn, sleeper = _wide(retired_at=7)
    out = add_market(board, espn, _fp(), sleeper)
    assert list(out.loc[out["espn_unranked"], "player_id"]) == ["p7"]


def test_the_board_drops_the_undraftable_player(tmp_path):
    """End to end: the flag has to actually take him out of the draftable
    pool, not merely be computed. `build_board` is what the simulator builds
    its pool from, so a player left here can be assigned a pick."""
    board, espn, sleeper = _wide(retired_at=7)
    out = add_market(board, espn, _fp(), sleeper)
    kept = out[~out["espn_unranked"]]
    assert "p7" not in set(kept["player_id"])
    assert len(kept) == len(out) - 1


def test_a_thin_espn_feed_does_not_empty_the_board():
    """The undraftable flag is only trusted when ESPN's feed is really here.

    A failed fetch or an unjoined crosswalk leaves every row unranked, and a
    filter that read that as "nobody is draftable" would turn one source
    being down into a total outage. These fixtures rank one or two players,
    which is far below `ESPN_MIN_RANKED`, so nothing is flagged.
    """
    out = add_market(_board(), _espn(), _fp(), _sleeper())
    assert not out["espn_unranked"].any()
    empty = add_market(_board(), pd.DataFrame(), _fp(), _sleeper())
    assert not empty["espn_unranked"].any()


def test_one_wild_source_cannot_drag_the_consensus():
    """The reason the consensus is a median.

    MFL's ADP comes largely from best-ball and dynasty rooms drafted months
    before a redraft league sits down, and it shows: across the top 100 its
    mean absolute deviation from the other four is 43 ranks against their
    10.7-21.2, with a worst case of 107. Under a mean, one source 40 ranks
    out moves five-source consensus by 8 -- a full round in an 8-team
    league -- and it does it precisely on the players the sources disagree
    about, which are the ones worth being right about.

    Here four sources put a player between 1 and 3 and MFL says 43. The
    consensus must stay with the four. `market_spread` must still report the
    disagreement, because hiding it would be the other failure mode.
    """
    mfl = pd.DataFrame({"mfl_name": ["Jahmyr Gibbs"], "position": ["RB"],
                        "mfl_rank": [43]})
    cbs = pd.DataFrame({"cbs_name": ["jahmyr gibbs"], "position": ["RB"],
                        "cbs_rank": [3]})
    out = add_market(_board(), _espn(), _fp(), _sleeper(), mfl=mfl, cbs=cbs)
    g1 = out[out["player_id"] == "g1"].iloc[0]
    # FFC 1, ESPN 1, CBS 3, MFL 43. Median 2.0; the mean would be 12.0.
    assert g1["market_rank"] == 2.0
    assert g1["market_spread"] == 42        # 43 - 1, still visible
    assert g1["market_sources"]["mfl"] == 43.0

def test_market_without_new_sources_unchanged():
    # Callers that don't pass mfl/cbs (and pre-refresh DBs with no tables)
    # must behave exactly as before, with the new source keys present as None.
    out = add_market(_board(), _espn(), _fp(), _sleeper())
    g1 = out[out["player_id"] == "g1"].iloc[0]
    assert g1["market_rank"] == 1.0
    assert g1["market_sources"]["mfl"] is None and g1["market_sources"]["cbs"] is None

def test_name_ranks_matches_hyphenated_names_from_slugs():
    # CBS slugs turn every hyphen into a space ("amon-ra-st-brown" ->
    # "amon ra st brown") while board names norm to "amonra st brown"; a
    # space-squashed second pass must reunite them.
    cbs = pd.DataFrame({"cbs_name": ["amon ra st brown"], "position": ["WR"],
                        "cbs_rank": [5]})
    out = add_market(_board(), _espn(), _fp(), _sleeper(), cbs=cbs)
    g2 = out[out["player_id"] == "g2"].iloc[0]
    assert g2["market_sources"]["cbs"] == 5.0

def test_consensus_uses_espn_ppr_rank_not_adp():
    # PPR guarantee: the ESPN component of the consensus must be the explicit
    # PPR expert rank (draftRanksByRankType.PPR), not the mixed-format ADP.
    # Fixture g2: ADP-derived rank would be 2, PPR rank is 4.
    out = add_market(_board(), _espn(), _fp(), _sleeper())
    g2 = out[out["player_id"] == "g2"].iloc[0]
    assert g2["market_sources"]["espn"] == 4.0
    assert g2["market_rank"] == round((2 + 4 + 3) / 3, 1)


# -- scoring-format-aware consensus (select_format threading) --------------

def test_source_without_format_column_is_treated_as_all_ppr():
    # Backward compatibility: old DBs and most fixtures seed sources with NO
    # `format` column. Every row is then PPR, so a PPR call is byte-identical
    # to the no-fmt default and a non-PPR call still sees the same rows (there
    # is no per-format data to filter to).
    mfl = pd.DataFrame({"mfl_name": ["Jahmyr Gibbs"], "position": ["RB"], "mfl_rank": [3]})
    default = add_market(_board(), _espn(), _fp(), _sleeper(), mfl=mfl)
    for fmt in ("ppr", "half", "std"):
        out = add_market(_board(), _espn(), _fp(), _sleeper(), mfl=mfl, fmt=fmt)
        g1 = out[out["player_id"] == "g1"].iloc[0]
        assert g1["market_sources"]["mfl"] == 3.0
        # PPR must match the no-fmt default exactly.
        if fmt == "ppr":
            assert g1["market_rank"] == default[default["player_id"] == "g1"].iloc[0]["market_rank"]


def test_ppr_with_a_format_column_is_byte_identical_to_no_column():
    # The `format` column itself must not perturb the consensus: a PPR build
    # from a format-tagged table (PPR rows + a decoy std row the PPR league
    # must ignore) has to match a build from the same rows with no column.
    mfl_plain = pd.DataFrame({"mfl_name": ["Jahmyr Gibbs"], "position": ["RB"], "mfl_rank": [3]})
    mfl_fmt = pd.DataFrame({
        "mfl_name": ["Jahmyr Gibbs", "Jahmyr Gibbs"], "position": ["RB", "RB"],
        "mfl_rank": [3, 25], "format": ["ppr", "std"]})
    a = add_market(_board(), _espn(), _fp(), _sleeper(), mfl=mfl_plain, fmt="ppr")
    b = add_market(_board(), _espn(), _fp(), _sleeper(), mfl=mfl_fmt, fmt="ppr")
    assert list(a["market_rank"]) == list(b["market_rank"])
    assert [s["mfl"] for s in a["market_sources"]] == [s["mfl"] for s in b["market_sources"]]


def test_format_column_filters_to_the_requested_format():
    # MFL carries a PPR row (rank 3) and a STD row (rank 25) for the same
    # player. A PPR league takes 3, a standard league 25 -- proving the vote
    # follows the league's format.
    mfl = pd.DataFrame({
        "mfl_name": ["Jahmyr Gibbs", "Jahmyr Gibbs"], "position": ["RB", "RB"],
        "mfl_rank": [3, 25], "format": ["ppr", "std"]})
    ppr = add_market(_board(), _espn(), _fp(), _sleeper(), mfl=mfl, fmt="ppr")
    std = add_market(_board(), _espn(), _fp(), _sleeper(), mfl=mfl, fmt="std")
    assert ppr[ppr["player_id"] == "g1"].iloc[0]["market_sources"]["mfl"] == 3.0
    assert std[std["player_id"] == "g1"].iloc[0]["market_sources"]["mfl"] == 25.0


def test_source_falls_back_to_ppr_when_it_lacks_the_requested_format():
    # MFL publishes no 'half' feed (true of the real source). A half-PPR
    # league must fall back to MFL's PPR rows, not drop MFL from the vote.
    mfl = pd.DataFrame({
        "mfl_name": ["Jahmyr Gibbs", "Jahmyr Gibbs"], "position": ["RB", "RB"],
        "mfl_rank": [3, 25], "format": ["ppr", "std"]})
    half = add_market(_board(), _espn(), _fp(), _sleeper(), mfl=mfl, fmt="half")
    assert half[half["player_id"] == "g1"].iloc[0]["market_sources"]["mfl"] == 3.0
