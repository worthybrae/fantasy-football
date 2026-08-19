from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from pipeline.db import get_conn, read_table, write_table
from scoring import league
from scoring.draft_sim import (run_sim, DEFAULT_AVAILABILITY, FLEX_POSITIONS, POSITION_FLOOR,
                               best_lineup_points, build_pool, projections,
                               roster_value)

S = league.default_settings()   # QB/2RB/2WR/TE/2FLEX/K/DST, 5 bench


def test_best_lineup_fills_dedicated_slots_then_flex():
    roster = [("QB", 300.0), ("RB", 250.0), ("RB", 200.0), ("RB", 180.0),
              ("WR", 240.0), ("WR", 220.0), ("WR", 190.0),
              ("TE", 150.0), ("K", 120.0), ("DST", 110.0)]
    # Starters: QB 300, RB 250+200, WR 240+220, TE 150, K 120, DST 110 = 1590
    # FLEX x2 take the best leftovers: RB 180 and WR 190 = 370
    assert best_lineup_points(roster, S) == 1960.0


def test_best_lineup_ignores_surplus_beyond_flex():
    roster = [("WR", 100.0)] * 10
    # 2 WR starters + 2 FLEX = 4 slots filled, the other six are bench.
    assert best_lineup_points(roster, S) == 400.0


def test_best_lineup_handles_an_unfilled_slot():
    assert best_lineup_points([("QB", 300.0)], S) == 300.0


def test_best_lineup_and_roster_value_handle_an_empty_roster():
    # A valuation primitive later tasks build on -- an empty roster (before
    # any picks) must not raise or return NaN, it must return 0.0.
    assert best_lineup_points([], S) == 0.0
    assert roster_value([], S) == 0.0


def test_best_lineup_matches_brute_force_on_an_adversarial_roster():
    """Cross-check the greedy optimizer against a true brute-force search.

    Property 2 of the task brief: greedy is only optimal here because FLEX
    accepts a superset of no dedicated slot's eligibility and every other
    slot is single-position. This roster is deliberately adversarial (a lone
    elite TE that should win a FLEX slot over lesser dedicated-eligible
    players, uneven position depth, surplus K/DST) to stress that claim
    rather than take it on faith.
    """
    roster = [
        ("QB", 100.0),
        ("RB", 50.0), ("RB", 45.0), ("RB", 40.0), ("RB", 10.0),
        ("WR", 48.0), ("WR", 44.0), ("WR", 12.0), ("WR", 8.0),
        ("TE", 90.0), ("TE", 5.0), ("TE", 3.0),
        ("K", 20.0), ("K", 15.0),
        ("DST", 25.0), ("DST", 5.0),
    ]

    def brute_force(roster, settings):
        by_pos = {}
        for pos, pts in roster:
            by_pos.setdefault(pos, []).append(pts)

        total = 0.0
        for pos, count in settings.starters.items():
            if pos not in FLEX_POSITIONS:
                vals = sorted(by_pos.get(pos, []), reverse=True)
                total += sum(vals[:count])

        flex_pool = []
        needs = {}
        for pos in FLEX_POSITIONS:
            for v in by_pos.get(pos, []):
                flex_pool.append((pos, v))
            needs[pos] = settings.starters.get(pos, 0)

        slots_needed = min(sum(needs.values()) + settings.flex_slots, len(flex_pool))
        best = 0.0
        for combo in combinations(range(len(flex_pool)), slots_needed):
            counts, s = {}, 0.0
            for i in combo:
                p, v = flex_pool[i]
                counts[p] = counts.get(p, 0) + 1
                s += v
            if all(counts.get(p, 0) >= needs[p] for p in FLEX_POSITIONS):
                best = max(best, s)
        return total + best

    assert best_lineup_points(roster, S) == brute_force(roster, S)


def test_roster_value_adds_backup_insurance():
    healthy = [("QB", 300.0, 100.0)]
    starter_only = roster_value(healthy, S)
    with_backup = roster_value(healthy + [("QB", 200.0, 100.0)], S)
    # A backup behind a perfectly durable starter adds nothing.
    assert with_backup == starter_only


def test_roster_value_backup_matters_more_behind_a_fragile_starter():
    fragile = [("RB", 300.0, 0.0)]
    solo = roster_value(fragile, S)
    covered = roster_value(fragile + [("RB", 200.0, 100.0)], S)
    assert covered > solo


def test_roster_value_insurance_credit_is_the_backups_own_rate_not_the_dropoff():
    """Pins down which of two plausible formulas roster_value implements.

    Neither brief-given test above distinguishes "credit = missed-share x
    backup's own points" from "credit = missed-share x (starter - backup)"
    (a literal drop-off/difference): the durable-starter case has missed=0
    either way, and the fragile-RB case has the "backup" fill a genuine
    second starting slot rather than sit as insurance. With a QB (a single,
    non-FLEX slot) at 50% durability plus one worse backup, the two formulas
    give different answers (350 vs 400) -- this test requires the former,
    matching the task brief's property that a backup is worth at most what
    he actually provides.
    """
    roster = [("QB", 300.0, 50.0), ("QB", 100.0, 100.0)]
    # starters_only lineup = 300 (QB has 1 slot, not FLEX-eligible).
    # missed = 1 - 50/100 = 0.5; credit = 0.5 * min(100, 300) = 50.
    assert roster_value(roster, S) == 350.0


def test_roster_value_backup_credit_uses_full_missed_share_at_zero_durability():
    """durability=0.0 must mean "expected to miss every game" (missed=1.0).

    0.0 is a legitimate, real value on the durability scale (a player who
    played zero of his possible games), and Python's `x or default` idiom
    treats 0.0 as falsy -- silently substituting a default and halving the
    insurance credit. This roster isolates that line: the backup only
    contributes as insurance (QB has one slot, so the second QB never
    becomes a starter), so the total is entirely determined by how `missed`
    is computed at durability=0.
    """
    roster = [("QB", 300.0, 0.0), ("QB", 100.0, 100.0)]
    # starters_only lineup = 300; missed = 1 - 0/100 = 1.0; credit = 1.0 * 100.
    assert roster_value(roster, S) == 400.0


def test_roster_value_ignores_bench_depth_beyond_the_first_backup():
    two_deep = [("QB", 300.0, 50.0), ("QB", 200.0, 100.0)]
    three_deep = two_deep + [("QB", 50.0, 100.0)]
    assert roster_value(three_deep, S) == roster_value(two_deep, S)


def test_roster_value_credits_a_shared_backup_once_not_per_starter():
    """A multi-slot position's single backup must be credited once per
    roster, not once per starter he sits behind.

    RB has 2 starting slots under default_settings. WR filler is included
    and deliberately outranks the RB backup (240/230 > 150) so it -- not the
    RB backup -- claims both FLEX slots; this isolates the RB insurance term
    from best_lineup_points's own FLEX logic, so the only thing left to
    explain the total is the insurance loop itself.

    Without the fix, a backup worth 150 for the season gets counted twice
    (once behind each zero-durability starter) for 300 of "insurance" --
    twice what one human being can actually provide. With the fix, his
    missed-share credit is summed across the two starters (1.0 + 1.0,
    clamped to 1.0) and applied to his 150 points once.
    """
    roster = [
        ("RB", 300.0, 0.0), ("RB", 280.0, 0.0), ("RB", 150.0, 100.0),
        ("WR", 260.0, 100.0), ("WR", 250.0, 100.0),
        ("WR", 240.0, 100.0), ("WR", 230.0, 100.0),
    ]
    # starters_only: RB 300+280=580, WR 260+250=510, FLEX takes WR 240+230=470
    #   (the RB backup, 150, loses the FLEX competition to the WR leftovers)
    #   => 1560
    # RB insurance: missed_total = min(1.0, 1.0 + 1.0) = 1.0; credit = 1.0 * 150 = 150
    # WR insurance: durability 100 everywhere => missed_total = 0; credit = 0
    assert roster_value(roster, S) == 1710.0


def test_roster_value_position_insurance_never_exceeds_the_backups_own_points():
    backup_points = 150.0
    wr_filler = [("WR", 260.0, 100.0), ("WR", 250.0, 100.0),
                 ("WR", 240.0, 100.0), ("WR", 230.0, 100.0)]
    for d1, d2 in [(0.0, 0.0), (0.0, 100.0), (50.0, 50.0), (100.0, 100.0), (30.0, 70.0)]:
        without_backup = [("RB", 300.0, d1), ("RB", 280.0, d2)] + wr_filler
        with_backup = without_backup + [("RB", backup_points, 100.0)]
        # WR filler is identical and its own insurance term is 0 either way
        # (all WR durability=100), so this difference isolates exactly the
        # RB position's insurance credit.
        insurance = roster_value(with_backup, S) - roster_value(without_backup, S)
        assert 0.0 <= insurance <= backup_points + 1e-9


_ZERO_BENCH_ROSTER = [
    ("QB", 300.0, 50.0),
    ("RB", 250.0, 50.0), ("RB", 200.0, 50.0), ("RB", 100.0, 50.0),
    ("WR", 240.0, 50.0), ("WR", 220.0, 50.0), ("WR", 90.0, 50.0),
    ("TE", 150.0, 50.0), ("K", 120.0, 50.0), ("DST", 110.0, 50.0),
]


def test_roster_value_does_not_credit_a_flex_starter_as_bench_insurance():
    """A roster that exactly fills every starting slot has no bench at all,
    so it can have no insurance: roster_value must equal best_lineup_points.

    Ten players, ten slots (QB / 2 RB / 2 WR / TE / K / DST / 2 FLEX). RB3
    (100) and WR3 (90) are the two FLEX starters -- best_lineup_points has
    already counted both. The insurance loop used to index `values[count]`,
    i.e. the player just past the *dedicated* count, which at RB and WR is
    exactly that FLEX starter, so each of them got paid twice: 1780 became
    1970. The backup has to be the best player the lineup did NOT assign.
    """
    starters_only = [(pos, points) for pos, points, _ in _ZERO_BENCH_ROSTER]
    # QB 300 + RB 250+200 + WR 240+220 + TE 150 + K 120 + DST 110 = 1590,
    # plus FLEX taking the two leftovers RB 100 and WR 90 = 1780.
    assert best_lineup_points(starters_only, S) == 1780.0
    assert roster_value(_ZERO_BENCH_ROSTER, S) == 1780.0


def test_roster_value_insurance_uses_the_first_unassigned_player_at_the_position():
    """The other half of the same rule: a real bench player still pays.

    Same zero-bench roster plus a genuine RB4 (60) who no slot assigns. RB's
    two dedicated starters are at 50% availability, so missed_total clamps to
    1.0 and the credit is his own full 60 -- applied to him, not to RB3, who
    is starting at FLEX.
    """
    with_bench = _ZERO_BENCH_ROSTER + [("RB", 60.0, 100.0)]
    assert roster_value(with_bench, S) == 1840.0


def test_build_pool_availability_is_a_games_played_rate_not_a_percentile(tmp_path):
    """SimPool's availability must be `games / possible`, not the board's
    within-position `durability` percentile.

    board["durability"] comes out of factors.normalize_within_position, i.e.
    rank(pct=True) * 100. Feeding that into roster_value's
    `missed = 1 - availability / 100` models the median player at every
    position as missing half the season, and makes any two starters at a
    multi-slot position sum past the min(1.0, ...) clamp. The rate the
    insurance term actually wants is durability_factor's `durability_raw`,
    which the board drops before _BOARD_COLUMNS.
    """
    conn = get_conn(str(tmp_path / "t.duckdb"))
    write_table(conn, "weekly", pd.DataFrame([
        {"player_id": "p1", "player_display_name": "Fifteen Games",
         "position": "WR", "recent_team": "DET", "season": 2025, "week": w,
         "receptions": 5, "receiving_yards": 60, "targets": 7, "carries": 0}
        for w in range(1, 16)]))
    board = pd.DataFrame([
        # durability 99.0 is a percentile; his real rate is 15/17 = 88.2%.
        {"player_id": "p1", "name": "Fifteen Games", "position": "WR",
         "team": "DET", "market_rank": 1.0, "durability": 99.0, "stats": None},
        # No weekly history at all (a K, a DST, or a rookie).
        {"player_id": "p2", "name": "No History", "position": "K",
         "team": "SF", "market_rank": 2.0, "durability": 50.0, "stats": None},
    ])
    pool = build_pool(conn, board, S)
    by_id = dict(zip(pool.player_id, pool.availability))
    assert by_id["p1"] == pytest.approx(15 / 17 * 100.0)
    assert by_id["p2"] == DEFAULT_AVAILABILITY


def test_build_pool_survives_a_duplicate_player_id_on_the_board(tmp_path):
    """Two board rows sharing a `player_id` must not take build_pool down.

    `_add_adp_only_players` synthesizes `player_id = "adp_" + norm` from the
    normalized name alone, with no position in the key, so one name at two
    positions in the ADP feed (neither matching the weekly universe) reaches
    the board twice under one id -- the collision build_board's own
    drop_duplicates comment documents and tests/test_api.py's
    `_seed_adp_only` fixture already produces.

    build_board was hardened against this (`uni["proj_points"] =
    proj.to_numpy(...)`); build_pool, three lines down the same call chain,
    still did `board.set_index("player_id")` and `.map()`, which raises
    `InvalidIndexError: Reindexing only valid with uniquely valued Index
    objects`. api/live.py's build_session calls both in that order, so an
    ADP-feed name collision meant no live draft could start at all.

    Both branches covered: `proj_points` present (the production path, where
    the board already carries the column) and absent (a bare fixture board,
    where projections() computes it).
    """
    conn = get_conn(str(tmp_path / "t.duckdb"))
    write_table(conn, "weekly", pd.DataFrame(columns=[
        "player_id", "player_display_name", "position", "recent_team",
        "season", "week", "receptions", "receiving_yards", "targets",
        "carries"]))
    board = pd.DataFrame([
        {"player_id": "adp_dup_guy", "name": "Dup Guy", "position": "RB",
         "team": "SF", "market_rank": 1.0, "durability": 50.0,
         "stats": {"ppg": 12.0}, "proj_points": 180.0},
        {"player_id": "adp_dup_guy", "name": "Dup Guy", "position": "TE",
         "team": "GB", "market_rank": 2.0, "durability": 50.0,
         "stats": {"ppg": 8.0}, "proj_points": 120.0},
    ])

    pool = build_pool(conn, board, S)
    # Each colliding row keeps its OWN projection -- assigned positionally,
    # so neither row inherits the other's number.
    assert list(pool.points) == [180.0, 120.0]
    assert list(pool.position) == ["RB", "TE"]

    bare = build_pool(conn, board.drop(columns=["proj_points"]), S)
    # projections() falls through to stats.ppg * GAMES for both rows.
    assert len(bare.points) == 2
    assert bare.points[0] > bare.points[1]


def test_projections_prefer_espn_then_fall_back_to_weighted_ppg(tmp_path):
    conn = get_conn(str(tmp_path / "t.duckdb"))
    write_table(conn, "espn_adp", pd.DataFrame([
        {"espn_id": 1, "espn_name": "Has Projection", "position": "WR",
         "espn_adp": 5.0, "espn_ppr_rank": 3, "espn_proj": 289.0}]))
    board = pd.DataFrame([
        {"player_id": "p1", "name": "Has Projection", "position": "WR",
         "team": "DET", "stats": {"ppg": 15.0}},
        {"player_id": "p2", "name": "No Projection", "position": "WR",
         "team": "GB", "stats": {"ppg": 10.0}},
        {"player_id": "p3", "name": "Nothing At All", "position": "K",
         "team": "GB", "stats": None},
    ])
    proj = projections(conn, board)
    assert proj["p1"] == 289.0
    assert proj["p2"] == 170.0                 # 10.0 ppg x 17
    assert proj["p3"] > 0                       # position floor, never NaN
    assert not proj.isna().any()


def test_projections_join_defenses_by_team_not_by_name(tmp_path):
    """Every DST used to fall through to POSITION_FLOOR.

    ESPN calls a defense "Ravens D/ST"; the board names it from the ADP
    feed's nickname ("Ravens"), so the (normalized name, position) join could
    never hit and the whole position was scored at a constant floor. A team
    has exactly one defense, so team is the join -- the same rule
    `board._merge_adp` already uses for DST ADP.
    """
    conn = get_conn(str(tmp_path / "t.duckdb"))
    write_table(conn, "espn_adp", pd.DataFrame([
        {"espn_id": 16018, "espn_name": "Ravens D/ST", "position": "DST",
         "team": "BAL", "espn_adp": 140.0, "espn_ppr_rank": 150,
         "espn_proj": 132.0}]))
    board = pd.DataFrame([
        {"player_id": "adp_ravens", "name": "Ravens", "position": "DST",
         "team": "BAL", "stats": None}])
    proj = projections(conn, board)
    assert proj["adp_ravens"] == 132.0
    assert proj["adp_ravens"] != POSITION_FLOOR["DST"]


def test_projections_ignores_non_positive_espn_projection(tmp_path):
    conn = get_conn(str(tmp_path / "t.duckdb"))
    write_table(conn, "espn_adp", pd.DataFrame([
        {"espn_id": 2, "espn_name": "Zero Projection", "position": "RB",
         "espn_adp": 50.0, "espn_ppr_rank": 80, "espn_proj": 0.0}]))
    board = pd.DataFrame([
        {"player_id": "p1", "name": "Zero Projection", "position": "RB",
         "team": "DAL", "stats": {"ppg": 8.0}},
    ])
    proj = projections(conn, board)
    assert proj["p1"] == 136.0                 # espn_proj <= 0 is not used; falls to ppg x 17


def test_projections_zero_ppg_falls_back_to_position_floor(tmp_path):
    # `ppg` of exactly 0.0 is falsy in `float(ppg) * GAMES if ppg else None`,
    # so it is treated the same as "no stat history" and sent to the
    # position floor rather than projected as a literal 0.0. Pinning this so
    # it reads as a deliberate choice (a real player already has a game or
    # two of non-zero history well before his season ppg could be exactly
    # 0.0) rather than an accident discovered later.
    conn = get_conn(str(tmp_path / "t.duckdb"))
    board = pd.DataFrame([
        {"player_id": "p1", "name": "Zero Ppg", "position": "RB",
         "team": "DAL", "stats": {"ppg": 0.0}},
    ])
    proj = projections(conn, board)
    assert proj["p1"] == POSITION_FLOOR["RB"]


def test_build_pool_ranks_on_ffc_not_the_consensus(tmp_path):
    """The simulator and the fit must order players by the same source.

    `draft_model._enrich_pool` fits `reach`/`fall` against the FFC
    `adp_rank` (see its docstring for why FFC and not ESPN). If `build_pool`
    ordered the live pool by `board["market_rank"]` -- the five-source
    consensus -- a coefficient learned on FFC's ordering would be applied to
    a board that disagrees with it, which is the residual Task 4 logged and
    Task 9 closed. Here the two orderings are deliberately reversed against
    each other, so ranking on the wrong one fails.
    """
    conn = get_conn(str(tmp_path / "ffc.duckdb"))
    board = pd.DataFrame([
        {"player_id": "p1", "name": "Ffc First", "position": "RB",
         "team": "DET", "ffc_rank": 1.0, "market_rank": 50.0,
         "durability": 90.0, "stats": None},
        {"player_id": "p2", "name": "Ffc Second", "position": "WR",
         "team": "DAL", "ffc_rank": 2.0, "market_rank": 20.0,
         "durability": 90.0, "stats": None},
        {"player_id": "p3", "name": "Ffc Third", "position": "TE",
         "team": "GB", "ffc_rank": 3.0, "market_rank": 1.0,
         "durability": 90.0, "stats": None},
    ])
    pool = build_pool(conn, board, S)
    ranks = dict(zip(pool.player_id, pool.market_rank))
    assert ranks == {"p1": 1.0, "p2": 2.0, "p3": 3.0}


def test_build_pool_ranks_on_the_cheat_sheet_ahead_of_ffc(tmp_path):
    """The same reference ladder `draft_model._enrich_pool` fits on: ESPN's
    preseason cheat sheet first, FFC only for what it misses.

    The two are reversed against each other here, and the third player is
    absent from the sheet entirely, so ranking on FFC alone or dropping the
    unranked player both fail.
    """
    from scoring.config import CURRENT_SEASON
    conn = get_conn(str(tmp_path / "cs.duckdb"))
    write_table(conn, "historic_espn_cs", pd.DataFrame([
        {"season": CURRENT_SEASON, "cs_rank": 1, "position": "WR",
         "cs_name": "Sheet First", "team": "DAL", "auction_value": 55.0},
        {"season": CURRENT_SEASON, "cs_rank": 2, "position": "RB",
         "cs_name": "Sheet Second", "team": "DET", "auction_value": 50.0},
    ]))
    board = pd.DataFrame([
        {"player_id": "p1", "name": "Sheet Second", "position": "RB",
         "team": "DET", "ffc_rank": 1.0, "market_rank": 1.0,
         "durability": 90.0, "stats": None},
        {"player_id": "p2", "name": "Sheet First", "position": "WR",
         "team": "DAL", "ffc_rank": 2.0, "market_rank": 2.0,
         "durability": 90.0, "stats": None},
        {"player_id": "p3", "name": "Not On Sheet", "position": "TE",
         "team": "GB", "ffc_rank": 3.0, "market_rank": 3.0,
         "durability": 90.0, "stats": None},
    ])
    pool = build_pool(conn, board, S)
    assert dict(zip(pool.player_id, pool.market_rank)) == {
        "p2": 1.0, "p1": 2.0, "p3": 3.0}


def test_build_pool_blends_ffc_into_the_cheat_sheet_order(tmp_path):
    """The simulator's board must be the same blend `_enrich_pool` fits on.

    The cheat sheet leads at 3:1, so a small FFC disagreement changes
    nothing -- that is what the test above pins. This one pins the other
    half: FFC is genuinely in the mix, and a player the market likes far
    more than the sheet does moves up.

    This is the Chase Brown case from 2026, where ESPN's sheet says 21st and
    5,789 real PPR mock drafts say 12.2. `c` sits third on the sheet and
    first on FFC; at w=0.25 that is worth 0.75*3 + 0.25*1 = 2.50 against
    `b`'s 0.75*2 + 0.25*5 = 2.75, so `c` passes `b` and nothing else moves.
    Under cheat-sheet-only ordering they stay a, b, c, d, e.

    If this ever fails after a change to `FFC_BLEND_WEIGHT`, the fix is to
    re-derive the arithmetic above -- not to loosen the assertion. The fit
    and the simulator reading the same board is the invariant; a reach
    coefficient learned on one board means something else on another.
    """
    from scoring.config import CURRENT_SEASON
    conn = get_conn(str(tmp_path / "blend.duckdb"))
    sheet = ["a", "b", "c", "d", "e"]                      # cheat-sheet order
    ffc = {"a": 2.0, "b": 5.0, "c": 1.0, "d": 3.0, "e": 4.0}
    write_table(conn, "historic_espn_cs", pd.DataFrame([
        {"season": CURRENT_SEASON, "cs_rank": i + 1, "position": "RB",
         "cs_name": f"Player {n.upper()}", "team": "DET", "auction_value": 10.0}
        for i, n in enumerate(sheet)]))
    board = pd.DataFrame([
        {"player_id": n, "name": f"Player {n.upper()}", "position": "RB",
         "team": "DET", "ffc_rank": ffc[n], "market_rank": ffc[n],
         "durability": 90.0, "stats": None}
        for n in sheet])
    pool = build_pool(conn, board, S)
    assert list(pool.player_id) == ["a", "c", "b", "d", "e"]
    assert list(pool.market_rank) == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_build_pool_drafts_the_current_season_not_the_rules_season(tmp_path):
    """`settings.season` is the season the league's RULES were imported from
    (the newest completed draft), not the season being drafted.

    Reading it as the draft season ranked a 2026 board on the 2025 cheat
    sheet and read player attributes a year stale. The fixture gives the two
    seasons contradictory sheets, so honouring `settings.season` fails.
    """
    from scoring.config import CURRENT_SEASON
    from dataclasses import replace
    stale = CURRENT_SEASON - 1
    conn = get_conn(str(tmp_path / "season.duckdb"))
    write_table(conn, "historic_espn_cs", pd.DataFrame([
        # Last season's sheet ranks them one way...
        {"season": stale, "cs_rank": 1, "position": "RB",
         "cs_name": "Last Year Guy", "team": "DET", "auction_value": 50.0},
        {"season": stale, "cs_rank": 2, "position": "WR",
         "cs_name": "This Year Guy", "team": "DAL", "auction_value": 40.0},
        # ...this season's reverses it.
        {"season": CURRENT_SEASON, "cs_rank": 1, "position": "WR",
         "cs_name": "This Year Guy", "team": "DAL", "auction_value": 55.0},
        {"season": CURRENT_SEASON, "cs_rank": 2, "position": "RB",
         "cs_name": "Last Year Guy", "team": "DET", "auction_value": 45.0},
    ]))
    board = pd.DataFrame([
        {"player_id": "p1", "name": "Last Year Guy", "position": "RB",
         "team": "DET", "ffc_rank": 1.0, "market_rank": 1.0,
         "durability": 90.0, "stats": None},
        {"player_id": "p2", "name": "This Year Guy", "position": "WR",
         "team": "DAL", "ffc_rank": 2.0, "market_rank": 2.0,
         "durability": 90.0, "stats": None},
    ])
    settings = replace(S, season=stale)
    pool = build_pool(conn, board, settings)
    assert dict(zip(pool.player_id, pool.market_rank)) == {"p2": 1.0, "p1": 2.0}


def test_build_pool_falls_back_to_the_consensus_without_ffc_rank(tmp_path):
    """A board with no `ffc_rank` column (a bare fixture) still ranks."""
    conn = get_conn(str(tmp_path / "noffc.duckdb"))
    board = pd.DataFrame([
        {"player_id": "p1", "name": "Second", "position": "RB",
         "team": "DET", "market_rank": 9.0, "durability": 90.0, "stats": None},
        {"player_id": "p2", "name": "First", "position": "WR",
         "team": "DAL", "market_rank": 4.0, "durability": 90.0, "stats": None},
    ])
    pool = build_pool(conn, board, S)
    assert dict(zip(pool.player_id, pool.market_rank)) == {"p2": 1.0, "p1": 2.0}


def test_build_pool_ranks_market_known_players_before_unranked_ones(tmp_path):
    """adp_rank must land on the scale draft_model's reach/fall coefficients
    were fitted on: historic_adp.adp_rank is a dense rank over the players
    ONE season's ADP source actually ranked, not over `board`'s broader union
    of every player with a stat line plus ADP-only rookies and K/DST.

    build_pool used to fold a `market_rank.fillna(len(board) + 1)` sentinel
    into the same `sort_values` as the real market_rank values. That is not
    safe: fp_rank/mfl_rank/cbs_rank are raw external ranks, not re-ranked
    within `board`, so a real market_rank can exceed len(board) once board is
    a filtered subset of what those sources rank -- and a sentinel of
    len(board) + 1 can then be SMALLER than that real value, sorting a
    genuinely-ranked player behind the unranked ones. This pins the required
    fix: rank densely among only the players carrying a real market_rank
    (1..k, in market order), then continue the sequence for the rest (k+1..)
    rather than interleaving them.

    `SimPool.market_rank` must land on that exact same dense scale, not
    `board["market_rank"]`'s raw value: a prior version of this fix filled
    only the unranked players' NaN with their dense pool position and left
    the four genuinely-ranked players at their raw board["market_rank"]
    (5.0, 20.0, 500.0 for p2/p1/p3) -- p3's real 500 stayed 500, but p4
    (truly unranked) landed at a dense-position fallback of 4.0, BETTER
    than p2's real, dense-rank-1 market_rank of 5.0. A player nobody ranked
    read as less of a reach than the market's actual number one. Asserting
    `market_rank` here, not just `adp_rank`, is what catches that: both
    fields must be identical, since both must be the dense rank.
    """
    conn = get_conn(str(tmp_path / "t.duckdb"))
    board = pd.DataFrame([
        {"player_id": "p1", "name": "Second By Market", "position": "WR",
         "team": "DET", "market_rank": 20.0, "durability": 90.0, "stats": None},
        {"player_id": "p2", "name": "First By Market", "position": "RB",
         "team": "DAL", "market_rank": 5.0, "durability": 90.0, "stats": None},
        # A real market_rank that is numerically larger than len(board) --
        # exactly the case a fillna(len(board) + 1) sentinel gets wrong.
        {"player_id": "p3", "name": "Large Real Rank", "position": "WR",
         "team": "GB", "market_rank": 500.0, "durability": 90.0, "stats": None},
        {"player_id": "p4", "name": "Unranked One", "position": "TE",
         "team": "GB", "market_rank": None, "durability": 90.0, "stats": None},
        {"player_id": "p5", "name": "Unranked Two", "position": "K",
         "team": "SF", "market_rank": None, "durability": 90.0, "stats": None},
    ])
    pool = build_pool(conn, board, S)
    for ranks in (dict(zip(pool.player_id, pool.adp_rank)),
                 dict(zip(pool.player_id, pool.market_rank))):
        # Known market_rank players occupy dense ranks 1..3, in market
        # order -- unaffected by p3's real rank value (500.0) exceeding
        # len(board) (5).
        assert ranks["p2"] == 1.0     # market_rank 5.0
        assert ranks["p1"] == 2.0     # market_rank 20.0
        assert ranks["p3"] == 3.0     # market_rank 500.0 -- still ranked 3rd
        # Unranked players continue the sequence after the known ones, not
        # interleaved with them -- and, critically, never ahead of a real
        # rank: p4/p5 must land at 4.0/5.0, not at some value below p2's
        # real 1.0-equivalent (5.0 raw) that would make them read as a
        # smaller reach than the market's actual number one.
        assert sorted([ranks["p4"], ranks["p5"]]) == [4.0, 5.0]
    # The two fields must be identical, not just individually correct --
    # `reach`/`fall` (_log_rank_features) read `market_rank`, and every
    # other market-order consumer (_candidate_indices) reads `adp_rank`;
    # a real bug that only broke one of them (e.g. the fillna reintroduced
    # for just one field) must fail here even if that one field's own
    # assertions above happen to still pass.
    np.testing.assert_array_equal(pool.adp_rank, pool.market_rank)


from scoring.draft_model import FEATURE_NAMES

# run_sim now refuses an all-zero pooled vector: with cold-start returning a
# real market prior, "pooled is zeros" can only mean a degenerate build, which
# would draft every opponent uniformly. These smoke tests never cared about the
# pooled *values* -- only that a fit exists -- so they use a small non-zero
# stand-in rather than zeros, which the guard (correctly) rejects.
_POOLED = np.full(len(FEATURE_NAMES), 0.05)
from scoring.draft_sim import SimPool, _run_draft, rollout, snake_slots


def test_picks_until_my_next_turn_counts_the_teams_in_between():
    """The gap prices every one of my picks, so an off-by-one here shifts
    the whole policy quietly rather than failing. Counted on a real snake:
    8 teams, 3 rounds, slot 4 picks at offsets 3, 12 and 19."""
    from scoring.draft_sim import _picks_until_my_next_turn, snake_slots
    slots = snake_slots(8, 3)
    assert [i for i, s in enumerate(slots) if s == 4] == [3, 12, 19]
    assert _picks_until_my_next_turn(slots, 3, 4) == 8
    assert _picks_until_my_next_turn(slots, 12, 4) == 6
    # Last pick of the draft: nothing left to forgo, which is None rather
    # than 0 -- `_greedy_choice` treats them differently on purpose.
    assert _picks_until_my_next_turn(slots, 19, 4) is None


def test_picks_until_my_next_turn_is_zero_at_the_snake_turn():
    """At the turn I pick twice with nobody in between, so the gap is 0 and
    genuinely nothing can be taken from me. Distinct from None, which means
    no next pick at all."""
    from scoring.draft_sim import _picks_until_my_next_turn, snake_slots
    slots = snake_slots(8, 3)
    assert [i for i, s in enumerate(slots) if s == 1] == [0, 15, 16]
    assert _picks_until_my_next_turn(slots, 15, 1) == 0
    assert _picks_until_my_next_turn(slots, 0, 1) == 14


def test_snake_slots_reverses_every_other_round():
    assert snake_slots(4, 3) == [1, 2, 3, 4, 4, 3, 2, 1, 1, 2, 3, 4]


def _pool(n=60):
    positions = np.array(["RB", "WR", "QB", "TE", "K", "DST"] * (n // 6))
    return SimPool(
        player_id=np.array([f"p{i}" for i in range(n)]),
        norm=np.array([f"player {i}" for i in range(n)]),
        position=positions,
        adp_rank=np.arange(1, n + 1, dtype=float),
        points=np.linspace(300.0, 60.0, n),
        availability=np.full(n, 90.0),
        vor=np.linspace(300.0, 60.0, n),
        # Task 4 attribute columns: neutral/unknown for every player. Tests
        # using this fixture drive scoring with `_flat_betas` (all zeros) or
        # `_adp_betas` (reach only), so these values are never read for their
        # magnitude -- they only have to be finite and the right length.
        market_rank=np.arange(1, n + 1, dtype=float),
        age=np.full(n, np.nan),
        no_track_record=np.full(n, True), hype=np.full(n, np.nan),
        trend=np.zeros(n))


def _flat_betas(managers):
    return {m: np.zeros(len(FEATURE_NAMES)) for m in managers}


def test_rollout_is_deterministic_under_a_fixed_seed():
    pool = _pool()
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    args = (pool, S, slots, 4, taken, _flat_betas(slots.values()))
    a = rollout(*args, rng=np.random.default_rng(7))
    b = rollout(*args, rng=np.random.default_rng(7))
    assert a == b


def test_rollout_returns_a_positive_roster_value():
    pool = _pool()
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    value = rollout(pool, S, slots, 4, taken, _flat_betas(slots.values()),
                    rng=np.random.default_rng(1))
    assert value > 0


def test_forcing_the_top_player_beats_forcing_the_worst():
    pool = _pool()
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    betas = _flat_betas(slots.values())
    best = np.mean([rollout(pool, S, slots, 1, taken, betas,
                            rng=np.random.default_rng(i), forced=0)
                    for i in range(20)])
    worst = np.mean([rollout(pool, S, slots, 1, taken, betas,
                             rng=np.random.default_rng(i), forced=len(pool.points) - 1)
                     for i in range(20)])
    assert best > worst


def test_rollout_respects_already_taken_players():
    pool = _pool()
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    taken[:40] = True                    # only 20 players left, 8 teams x 15 rounds
    value = rollout(pool, S, slots, 1, taken, _flat_betas(slots.values()),
                    rng=np.random.default_rng(3), taken_order=list(range(40)))
    assert value > 0                     # runs out of players without crashing


def test_rollout_handles_taken_covering_the_whole_draft_without_crashing():
    """Locks in rollout's resume contract for the fully-drafted edge case.

    `taken`'s count of True values IS the number of picks already made
    (rollout's / _run_draft's docstring). If that count already meets or
    exceeds the total number of picks in the draft (teams x rounds), there
    is nothing left to simulate, and _run_draft returns explicitly rather
    than falling through a `slots[already:]` slice that happens to be empty.
    With no `taken_order`, _run_draft has no per-team ownership record for
    players marked `taken` before it was called -- only a pool-wide "off the
    board" mask -- so 0.0 is the correct, documented answer here, not a
    numeric regression. That unattributed call is now also the case
    _run_draft warns about, which this asserts alongside the value: the
    degenerate path stays safe, and it stays loud.
    """
    pool = _pool(150)                     # more players than teams x rounds (120)
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.ones(len(pool.player_id), dtype=bool)   # already >= len(slots)
    with pytest.warns(RuntimeWarning, match="taken_order"):
        value = rollout(pool, S, slots, 1, taken, _flat_betas(slots.values()),
                        rng=np.random.default_rng(2))
    assert value == 0.0


# --- Feature parity: _live_features (numpy, rollout-fast) must produce the
# same vectors as draft_model.feature_matrix (pandas, fitted-on). The fitted
# coefficients only mean anything against the same feature definitions they
# were fitted on -- a silent mismatch here would invalidate every simulation
# result while everything still runs and looks plausible. Not covered by the
# brief's own tests, so it is added here.

from scoring.draft_model import PickObservation, feature_matrix
from scoring.draft_sim import _live_features


def _parity_fixture():
    """Twelve players across all six positions, two already off the board
    (both WR, so `available` is a genuine subset of the pool, not the whole
    thing), a roster with mixed true/false needs (RB satisfied, WR/QB/TE/K/
    DST not), and a recent-run window that exactly fills RUN_WINDOW -- so
    `need` and `run` come out with a mix of zeros and non-zeros on both
    sides, rather than being trivially zero everywhere.

    The five Task 4 attribute columns are distinct per player (not a
    uniform neutral value) and set on the FULL twelve, taken players
    included, not just the ten `available` ones -- a `_live_features`
    column built from the unsliced `pool.<field>` instead of
    `pool.<field>[available]` would otherwise go undetected here, since a
    uniform or fully-available fixture can't tell "right value, wrong
    player" apart from "right value". RB/WR/QB/TE each repeat (2-3 players),
    so `age`'s per-position centring has a real, non-singleton group to
    centre within, not just the trivial "lone player is his own mean" case.
    """
    positions = np.array(["QB", "RB", "WR", "TE", "K", "DST",
                          "RB", "WR", "QB", "TE", "RB", "WR"])
    norms = np.array([f"player {i}" for i in range(len(positions))])
    adp_rank = np.arange(1, len(positions) + 1, dtype=float)
    taken = np.zeros(len(positions), dtype=bool)
    taken[[2, 7]] = True                      # both taken players are WR
    available = np.flatnonzero(~taken)

    n = len(positions)
    age = np.array([28.0, 24.0, 26.0, 29.0, 31.0, 33.0,
                    27.0, 23.0, 35.0, 30.0, 25.0, 22.0])
    no_track_record = np.array([False, False, True, False, True, False,
                                False, True, False, False, True, False])
    hype = np.array([-10.0, 5.0, 3.0, 12.0, -3.0, 7.0,
                     -8.0, 15.0, np.nan, -6.0, 9.0, -1.0])
    trend = np.array([1.2, -0.5, 0.0, 2.1, -1.8, 0.3,
                      -0.9, 1.5, 0.7, -0.2, 2.4, -1.1])

    pool = SimPool(
        player_id=np.array([f"p{i}" for i in range(n)]),
        norm=norms, position=positions, adp_rank=adp_rank,
        points=np.linspace(300.0, 60.0, n),
        availability=np.full(n, 90.0),
        vor=np.linspace(300.0, 60.0, n),
        market_rank=adp_rank, age=age, no_track_record=no_track_record,
        hype=hype, trend=trend)

    roster = {"RB": 2, "WR": 1, "QB": 1}       # RB need false, others true
    recent = ["WR", "RB", "RB", "QB", "TE"]    # fills RUN_WINDOW exactly

    obs_pool = pd.DataFrame({
        "norm": norms[available], "position": positions[available],
        "adp_rank": adp_rank[available], "market_rank": adp_rank[available],
        "age": age[available],
        "no_track_record": no_track_record[available],
        "hype": hype[available], "trend": trend[available]})

    return pool, available, roster, recent, obs_pool


def test_live_features_matches_feature_matrix_on_an_early_round_pick():
    pool, available, roster, recent, obs_pool = _parity_fixture()
    overall_pick = 5                            # round 1 of 8 teams: early
    obs = PickObservation(season=2024, overall_pick=overall_pick, manager="m",
                          chosen=0, pool=obs_pool, roster=roster, recent=recent)
    expected = feature_matrix(obs, S)
    actual = _live_features(pool, available, overall_pick, roster, recent, S)
    np.testing.assert_allclose(actual, expected)


def test_live_features_matches_feature_matrix_on_a_late_round_pick():
    pool, available, roster, recent, obs_pool = _parity_fixture()
    overall_pick = 50                           # round 7 of 8 teams: late
    obs = PickObservation(season=2024, overall_pick=overall_pick, manager="m",
                          chosen=0, pool=obs_pool, roster=roster, recent=recent)
    expected = feature_matrix(obs, S)
    actual = _live_features(pool, available, overall_pick, roster, recent, S)
    np.testing.assert_allclose(actual, expected)


# --- Roster caps (property 4): K and DST capped at 1, QB at most 3, imposed
# as a mask rather than learned. rollout()'s public interface only returns a
# float -- it does not expose per-manager roster composition -- so a direct
# unit test of the cap table itself pins down the table in isolation, and
# `_run_draft` (which returns the full per-slot roster state rollout() is
# built on, without changing rollout()'s own scalar return type) lets a
# second test check that a full simulated draft actually honours it.

from scoring.draft_sim import _roster_cap, _run_draft, survival


def test_roster_cap_limits_kicker_defense_and_qb_regardless_of_starters():
    caps = _roster_cap(S)
    assert caps["K"] == 1
    assert caps["DST"] == 1
    assert caps["QB"] <= 3


# The roster FLOOR, the counterpart of the cap above. Same argument as
# _roster_cap's own ("learned coefficients cannot express a hard ceiling, so
# it is imposed as a mask instead"), applied to the other end: with two picks
# left and an empty kicker and defense slot, a manager takes a kicker and a
# defense. The full measurement is in `_must_fill_mask`'s docstring; the
# short version is that `pos_DST` is fitted on a draft_picks table with ZERO
# DST rows in it, and without this floor no modelled opponent ever drafted a
# defense -- 7 of 8 simulated teams finished an entire 120-pick draft with an
# empty DST starter slot, every defense came back at 1.000 survival at every
# point of the draft, and gain_now for every defense was exactly 0.0000.

def _two_round_league():
    """Two teams, one QB and one DST to start, two rounds. Snake [1,2,2,1],
    so slot 2 holds back-to-back picks and both of its picks are visible in
    one four-pick draft -- the whole must-fill argument, with nothing else
    in it."""
    return league.LeagueSettings(
        season=2026, teams=2, starters={"QB": 1, "DST": 1},
        flex_slots=0, bench=0, scoring={}, draft_type="SNAKE")


def _qb_then_dst_pool(n_qb=6, n_dst=6):
    """The real board's shape in miniature: every defense ranks below every
    quarterback in the market, which is what makes `reach` (-8.09) plus
    `pos_DST` (-11.14) refuse them."""
    pos = ["QB"] * n_qb + ["DST"] * n_dst
    n = n_qb + n_dst
    return SimPool(
        player_id=np.array([f"p{i}" for i in range(n)]),
        norm=np.array([f"player {i}" for i in range(n)]),
        position=np.array(pos), adp_rank=np.arange(1, n + 1, dtype=float),
        points=np.linspace(300.0, 60.0, n), availability=np.full(n, 90.0),
        vor=np.linspace(300.0, 60.0, n),
        market_rank=np.arange(1, n + 1, dtype=float),
        age=np.full(n, np.nan), no_track_record=np.full(n, True),
        hype=np.full(n, np.nan), trend=np.zeros(n))


def test_the_roster_floor_does_not_bind_while_a_roster_has_slack():
    """It must not touch the rounds the fit is actually good in. Measured
    over 8 seeds of a full 120-pick draft on the real pool, the earliest pick
    at which it binds for anybody is 106 -- round 14 of 15."""
    from scoring.draft_sim import _must_fill_mask
    pool = _pool()
    idx = np.arange(len(pool.player_id))

    # Nothing drafted, 15 picks to go: 8 starter slots open, no constraint.
    assert _must_fill_mask(pool, idx, {}, S, 15) is None
    # Even down to 9 picks left with every starter slot still open.
    assert _must_fill_mask(pool, idx, {}, S, 9) is None
    # A roster with nothing left to fill is never constrained either.
    full = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1}
    assert _must_fill_mask(pool, idx, full, S, 1) is None


def test_the_roster_floor_binds_when_the_picks_run_down_to_the_open_slots():
    from scoring.draft_sim import _must_fill_mask
    pool = _pool()
    idx = np.arange(len(pool.player_id))

    # One slot open (DST), one pick left: this pick is that slot.
    counts = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1}
    mask = _must_fill_mask(pool, idx, counts, S, 1)
    assert mask is not None
    assert set(pool.position[idx[mask]]) == {"DST"}
    # Two open, two left: either, and nothing else.
    counts = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
    mask = _must_fill_mask(pool, idx, counts, S, 2)
    assert set(pool.position[idx[mask]]) == {"K", "DST"}
    # Two open, three left -- still slack, so still the fit's decision.
    assert _must_fill_mask(pool, idx, counts, S, 3) is None
    # No player at a needed position is available: None, never an empty mask,
    # so the caller can apply the result unconditionally.
    only_qbs = idx[pool.position[idx] == "QB"]
    counts = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1}
    assert _must_fill_mask(pool, only_qbs, counts, S, 1) is None


def test_an_opponent_fills_its_last_starter_slot_instead_of_a_second_qb():
    """Without the floor this opponent finishes with two quarterbacks and no
    defense, in every seed. The QB cap is 3 (_roster_cap), so a second QB is
    perfectly legal and the fit much prefers it: every defense here ranks
    below every quarterback in the market, so `reach` and `pos_DST` together
    put them out of reach.

    betas={} and slot_managers={} on purpose -- that is the unresolved
    opponent, which now follows COLD_START_PRIOR (see _run_draft's own
    comment). Six seeds, because the choice is sampled from a softmax and one
    seed proves nothing about a distribution.
    """
    settings, pool = _two_round_league(), _qb_then_dst_pool()
    n = len(pool.player_id)
    for seed in range(6):
        rosters = _run_draft(pool, settings, {}, 1, np.zeros(n, dtype=bool),
                             {}, np.random.default_rng(seed), taken_order=[])
        assert rosters[2]["counts"] == {"QB": 1, "DST": 1}, f"seed {seed}"


def test_survival_stops_pinning_every_defense_at_certainty():
    """The live consequence, and the reason the floor has to be in `survival`
    too and not only in `_run_draft`: this loop IS what "will he still be
    there" is counted from. With opponents that never draft a defense, every
    defense comes back at exactly 1.000, `gain.expected_best_next` collapses
    to the position's own leader, and `gain_now` is exactly 0.0000 for every
    defense at every pick of the draft -- so the room can never surface one.

    Measured on the real 249-player pool at the owner's final pick, same
    board state either way: best defense rank 3, gain_now +0.000, survival
    1.000 before; rank 1, gain_now +4.977, survival 0.235 after.
    """
    settings, pool = _two_round_league(), _qb_then_dst_pool()
    n = len(pool.player_id)
    frame = survival(pool, settings, {}, 1, np.zeros(n, dtype=bool), {},
                     n_rollouts=200, seed=11, taken_order=[],
                     on_the_clock=True, horizon=0)
    dst = frame[pool.position == "DST"]["avail_pct"].to_numpy()
    assert dst.max() < 1.0, "every defense still survives with certainty"
    # The BEST defense is the one at risk -- that ordering is the signal
    # `expected_best_next` reads, and a flat 1.000 destroys it.
    assert dst[0] < dst[-1]


def test_turns_left_counts_the_pick_on_the_clock_as_one_of_mine():
    from scoring.draft_sim import _turns_left
    slots = snake_slots(2, 2)                       # [1, 2, 2, 1]
    assert _turns_left(slots, 2) == [2, 2, 1, 1]


def test_greedy_takes_the_running_back_over_the_higher_scoring_quarterback():
    """The defect this fixes, from a real run: the simulated user opened
    with Josh Allen, market rank 24, at pick 4. A QB outscores every RB in
    raw points, so a one-ply greedy on raw points always takes one early --
    the error that pricing a candidate against his own position exists to
    prevent.

    What that price is, is the thing this test pins. It is NOT a static
    replacement level: that prices a quarterback against the 9th-best QB of
    the original board however deep the draft has gone, which is right on
    average and wrong exactly when it matters. It is what I could get at
    that position when I pick again -- so the fixture is built around the
    only quantity that can distinguish the two, how fast each position
    decays over the gap.

    QB declines 2 points a slot and RB declines 40. Both leaders are worth
    within 80 raw points of each other, both positions are equally deep, and
    the *static* replacement margins are deliberately set so they would
    favour the QB. Only the next-turn price separates them: waiting a full
    turn costs ~8 points at QB and ~160 at RB.
    """
    import numpy as np
    from scoring.draft_sim import SimPool, _greedy_choice, _roster_cap

    def pool_of(qb_points, rb_points):
        n_qb, n_rb = len(qb_points), len(rb_points)
        n = n_qb + n_rb
        ids = np.array([f"qb{i}" for i in range(n_qb)] +
                       [f"rb{i}" for i in range(n_rb)])
        points = np.concatenate([qb_points, rb_points])
        # Market rank follows raw points across the whole pool, which is what
        # makes the gap bite: `_next_turn_survivors` removes the top `gap` by
        # market rank, so the intervening picks take the best players left
        # regardless of position. The old fixture ranked every QB ahead of
        # every RB, under which no RB is ever taken by anybody and no gap can
        # cost anything -- it could not have distinguished the two policies.
        return SimPool(
            player_id=ids, norm=ids,
            position=np.array(["QB"] * n_qb + ["RB"] * n_rb),
            adp_rank=np.arange(1, n + 1, dtype=float),
            points=points,
            availability=np.full(n, 95.0), vor=np.zeros(n),
            market_rank=(-points).argsort().argsort().astype(float) + 1.0,
            age=np.full(n, 25.0), no_track_record=np.full(n, False),
            hype=np.zeros(n), trend=np.zeros(n))

    roster = {"counts": {}, "indices": []}

    # The flat position's leader outscores the steep position's, so raw
    # points and next-turn price give opposite answers and the test can tell
    # which one is running. Both 12 deep.
    flat = np.array([420.0, 416, 412, 408, 404, 400, 396, 392, 388, 384, 380, 376])
    steep = np.array([401.0, 340, 300, 275, 258, 246, 238, 233, 230, 228, 227, 226])

    pool = pool_of(flat, steep)          # QB flat, RB steep
    available = np.arange(len(pool.player_id))

    # Seven picks pass before my next turn, which takes the top seven by
    # market rank: five QBs, the lead RB, and one more QB. What is left is a
    # 396 quarterback (24 below the one on offer) and a 340 running back
    # (61 below). The RB is worth more even though he scores 19 fewer raw
    # points, and a static replacement level cannot see it.
    choice = _greedy_choice(pool, available, roster, S, _roster_cap(S), gap=7)
    assert pool.position[choice] == "RB"

    # Same pool, my last pick of the draft. Nothing can be taken from me
    # after it, so there is no next turn to price against and the honest
    # comparison is raw roster value -- which the 420-point quarterback
    # wins. `gap=None` is not `gap=0`, and this is the difference.
    choice = _greedy_choice(pool, available, roster, S, _roster_cap(S), gap=None)
    assert pool.position[choice] == "QB"

    # Inverse: hand the QBs the steep curve and the RBs the flat one and the
    # answer must flip to QB. Depths, gap and both point ranges are
    # unchanged -- only which position carries which curve. The RBs now hold
    # the *higher* raw numbers, so a policy tracking raw points or shortlist
    # scan order would still answer RB here.
    swapped = pool_of(steep, flat)
    choice = _greedy_choice(swapped, available, roster, S, _roster_cap(S), gap=7)
    assert swapped.position[choice] == "QB"


def test_rollout_never_drafts_past_a_roster_cap_even_when_the_shortlist_is_all_one_position():
    """Regression test for a roster-cap escape hatch.

    _greedy_choice's shortlist used to be the top GREEDY_CANDIDATES players
    by raw points pool-wide, computed BEFORE any legality filter. With QB
    points set far above every other position's, the top-40 shortlist is
    entirely QB; once my slot's QB count reaches its cap (3), the old code
    found every shortlisted candidate illegal, left best_idx unset, and fell
    through to the cap-blind "best remaining player" fallback -- taking a
    4th QB anyway. This builds exactly that scenario end-to-end through
    _run_draft and checks the actual resulting rosters, not just the cap
    table in isolation.

    Non-QB supply here is deliberately generous (40 apiece, 200 total,
    against a max aggregate demand across all 8 teams of 4 RB + 4 WR + 3 TE
    + 1 K + 1 DST = 13/team = 104) so that legal alternatives always exist
    SOMEWHERE in the pool once a manager is QB-capped -- an earlier, thinner
    version of this fixture (8 non-QB players apiece) let the whole market
    run out of non-QB supply, which manufactures a genuinely unavoidable
    "nothing legal left" case rather than exercising the shortlist bug this
    test targets. With this fixture, the pre-fix `_greedy_choice` (its
    shortlist built from the top GREEDY_CANDIDATES by points BEFORE any
    legality filter) rosters my slot 15 QB -- every single pick -- because
    the shortlist is 100% QB every turn and the old cap-blind fallback
    always wins; the fix should never exceed the cap of 3.
    """
    n = 400
    positions = np.array(["QB"] * 200 + (["RB", "WR", "TE", "K", "DST"] * 40))
    points = np.concatenate([np.linspace(1000.0, 300.0, 200),
                             np.linspace(50.0, 5.0, 200)])
    pool = SimPool(
        player_id=np.array([f"p{i}" for i in range(n)]),
        norm=np.array([f"player {i}" for i in range(n)]),
        position=positions, adp_rank=np.arange(1, n + 1, dtype=float),
        points=points, availability=np.full(n, 90.0), vor=points,
        market_rank=np.arange(1, n + 1, dtype=float), age=np.full(n, np.nan),
        no_track_record=np.full(n, True), hype=np.full(n, np.nan),
        trend=np.zeros(n))
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(n, dtype=bool)
    betas = _flat_betas(slots.values())
    caps = _roster_cap(S)
    for seed in range(10):
        rosters = _run_draft(pool, S, slots, 1, taken, betas,
                             rng=np.random.default_rng(seed))
        for slot, roster in rosters.items():
            for pos, count in roster["counts"].items():
                assert count <= caps.get(pos, 99), \
                    f"seed {seed}: slot {slot} rostered {count} {pos} (cap {caps.get(pos)})"


# --- Mid-draft pick attribution. `drafted.pick_no` records what order the
# board was marked up in; combined with the snake order that says which slot
# was on the clock for each of those picks. Without it every roster resumed
# empty: my own earlier picks vanished from the roster the search values,
# every opponent's `need` read 1.0 at every position, and `_roster_cap`
# re-armed from zero (a manager holding three QBs could take three more).


def test_run_draft_attributes_already_taken_players_to_the_slot_that_took_them():
    """16 picks made in an 8-team, 15-round league = two full rounds.

    snake_slots puts slot 4 on the clock at offsets 3 and 12, so those two
    picks are mine and must already be on my roster when the resume starts --
    leaving me with a full 15, not the 13 the unattributed version produced.
    """
    pool = _pool(300)
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    taken[:16] = True
    order = list(range(16))

    rosters = _run_draft(pool, S, slots, 4, taken, _flat_betas(slots.values()),
                         rng=np.random.default_rng(0), taken_order=order)

    snake = snake_slots(S.teams, S.rounds)
    for slot in range(1, S.teams + 1):
        seeded = [i for i in range(16) if snake[i] == slot]
        assert rosters[slot]["indices"][:len(seeded)] == seeded
        assert len(rosters[slot]["indices"]) == S.rounds
        assert sum(rosters[slot]["counts"].values()) == S.rounds
    assert rosters[4]["indices"][:2] == [3, 12]


def test_run_draft_counts_already_taken_players_against_the_roster_cap():
    """A manager who already holds three QBs must not be able to take more.

    Slot 1 is on the clock at offsets 0, 15 and 16; this hands it a QB at
    each of them, so it starts the resume at the QB cap. Unattributed, its
    count restarted at zero and it drafted up to three more.
    """
    pool = _pool(300)
    qbs = [i for i in range(300) if pool.position[i] == "QB"]
    others = [i for i in range(300) if pool.position[i] != "QB"]
    slot_one_offsets = {0, 15, 16}
    order = [qbs.pop(0) if offset in slot_one_offsets else others.pop(0)
             for offset in range(24)]
    taken = np.zeros(len(pool.player_id), dtype=bool)
    taken[order] = True
    slots = {i: f"m{i}" for i in range(1, 9)}

    rosters = _run_draft(pool, S, slots, 4, taken, _flat_betas(slots.values()),
                         rng=np.random.default_rng(0), taken_order=order)

    assert rosters[1]["counts"]["QB"] == _roster_cap(S)["QB"] == 3


def test_run_draft_keeps_the_snake_aligned_when_a_taken_player_left_the_board():
    """A pick whose player is no longer in the pool still consumed its turn.

    `taken_order` carries None for it rather than dropping the entry, so
    every later pick still lands on the slot that actually made it.
    """
    pool = _pool(300)
    slots = {i: f"m{i}" for i in range(1, 9)}
    order = [None] + list(range(1, 16))       # pick 1's player is off the board
    taken = np.zeros(len(pool.player_id), dtype=bool)
    taken[1:16] = True

    rosters = _run_draft(pool, S, slots, 4, taken, _flat_betas(slots.values()),
                         rng=np.random.default_rng(0), taken_order=order)

    snake = snake_slots(S.teams, S.rounds)
    assert rosters[1]["indices"][0] != 0      # slot 1's pick 1 is not rostered
    for slot in range(1, S.teams + 1):
        seeded = [i for i in range(1, 16) if snake[i] == slot]
        assert rosters[slot]["indices"][:len(seeded)] == seeded


def test_run_sim_refuses_to_run_when_a_drafted_row_has_no_pick_no(
        tmp_path, monkeypatch):
    """Rows drafted before the pick_no migration cannot be attributed.

    Ordering them first (or in any other invented order) hands real players
    to the wrong teams and corrupts every roster, need and cap downstream
    with no warning, so run_sim refuses and says how to fix it.
    """
    conn = get_conn(str(tmp_path / "t.duckdb"))
    conn.execute("INSERT INTO drafted VALUES ('p3', NULL)")
    personal = np.zeros(len(FEATURE_NAMES))
    monkeypatch.setattr(draft_model_mod, "fit_all",
                        lambda conn, settings=None: {"__pooled__": _POOLED,
                                                     "m1": personal})
    monkeypatch.setattr(board_mod, "build_board",
                        lambda conn, weights=None, settings=None: pd.DataFrame())
    monkeypatch.setattr(draft_sim_mod, "build_pool",
                        lambda conn, board, settings: _pool(12))

    with pytest.raises(ValueError, match="pick_no"):
        run_sim(conn, my_slot=1, slot_managers={i: "m1" for i in range(1, 9)}, n_rollouts=2, seed=0)


def test_run_sim_seeds_rosters_from_the_recorded_pick_order(tmp_path, monkeypatch):
    """End-to-end: `drafted.pick_no` reaches the simulator as attribution.

    Three picks recorded through the same INSERT the API uses; run_sim must
    hand search_pick/survival a taken_order matching that pick order, not
    just a pool-wide mask.
    """
    conn = get_conn(str(tmp_path / "t.duckdb"))
    for pick_no, pid in enumerate(["p5", "p1", "p9"], start=1):
        conn.execute("INSERT INTO drafted VALUES (?, ?)", [pid, pick_no])
    zeros = np.zeros(len(FEATURE_NAMES))
    monkeypatch.setattr(draft_model_mod, "fit_all",
                        lambda conn, settings=None: {"__pooled__": _POOLED, "m1": zeros})
    monkeypatch.setattr(board_mod, "build_board",
                        lambda conn, weights=None, settings=None: pd.DataFrame())
    monkeypatch.setattr(draft_sim_mod, "build_pool",
                        lambda conn, board, settings: _pool(12))

    captured = {}

    def fake_search_pick(pool, settings, slot_managers, my_slot, taken, betas,
                         **kwargs):
        captured["taken_order"] = kwargs.get("taken_order")
        return pd.DataFrame(columns=["player_id", "ev", "se", "rank"])

    monkeypatch.setattr(draft_sim_mod, "search_pick", fake_search_pick)
    monkeypatch.setattr(draft_sim_mod, "survival",
                        lambda *a, **k: pd.DataFrame(columns=["player_id", "avail_pct"]))

    run_id = run_sim(conn, my_slot=1, slot_managers={i: "m1" for i in range(1, 9)}, n_rollouts=2, seed=0)

    # _pool(n) numbers players p0..p(n-1) in pool order, so the pool index of
    # "pK" is K -- pick order p5, p1, p9 must arrive as [5, 1, 9].
    assert captured["taken_order"] == [5, 1, 9]
    assert run_id.endswith("-3")               # three picks already made


from scoring.draft_sim import (run_sim, SEARCH_COLUMNS, _candidate_indices,
                               search_pick, survival)


def test_search_pick_ranks_the_best_candidate_first():
    pool = _pool()
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    out = search_pick(pool, S, slots, 1, taken, _flat_betas(slots.values()),
                      n_rollouts=25, n_candidates=6, seed=11)
    # `applied_pct` joined the schema with the fix for dead candidates: how
    # often forcing this candidate actually happened rather than falling
    # through to the greedy baseline.
    assert list(out.columns) == ["player_id", "ev", "se", "applied_pct", "rank"]
    assert out["rank"].tolist() == [1, 2, 3, 4, 5, 6]
    assert out["ev"].is_monotonic_decreasing
    assert (out["se"] >= 0).all()
    assert (out["applied_pct"] == 1.0).all()      # my_slot 1 picks first

def test_search_pick_uses_common_random_numbers():
    # Same seed, same candidates -> byte-identical EVs across calls.
    pool = _pool()
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    betas = _flat_betas(slots.values())
    a = search_pick(pool, S, slots, 1, taken, betas, n_rollouts=15,
                    n_candidates=4, seed=5)
    b = search_pick(pool, S, slots, 1, taken, betas, n_rollouts=15,
                    n_candidates=4, seed=5)
    assert a["ev"].tolist() == b["ev"].tolist()

def _anti_correlated_pool(n=60):
    """Market rank and VOR deliberately disagree: pool index 0 is the market's
    number one and the worst by VOR, index n-1 the reverse."""
    positions = np.array(["RB", "WR", "QB", "TE", "K", "DST"] * (n // 6))
    return SimPool(
        player_id=np.array([f"p{i}" for i in range(n)]),
        norm=np.array([f"player {i}" for i in range(n)]),
        position=positions,
        adp_rank=np.arange(1, n + 1, dtype=float),
        points=np.linspace(60.0, 300.0, n),
        availability=np.full(n, 90.0),
        vor=np.linspace(60.0, 300.0, n),
        market_rank=np.arange(1, n + 1, dtype=float), age=np.full(n, np.nan),
        no_track_record=np.full(n, True), hype=np.full(n, np.nan),
        trend=np.zeros(n))


def test_candidate_set_draws_from_both_market_rank_and_vor():
    """The second candidate source has to actually operate.

    `dict.fromkeys(by_market + by_vor)[:n_candidates]` returned `by_market`
    verbatim -- both slices held n_candidates already-unique entries, so the
    truncation threw the whole second source away. With market rank and VOR
    anti-correlated the candidates came back p0..p11 and the top twelve by
    VOR (p59..p48) contributed nothing at all.
    """
    pool = _anti_correlated_pool()
    available = np.arange(len(pool.player_id))
    everyone_survives = np.ones(len(pool.player_id))

    picked = _candidate_indices(pool, available, everyone_survives, 12)

    assert len(picked) == 12
    assert len(set(picked)) == 12
    top_market = set(range(6))                       # p0..p5
    top_vor = set(range(len(pool.player_id) - 6, len(pool.player_id)))   # p54..p59
    assert len(top_market & set(picked)) >= 5
    assert len(top_vor & set(picked)) >= 5


def test_candidate_set_skips_players_who_will_not_survive_to_my_pick():
    """A candidate an opponent has already taken by my turn falls through to
    the greedy policy, so under common random numbers every such candidate
    produces the identical draft and reads as a precisely-measured near-tie.
    Candidates below MIN_CANDIDATE_AVAIL are not evaluated at all.
    """
    pool = _pool()
    available = np.arange(len(pool.player_id))
    avail_pct = np.ones(len(pool.player_id))
    avail_pct[:8] = 0.05                             # the top of the board turns over

    picked = _candidate_indices(pool, available, avail_pct, 6)

    assert all(idx >= 8 for idx in picked)


def test_search_pick_drops_candidates_it_almost_never_actually_gets():
    """The end-to-end version of the same thing, in the reviewer's scenario.

    Slot 8 against ADP-disciplined opponents with nothing drafted: the top of
    the board is gone seven picks in. Before the fix, p0/p1/p2/p3 came back
    byte-identical (ev=1332.449153, se=7.406298) and were ranked 3rd through
    6th. Now no candidate is reported unless forcing him actually happened,
    and no two rows can be exactly equal by construction.
    """
    pool = _pool()
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)

    out = search_pick(pool, S, slots, 8, taken, _adp_betas(slots.values()),
                      n_rollouts=60, n_candidates=12, seed=0)

    assert not out.empty
    assert (out["applied_pct"] >= 0.25).all()
    assert not out.duplicated(["ev", "se"]).any()


def _adp_betas(managers):
    """Opponents who follow market order: a negative `reach` coefficient
    penalizes players whose ADP rank sits later than the current pick."""
    beta = np.zeros(len(FEATURE_NAMES))
    beta[FEATURE_NAMES.index("reach")] = -3.0
    return {m: beta for m in managers}

def test_survival_probabilities_are_between_zero_and_one_and_favor_late_adp():
    pool = _pool()
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    out = survival(pool, S, slots, 8, taken, _adp_betas(slots.values()),
                   n_rollouts=40, seed=2)
    assert ((out["avail_pct"] >= 0) & (out["avail_pct"] <= 1)).all()
    first = out[out["player_id"] == "p0"]["avail_pct"].iloc[0]
    last = out[out["player_id"] == "p59"]["avail_pct"].iloc[0]
    assert last > first


# --- Supplementary tests for task-11 properties the brief's own tests only
# exercise indirectly: _next_pick_for's pick arithmetic in isolation,
# survival's "stop BEFORE my own pick" boundary at its tightest (my turn is
# immediately next), and run_sim's uses_personal gating -- the entire point
# of Task 8's held-out personal-vs-pooled comparison is wasted if the
# simulator ignores the flag it produced.

from scoring.draft_sim import _next_pick_for


def test_next_pick_for_returns_the_overall_pick_number_of_my_next_turn():
    """8 teams, 15 rounds (S = default_settings()) -> snake_slots has 120
    entries. Slot 5's turns land at overall picks 5 (round 1, forward),
    12 (round 2, reversed: 8,7,6,5,...), 21 (round 3, forward), ... .
    `already` counts picks made so far, not rounds, so this pins down both
    "my very first turn" and "resuming mid-draft after my own last pick"."""
    assert _next_pick_for(S, my_slot=5, already=0) == 5
    assert _next_pick_for(S, my_slot=5, already=5) == 12    # right after my pick 5
    assert _next_pick_for(S, my_slot=5, already=11) == 12   # right before my pick 12
    # already >= the whole draft's pick count (120): no turn remains: the
    # documented off-the-end fallback (len(slots) + 1), not an IndexError or
    # a wraparound back to an earlier pick.
    assert _next_pick_for(S, my_slot=5, already=120) == 121


def test_survival_does_not_simulate_when_my_turn_is_immediately_next():
    """Correctness point 2: survival stops BEFORE my next pick rather than
    running the draft out. Isolate the tightest case -- my_slot picks first
    overall (slot 1, nobody drafted yet) -- so `_next_pick_for` returns 1
    and the simulation loop's range is empty: zero picks get simulated, and
    "who can I wait on" for the pick being valued is answered as
    "everyone", not whatever the board would look like several picks later.

    This is the DEFAULT (`on_the_clock=False`) reading, the one
    search_pick/run_sim want: they value the pick they are about to make,
    and a player on the board right now is available for it with
    certainty. The live ranking asks the other question and passes
    `on_the_clock=True` -- see the test below.
    """
    pool = _pool()
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    out = survival(pool, S, slots, 1, taken, _flat_betas(slots.values()),
                   n_rollouts=10, seed=0)
    assert (out["avail_pct"] == 1.0).all()


def test_survival_on_the_clock_measures_the_turn_after_this_one():
    """The whole-branch Critical: at the exact moment the user is on the
    clock, `survival` returned 1.0 for every available player.

    `_next_pick_for` scans from `already` inclusively, so with slot 1 and
    nobody drafted it answered "my next turn is pick 1" -- the pick being
    made right now -- and the rollout loop had nothing to range over. That
    is the moment api/live.py's `_recompute` is called with (`made =
    count(*) FROM drafted`, i.e. exactly picks_made), so the LAST ranking
    computed before the user picks was always the degenerate one, and it
    is the one on screen while they pick.

    `on_the_clock=True` measures the turn after this one instead: slot 1's
    pick 16 in an 8-team snake, with picks 2..15 simulated in between, so
    the deep pool cannot all survive.
    """
    pool = _pool()
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    out = survival(pool, S, slots, 1, taken, _adp_betas(slots.values()),
                   n_rollouts=40, seed=0, on_the_clock=True)
    assert (out["avail_pct"] < 1.0).any(), \
        "every player survived: the rollout loop ran zero picks again"
    # 14 opponents pick between pick 1 and slot 1's pick 16, so at most 14
    # players can have been removed -- and the market-following betas take
    # them off the top of the board, so p0 in particular cannot be certain.
    assert out.loc[out["player_id"] == "p0", "avail_pct"].iloc[0] < 1.0
    assert ((out["avail_pct"] >= 0) & (out["avail_pct"] <= 1)).all()


from scoring.gain import rank_available


def test_survival_on_the_clock_makes_gain_now_non_degenerate():
    """The consequence the Critical was actually read through.

    With survival pinned at 1.0, `gain.expected_best_next` short-circuits
    after the first element (`none_better *= 1 - 1.0`), so `next_best[pos]`
    is exactly `max(vor at pos)` and `gain_now` is identically 0.0 for the
    leader at EVERY position -- six rows tied at zero, above every other
    player, with `sort_values`' default (unstable) quicksort deciding which
    three of them the room captions "take one of these". A kicker was as
    likely to land there as a running back.

    Same state the recompute sees at the same moment: slot 1 on the clock
    at pick 1. Asserts both halves -- survival is no longer uniform, and
    the position leaders are no longer tied at exactly zero.
    """
    pool = _pool()
    slot_managers = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    avail = survival(pool, S, slot_managers, 1, taken,
                     _adp_betas(slot_managers.values()),
                     n_rollouts=40, seed=0, on_the_clock=True)["avail_pct"]
    frame = rank_available(pool, S, taken, {}, avail.to_numpy())
    leaders = frame.sort_values("vor_points", ascending=False).drop_duplicates(
        "position")
    assert not (leaders["gain_now"] == 0.0).all(), \
        "every position leader tied at exactly zero -- gain_now is degenerate"
    # The top three are a real ranking, not a coin flip between six tied
    # rows: the best row strictly beats the third-best.
    assert frame["gain_now"].iloc[0] > frame["gain_now"].iloc[2]


# --- The horizon. `gain_now` is one step of a position's supply curve --
# now versus the turn survival is measured to -- and over a one- or
# two-opponent-pick step that number is ~0 for everybody, so the ranking has
# no signal left and the top of the board is decided by the sort. Observed
# live at pick 1 of an 8-team mock with the owner in slot 2 (exactly ONE
# opponent pick before their turn): every available player at ~100%
# survival, and the top six read Gibbs, Nacua, McBride (TE), Josh Allen
# (QB), Houston DST, Brandon Aubrey (K) -- a defense fifth and a kicker
# sixth in round 1. The fix measures to the first turn at least
# `horizon_picks(settings)` opponent picks away instead. In the owner's own
# words, a tight end being available in round 9 is not a reason to take one
# in round 3, and only the SHAPE of the curve can tell those apart.

from scoring.draft_sim import _horizon_pick_for, horizon_picks


def test_horizon_picks_is_one_round_of_opponent_picks_derived_from_teams():
    """`teams - 1`: a round is `teams` picks and one of them is mine.

    Derived from the league, never a hardcoded 7 -- a 12-team league gets
    11. The dummy settings object only needs `.teams`.
    """
    assert horizon_picks(S) == S.teams - 1 == 7
    assert horizon_picks(type("S12", (), {"teams": 12})()) == 11


def test_horizon_threshold_splits_the_snakes_two_gaps_for_every_slot():
    """Why `teams - 1` is the natural line and not an arbitrary knob.

    Each slot's two turns in a round pair are 2*(slot-1) and
    2*(teams-slot) opponent picks apart, which always sum to 2*teams-2 --
    so exactly one of them is below `teams - 1` and one is at or above it.
    The threshold therefore skips the turnaround (my picks at or near the
    wheel, where the board barely moves in between) and keeps the genuine
    full-round wait, for every slot. Verified against snake_slots itself
    rather than against that formula.
    """
    slots = snake_slots(S.teams, S.rounds)
    h = horizon_picks(S)
    for my_slot in range(1, S.teams + 1):
        turns = [i for i, s in enumerate(slots) if s == my_slot]
        # Opponent picks between my first three turns: the two gaps.
        gaps = [turns[k + 1] - turns[k] - 1 for k in range(2)]
        assert sum(gaps) == 2 * S.teams - 2
        assert min(gaps) < h <= max(gaps), (my_slot, gaps)


def test_horizon_pick_for_at_zero_is_exactly_next_pick_for():
    """The horizon=0 case is not a special case: it is the general walk
    evaluated at zero, which is what `_next_pick_for` (run_sim's and
    search_pick's only reading of "my next pick") now calls. If these two
    ever disagreed, the offline callers would have quietly changed meaning.
    """
    for my_slot in (1, 2, 5, 8):
        for already in (0, 1, 7, 8, 15, 60, 119, 120, 130):
            assert (_horizon_pick_for(S, my_slot, already, 0)
                    == _next_pick_for(S, my_slot, already))


def test_horizon_pick_for_skips_turns_too_close_to_measure_anything():
    """8 teams, 15 rounds. Slot 2's turns are 2, 15, 18, 31, ... .

    From nothing drafted, my next turn (pick 2) is one opponent pick away
    and is skipped; pick 15 is 13 opponent picks away and is taken. From my
    own pick 15 (start=15, the on_the_clock reading), pick 18 is the wheel
    partner two opponent picks away and is skipped for pick 31 -- 12
    opponent picks, since my own pick 18 does not count against the
    threshold.
    """
    h = horizon_picks(S)
    assert _horizon_pick_for(S, 2, 0, h) == 15
    assert _horizon_pick_for(S, 2, 15, h) == 31
    # Slot 8 at the wheel: on the clock at pick 8 (start=8), its own pick 9
    # follows with NO opponent in between, so the horizon has to reach 24.
    assert _horizon_pick_for(S, 8, 8, h) == 24


def test_horizon_pick_for_falls_back_to_my_last_turn_then_off_the_end():
    """Two different fallbacks, deliberately.

    Late in the draft no turn of mine is a full round away any more. That
    is not a reason to answer "the end of the draft" -- my last turn is a
    real pick, and it is the most informative horizon that actually exists
    for me. Only when I hold NO turn after this one does the answer become
    `len(slots) + 1`, the off-the-end sentinel `_next_pick_for` has always
    returned, which the caller must render as "the end of the draft"
    rather than as a pick number (api/live.py's horizon_is_end_of_draft).
    """
    h = horizon_picks(S)
    slots = snake_slots(S.teams, S.rounds)
    mine = [i + 1 for i, s in enumerate(slots) if s == 2]
    assert mine[-1] == 114 and len(slots) == 120
    # On the clock at 111 (start=111): only pick 114 remains, 2 opponent
    # picks away -- under the threshold, so the fallback returns it anyway.
    assert _horizon_pick_for(S, 2, 111, h) == 114
    # On the clock at my last pick: no turn remains at all.
    assert _horizon_pick_for(S, 2, 114, h) == len(slots) + 1


def test_survival_with_a_horizon_is_not_degenerate_at_a_one_pick_gap():
    """The owner's exact scenario, in miniature: slot 2 with pick 1 on the
    clock, one opponent pick before my turn.

    Without the horizon every player survives (only one pick happens), so
    `gain.expected_best_next` equals each position's own leader and
    `gain_now` is 0.0 for the leader at EVERY position -- six rows tied at
    zero above everything else, which is how a kicker and a defense reached
    the top six of a round-1 board. With it, the same call measures to pick
    15 and the leaders separate.
    """
    pool = _pool()
    slot_managers = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    betas = _adp_betas(slot_managers.values())

    def ranked(horizon):
        avail = survival(pool, S, slot_managers, 2, taken, betas,
                         n_rollouts=40, seed=0, horizon=horizon)["avail_pct"]
        return rank_available(pool, S, taken, {}, avail.to_numpy())

    def leaders(frame):
        return frame.sort_values("vor_points", ascending=False) \
            .drop_duplicates("position")

    # One opponent pick can remove one player, so at most one position moves
    # at all: three of the six leaders (QB, K and DST here) come back at
    # EXACTLY 0.0, tied with each other and sorted above all 54 players with
    # a negative gain, and only three players in the whole pool have any
    # positive gain to rank on. That tie is the defect -- on the real board
    # it is what put a defense 5th and a kicker 6th at pick 1.
    before = ranked(0)
    assert int((before["gain_now"] > 0).sum()) == 3
    assert int((leaders(before)["gain_now"] == 0.0).sum()) == 3

    # Measuring to pick 15 instead: every leader separates, and there is a
    # real ranking underneath them rather than three rows and a tie.
    after = ranked(horizon_picks(S))
    assert not (leaders(after)["gain_now"] == 0.0).any()
    assert int((after["gain_now"] > 0).sum()) > 10


def test_survival_at_the_wheel_is_no_longer_a_uniform_one():
    """The residual artifact the one-step horizon carried, now retired.

    At the wheel my two picks are back to back, so measuring to my
    "next" turn simulated zero opponent picks and returned a truthful but
    useless 1.0 for everybody -- degenerate `gain_now` for one turn of
    every round pair. The horizon walks past that turn to the next one
    that is a real wait, so the same state now produces a real
    distribution.
    """
    pool = _pool()
    slot_managers = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    taken[:7] = True                      # picks 1-7 gone; slot 8 on the clock
    taken_order = list(range(7))
    betas = _adp_betas(slot_managers.values())

    old = survival(pool, S, slot_managers, 8, taken, betas, n_rollouts=20,
                   seed=0, taken_order=taken_order, on_the_clock=True)
    assert (old.loc[~taken, "avail_pct"] == 1.0).all(), \
        "fixture drifted off the documented wheel artifact"

    new = survival(pool, S, slot_managers, 8, taken, betas, n_rollouts=40,
                   seed=0, taken_order=taken_order, on_the_clock=True,
                   horizon=horizon_picks(S))
    assert (new.loc[~taken, "avail_pct"] < 1.0).any()


def test_survival_default_leaves_the_offline_callers_where_they_were():
    """`run_sim` and `search_pick` never pass `horizon`, and the default has
    to mean exactly what they already meant: my very next pick. Same seed,
    same rollouts -- the two frames must be identical, not merely close.
    """
    import inspect
    assert inspect.signature(survival).parameters["horizon"].default == 0
    pool = _pool()
    slot_managers = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    betas = _adp_betas(slot_managers.values())
    default = survival(pool, S, slot_managers, 5, taken, betas,
                       n_rollouts=30, seed=3)
    explicit = survival(pool, S, slot_managers, 5, taken, betas,
                        n_rollouts=30, seed=3, horizon=0)
    pd.testing.assert_frame_equal(default, explicit)


# --- Regression: `betas.get(slot_managers.get(slot))` misses to `None` for
# ANY slot a caller hasn't named a manager for -- every slot, for every
# league, until ESPN publishes `draftDayPickOrder` (slot_managers stays {}),
# and every manager in a cold-start league with no fitted history (betas
# stays {}, since api/live.py's `build_session` builds it as `{m:
# fits.get(m, pooled) for m in fits if m != "__pooled__"}`, which is {}
# when `fit_all` returns only `__pooled__`). The fallback for that miss used
# to be `np.zeros(len(FEATURE_NAMES))`, which makes `scores = X @ beta`
# identically zero and the softmax over the whole remaining pool uniform --
# every unresolved opponent draws completely at random, ignoring adp_rank,
# need, position, everything.
#
# Observed live: pick 22 of an 8-team mock draft, 9 picks before the owner's
# next turn (slot 6 on the clock, owner in slot 2 -- reproduced exactly
# below). Every available player, Derrick Henry and Josh Jacobs included,
# came back at 94-98% survival: indistinguishable from a kicker, because a
# uniform draw over the ~249-deep pool gives every player survival
# ~= 1 - 9/249 = 0.964 regardless of how good he is. `gain.expected_best_next`
# then collapsed to each position's own best player and `gain_now` was a
# wall of near-zeros, so pandas' unstable sort -- not the model -- decided
# that a kicker and a QB, not a running back, filled two of the top three
# recommendation slots.
def test_survival_falls_back_to_the_market_prior_not_zeros_when_unresolved():
    """With `slot_managers={}` and `betas={}` (both misses, the live bug's
    exact inputs) at a realistic 9-pick gap, the highest-ADP player still on
    the board must survive at MATERIALLY below 1.0 -- the market-following
    answer, not the old uniform fallback's near-certain "he's fine, waiting
    costs nothing" answer.

    8 teams, 21 picks already made (so pick 22 -- the observed pick -- is
    about to happen, on the clock is slot 6), owner sits in slot 2: verified
    separately that slot 2's next turn is pick 31, exactly 9 opponent picks
    away, matching the live report. The old zeros fallback measured on this
    exact fixture/scenario put the highest-ADP survivor at 0.731 -- a
    threshold of 0.5 cleanly fails against that and passes against the fix.
    """
    pool = _pool(60)
    already = 21
    my_slot = 2
    taken = np.zeros(len(pool.player_id), dtype=bool)
    taken[:already] = True
    taken_order = list(range(already))

    target = _next_pick_for(S, my_slot=my_slot, already=already)
    assert target - 1 - already == 9, "fixture drifted off the observed 9-pick gap"

    out = survival(pool, S, slot_managers={}, my_slot=my_slot, taken=taken,
                   betas={}, n_rollouts=300, seed=0, taken_order=taken_order,
                   on_the_clock=False)
    avail = out.set_index("player_id")["avail_pct"]

    top_of_board = avail["p21"]                 # adp_rank 22, the best player left
    assert top_of_board < 0.5, (
        f"top-ADP available player survived at {top_of_board:.3f} -- too "
        "close to the old uniform-random fallback's 0.731 on this same "
        "fixture, not the market-following prior's near-certain 'he's gone'")


def test_run_draft_follows_market_order_not_uniform_random_when_unresolved():
    """`_run_draft` has its own copy of the identical fallback, in the branch
    that simulates an opponent's pick -- same miss (`betas.get(
    slot_managers.get(slot))` -> None), same old zeros default, same
    uniform-random consequence. Not reachable from the live path
    (api/live.py never calls `_run_draft`/`rollout` directly, only
    `survival`) or from `run_sim` (which fills every referenced manager's
    beta via `betas.setdefault(manager, pooled)` before this code ever
    runs, and refuses to run at all if a slot has no manager) -- but
    `_run_draft`/`rollout`/`predict_board` are public and any other caller
    reaches this fallback exactly as directly as `survival`'s callers did,
    so it is fixed for the same reason.

    Pick 1 of a fresh draft (nobody taken) has nothing to distinguish
    players by except adp_rank, and COLD_START_PRIOR's `reach` coefficient
    (-8.09, the largest-magnitude weight in the vector) should make that
    pick close to deterministic: the rank-1 player, every time. Measured
    separately: pool index 0 in 50/50 rollouts under this fix, versus a
    mean pool index of ~31 (out of 60 -- a uniform draw) under the old
    zeros fallback.
    """
    pool = _pool(60)
    taken = np.zeros(len(pool.player_id), dtype=bool)
    first_picks = []
    for seed in range(20):
        record = []
        _run_draft(pool, S, {}, 8, taken, {}, rng=np.random.default_rng(seed),
                  record=record)
        first_picks.append(record[0][1])   # pool index chosen with pick 1

    assert first_picks == [0] * len(first_picks), (
        f"pick 1 did not always go to the rank-1 player: {first_picks} -- "
        "the market-following prior should make this near-deterministic, "
        "not spread uniformly across the pool the way the old zeros "
        "fallback did")


import scoring.board as board_mod
import scoring.draft_model as draft_model_mod
import scoring.draft_sim as draft_sim_mod
from scoring.draft_sim import run_sim


def test_run_sim_gates_personal_coefficients_on_the_manager_profiles_flag(
        tmp_path, monkeypatch):
    """run_sim must use a manager's personal fit only when
    manager_profiles.uses_personal is True for that manager, and the pooled
    fit otherwise (correctness point 4) -- that gate is the entire reason
    Task 8 computed a held-out personal-vs-pooled comparison in the first
    place; skipping it would mean every manager silently gets one fit type
    regardless of whether it actually predicts them better.

    fit_all only produces a personal fit that differs from pooled given
    real, multi-season draft history clearing MIN_PICKS_FOR_PERSONAL --
    building that fixture here would exercise fit_all's own convergence,
    not run_sim's gating decision. Instead `fit_all`, `build_board`, and
    `build_pool` are monkeypatched to return small, fully controlled
    values (fit_all returns the SAME distinctive, non-zero "personal" beta
    for both managers, and a distinct all-zero "pooled" beta), and
    `search_pick`/`survival` are monkeypatched to capture the `betas` dict
    run_sim actually builds and hands them, rather than running real
    rollouts. manager_profiles then flags one manager uses_personal=True and
    the other False. If the gate is honored, "reacher" ends up with the
    personal beta and "average" ends up with pooled -- even though fit_all
    handed both of them the identical personal fit, so the only thing that
    can explain "average" getting pooled is run_sim's own gating logic.
    """
    conn = get_conn(str(tmp_path / "t.duckdb"))
    write_table(conn, "manager_profiles", pd.DataFrame([
        {"manager": "reacher", "feature": "reach", "value": 9.0,
         "pooled_value": 0.0, "n_picks": 40, "heldout_gain": 0.2,
         "uses_personal": True, "summary": "reaches"},
        {"manager": "average", "feature": "reach", "value": 9.0,
         "pooled_value": 0.0, "n_picks": 5, "heldout_gain": -0.3,
         "uses_personal": False, "summary": "pooled, not enough signal"},
    ]))

    personal_beta = np.zeros(len(FEATURE_NAMES))
    personal_beta[FEATURE_NAMES.index("reach")] = 9.0
    pooled_beta = _POOLED
    fits = {"__pooled__": pooled_beta, "reacher": personal_beta,
            "average": personal_beta}

    monkeypatch.setattr(draft_model_mod, "fit_all",
                        lambda conn, settings=None: fits)
    monkeypatch.setattr(board_mod, "build_board",
                        lambda conn, weights=None, settings=None: pd.DataFrame())
    monkeypatch.setattr(draft_sim_mod, "build_pool",
                        lambda conn, board, settings: _pool())

    captured = {}

    def fake_search_pick(pool, settings, slot_managers, my_slot, taken, betas,
                         **kwargs):
        captured["betas"] = betas
        return pd.DataFrame(columns=["player_id", "ev", "se", "rank"])

    def fake_survival(pool, settings, slot_managers, my_slot, taken, betas,
                      **kwargs):
        return pd.DataFrame(columns=["player_id", "avail_pct"])

    monkeypatch.setattr(draft_sim_mod, "search_pick", fake_search_pick)
    monkeypatch.setattr(draft_sim_mod, "survival", fake_survival)

    run_sim(conn, my_slot=1, slot_managers={**{i: "average" for i in range(1, 9)}, 1: "reacher"},
           n_rollouts=5, seed=0)

    betas = captured["betas"]
    np.testing.assert_allclose(betas["reacher"], personal_beta)
    np.testing.assert_allclose(betas["average"], pooled_beta)


def test_run_sim_writes_sim_results_and_sim_survival_and_returns_the_run_id(
        tmp_path, monkeypatch):
    """End-to-end smoke test with real (not monkeypatched) search_pick/
    survival, on a tiny synthetic pool -- run_sim's own DB plumbing
    (`write_table` for both tables, the run_id format, the `my_slot` column
    on results) rather than the gating logic the test above isolates.

    `conn` has no `league` table, so the real `league.load` already falls
    back to `default_settings()` (matching `S`) with no mocking needed;
    only `build_board`/`build_pool` (real board construction needs stat and
    ADP tables this fixture does not seed) and `fit_all` (no draft history
    to fit) are replaced.
    """
    conn = get_conn(str(tmp_path / "t.duckdb"))
    zeros = np.zeros(len(FEATURE_NAMES))
    # fit_all must return at least one per-manager fit: run_sim now refuses
    # to run when there are none, because with none every opponent draws
    # uniformly over the whole pool. This mock's job is to stand in for
    # "manager models exist", not to make that check vacuous.
    monkeypatch.setattr(draft_model_mod, "fit_all",
                        lambda conn, settings=None: {"__pooled__": _POOLED,
                                                     **{f"m{i}": zeros for i in range(1, 9)}})
    monkeypatch.setattr(board_mod, "build_board",
                        lambda conn, weights=None, settings=None: pd.DataFrame())
    monkeypatch.setattr(draft_sim_mod, "build_pool",
                        lambda conn, board, settings: _pool(12))

    run_id = run_sim(conn, my_slot=1,
                     slot_managers={i: f"m{i}" for i in range(1, 9)},
                     n_rollouts=5, seed=0)

    assert run_id == "1-5-0-0"
    results = read_table(conn, "sim_results")
    avail = read_table(conn, "sim_survival")
    assert not results.empty
    assert (results["run_id"] == run_id).all()
    assert (results["my_slot"] == 1).all()
    # Provenance: which pick this run was for, and when it ran.
    assert (results["pick_no"] == 1).all()          # slot 1, nothing drafted
    assert results["created_at"].notna().all()
    assert not avail.empty
    assert (avail["run_id"] == run_id).all()


def test_run_sim_warns_when_a_fitted_reach_coefficient_is_positive(
        tmp_path, monkeypatch):
    """`reach` is max(0, adp_rank - pick) / teams, so a positive coefficient
    says "the further past his market rank a player is, the more I want him".
    exp(reach * beta) on a rank-500 player then dominates every other term
    and that manager is simulated drafting the deepest player in the pool.

    It is most likely a symptom of build_pool's dense 1..k rank not lining up
    with the population historic_adp's per-season rank was fitted over -- the
    branch's own logged open question -- and there is nowhere else it
    surfaces.
    """
    conn = get_conn(str(tmp_path / "t.duckdb"))
    reaching = np.zeros(len(FEATURE_NAMES))
    reaching[FEATURE_NAMES.index("reach")] = 0.4
    monkeypatch.setattr(draft_model_mod, "fit_all",
                        lambda conn, settings=None: {"__pooled__": reaching,
                                                     "gunslinger": reaching})
    monkeypatch.setattr(board_mod, "build_board",
                        lambda conn, weights=None, settings=None: pd.DataFrame())
    monkeypatch.setattr(draft_sim_mod, "build_pool",
                        lambda conn, board, settings: _pool(12))
    monkeypatch.setattr(draft_sim_mod, "search_pick",
                        lambda *a, **k: pd.DataFrame(columns=SEARCH_COLUMNS))
    monkeypatch.setattr(draft_sim_mod, "survival",
                        lambda *a, **k: pd.DataFrame(columns=["player_id", "avail_pct"]))

    with pytest.warns(RuntimeWarning, match="reach coefficient is positive"):
        run_sim(conn, my_slot=1, slot_managers={i: "gunslinger" for i in range(1, 9)},
                n_rollouts=2, seed=0)


def test_run_sim_refuses_a_degenerate_all_zero_pooled_model(tmp_path, monkeypatch):
    """The guard's new contract. A no-history league is NO LONGER refused --
    fit_all returns a market-following prior (cold_start_fits) and every
    opponent inherits it, which is a measured default rather than noise.

    What IS still refused is a pooled vector of all zeros: that produces a
    uniform draw over ~500 players, where the consensus number one comes back
    100% available at slot 8 -- a confidently wrong answer the board would
    merge with no way to tell. No production path yields it (fit_all always
    returns a non-zero prior), so reaching it means a build error upstream,
    and refusing beats simulating a uniform draft silently.
    """
    conn = get_conn(str(tmp_path / "t.duckdb"))
    monkeypatch.setattr(draft_model_mod, "fit_all",
                        lambda conn, settings=None: {
                            "__pooled__": np.zeros(len(FEATURE_NAMES))})
    monkeypatch.setattr(board_mod, "build_board",
                        lambda conn, weights=None, settings=None: pd.DataFrame())
    monkeypatch.setattr(draft_sim_mod, "build_pool",
                        lambda conn, board, settings: _pool(12))

    with pytest.raises(ValueError, match="all zero"):
        run_sim(conn, my_slot=1, slot_managers={i: "m1" for i in range(1, 9)},
                n_rollouts=2, seed=0)
    assert read_table(conn, "sim_results").empty


def test_run_sim_runs_a_cold_start_league_with_only_the_market_prior(tmp_path, monkeypatch):
    """The other half: a league with no history at all still runs. fit_all's
    real cold_start_fits returns just __pooled__ (the market prior), every
    opponent inherits it, and the sim produces a result rather than raising --
    which is what makes the tool usable for a brand-new user."""
    from scoring.draft_model import cold_start_fits
    conn = get_conn(str(tmp_path / "t.duckdb"))
    monkeypatch.setattr(draft_model_mod, "fit_all",
                        lambda conn, settings=None: cold_start_fits())
    monkeypatch.setattr(board_mod, "build_board",
                        lambda conn, weights=None, settings=None: pd.DataFrame())
    monkeypatch.setattr(draft_sim_mod, "build_pool",
                        lambda conn, board, settings: _pool(12))

    run_id = run_sim(conn, my_slot=1,
                     slot_managers={i: f"stranger{i}" for i in range(1, 9)},
                     n_rollouts=3, seed=0)
    assert not read_table(conn, "sim_results").empty

def test_run_sim_falls_back_to_pooled_when_manager_profiles_has_not_been_written(
        tmp_path, monkeypatch):
    """Regression test for a real bug found while timing run_sim (not one of
    the brief's own scenarios, which never happen to exercise this path --
    see below).

    `run_sim` reads `manager_profiles` with `pipeline.db.read_table`, which
    returns a columnless `pd.DataFrame()` when the table does not exist yet
    -- the ordinary state of a fresh DB before `make fit-managers` (or this
    task's own `write_profiles`) has ever run, e.g. right after
    `make espn-import`. The brief's original gating line,
    `rows = profiles[profiles["manager"] == manager]`, indexes that
    columnless frame by "manager" unconditionally once `fits` contains any
    real (non-"__pooled__") manager, which raises `KeyError: 'manager'`
    instead of falling back to pooled.

    The brief's own tests never hit this: this file's other run_sim tests
    mock `fit_all` to return only `{"__pooled__": ...}`, so the per-manager
    loop body -- where the crash lives -- never executes. Building an actual
    multi-manager, multi-season draft-history fixture during the timing
    measurement for correctness point 5 (see task-11-report.md) is what
    surfaced it. Fixed in scoring/draft_sim.py: `profiles.empty or
    "manager" not in profiles.columns` now short-circuits to
    personal=False (pooled) before the indexing that used to crash.
    """
    conn = get_conn(str(tmp_path / "t.duckdb"))
    # manager_profiles deliberately never written.
    personal_beta = np.zeros(len(FEATURE_NAMES))
    personal_beta[FEATURE_NAMES.index("reach")] = 5.0
    fits = {"__pooled__": _POOLED, "some_manager": personal_beta}
    monkeypatch.setattr(draft_model_mod, "fit_all", lambda conn, settings=None: fits)
    monkeypatch.setattr(board_mod, "build_board",
                        lambda conn, weights=None, settings=None: pd.DataFrame())
    monkeypatch.setattr(draft_sim_mod, "build_pool",
                        lambda conn, board, settings: _pool(12))

    captured = {}

    def fake_search_pick(pool, settings, slot_managers, my_slot, taken, betas,
                         **kwargs):
        captured["betas"] = betas
        return pd.DataFrame(columns=["player_id", "ev", "se", "rank"])

    monkeypatch.setattr(draft_sim_mod, "search_pick", fake_search_pick)
    monkeypatch.setattr(draft_sim_mod, "survival",
                        lambda *a, **k: pd.DataFrame(columns=["player_id", "avail_pct"]))

    run_sim(conn, my_slot=1, slot_managers={i: "some_manager" for i in range(1, 9)}, n_rollouts=5, seed=0)

    # No manager_profiles row exists to say "yes, use the personal fit" --
    # must fall back to pooled, not crash, and not silently use
    # the personal fit either.
    np.testing.assert_allclose(captured["betas"]["some_manager"], _POOLED)


def test_run_sim_refuses_a_draft_order_that_does_not_cover_every_slot(tmp_path, monkeypatch):
    """ESPN leaves draftDayPickOrder null until it publishes an order, so a
    caller building {slot: manager} from those rows collapses every team into
    one entry keyed None. That used to fall through to a zeros beta -- every
    opponent picking uniformly at random -- and report confident numbers."""
    import pandas as pd
    from pipeline.db import get_conn, write_table
    from scoring import draft_model, league
    conn = get_conn(str(tmp_path / "t.duckdb"))
    write_table(conn, "league", pd.DataFrame(
        [{"season": 2026, "settings_json": league.to_json(league.default_settings())}]))
    monkeypatch.setattr(draft_model, "fit_all", lambda *a, **k: {
        "__pooled__": _POOLED,
        "solo": np.zeros(len(FEATURE_NAMES))})
    with pytest.raises(ValueError, match="have no manager"):
        run_sim(conn, my_slot=1, slot_managers={None: "solo"}, n_rollouts=2)


def test_run_draft_records_every_pick_in_order_when_asked():
    pool = _pool()
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    record = []
    rosters = _run_draft(pool, S, slots, 4, taken, _flat_betas(slots.values()),
                         rng=np.random.default_rng(3), record=record)
    picks = [p for p, _ in record]
    assert picks == list(range(1, len(picks) + 1))
    # Every recorded index is a real pool index, and no player goes twice.
    indices = [i for _, i in record]
    assert len(set(indices)) == len(indices)
    assert all(0 <= i < len(pool.player_id) for i in indices)
    # The record and the rosters describe the same draft.
    from_rosters = sorted(i for st in rosters.values() for i in st["indices"])
    assert sorted(indices) == from_rosters


def test_run_draft_without_record_is_unchanged():
    pool = _pool()
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    betas = _flat_betas(slots.values())
    a = _run_draft(pool, S, slots, 4, taken, betas, rng=np.random.default_rng(9))
    b = _run_draft(pool, S, slots, 4, taken, betas, rng=np.random.default_rng(9),
                   record=[])
    assert {s: st["indices"] for s, st in a.items()} == \
           {s: st["indices"] for s, st in b.items()}


def test_predict_board_covers_every_pick_and_sums_sensibly():
    from scoring.draft_sim import predict_board
    pool = _pool()
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    board = predict_board(pool, S, slots, 4, taken, _adp_betas(slots.values()),
                          n_rollouts=20, seed=1)
    assert list(board.columns) == ["overall_pick", "round", "round_pick",
                                   "slot", "alt_rank", "player_id", "prob",
                                   "certain"]
    primaries = board[board["alt_rank"] == 0]
    # 60-player pool, 8 teams x 15 rounds -- the pool runs dry at pick 60.
    assert primaries["overall_pick"].tolist() == list(range(1, 61))
    assert (primaries["prob"] > 0).all()
    assert (primaries["prob"] <= 1).all()
    assert not primaries["certain"].any()


def test_predict_board_round_and_slot_follow_the_snake():
    from scoring.draft_sim import predict_board
    pool = _pool()
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    board = predict_board(pool, S, slots, 4, taken, _adp_betas(slots.values()),
                          n_rollouts=10, seed=2)
    p = board[board["alt_rank"] == 0].set_index("overall_pick")
    assert (p.loc[1, "round"], p.loc[1, "slot"]) == (1, 1)
    assert (p.loc[8, "round"], p.loc[8, "slot"]) == (1, 8)
    assert (p.loc[9, "round"], p.loc[9, "slot"]) == (2, 8)
    assert (p.loc[16, "round"], p.loc[16, "slot"]) == (2, 1)
    assert p.loc[9, "round_pick"] == 1


def test_predict_board_never_puts_one_player_in_two_cells_unless_forced():
    """The old global assignment guaranteed no duplicate primary whenever a
    repeat-free board existed at all -- it was solving for exactly that.
    Draft-order greedy (Task 7) deliberately trades that guarantee away: it
    only looks at one pick at a time, in ascending overall_pick, so a
    duplicate is now allowed whenever a pick's own rollouts recorded no
    candidate that an earlier pick hadn't already claimed.

    A blanket "no duplicates ever" assertion is therefore no longer a true
    statement about this system -- it would be pinning the retired
    algorithm's behavior, not this one's contract. What must still hold is
    narrower: every duplicate that does appear is *forced*, i.e. by the time
    that pick was processed, every player its own rollouts ever sent there
    was already somebody else's primary. `alternates` is set to the whole
    pool so each cell's full recorded candidate set is visible in `board`,
    not just the top two -- otherwise this couldn't be checked from the
    outside at all.
    """
    from scoring.draft_sim import predict_board
    pool = _pool()
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    board = predict_board(pool, S, slots, 4, taken, _adp_betas(slots.values()),
                          n_rollouts=25, seed=4, alternates=len(pool.player_id))
    by_pick = {pick: grp for pick, grp in board.groupby("overall_pick")}

    claimed_by = {}
    for pick in sorted(by_pick):
        cell = by_pick[pick]
        chosen = cell.loc[cell["alt_rank"] == 0, "player_id"].iloc[0]
        if chosen in claimed_by:
            # A repeat is legitimate only if this pick's own full candidate
            # set -- every player its own rollouts ever produced here, alt
            # rows included -- was already exhausted by an earlier pick.
            own_candidates = set(cell["player_id"])
            already_claimed = set(claimed_by)
            unclaimed = own_candidates - already_claimed
            assert not unclaimed, (
                f"pick {pick} repeated {chosen!r} while {unclaimed} of its "
                "own recorded candidates were still free")
        else:
            claimed_by[chosen] = pick


def test_predict_board_keeps_a_deduped_player_as_an_alternate():
    """The dedupe must not hide information.

    With ADP-disciplined opponents the same early player wins several
    adjacent cells on raw frequency. Assignment gives him exactly one, and
    the cells he lost must still list him among their alternates -- that is
    the whole reason hover exists.
    """
    from scoring.draft_sim import predict_board
    pool = _pool()
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    board = predict_board(pool, S, slots, 4, taken, _adp_betas(slots.values()),
                          n_rollouts=40, seed=5, alternates=2)
    top = board[(board["overall_pick"] == 1) & (board["alt_rank"] == 0)]
    top_id = top["player_id"].iloc[0]
    nearby = board[(board["overall_pick"].between(2, 5))
                   & (board["alt_rank"] > 0)]["player_id"].tolist()
    assert top_id in nearby


def test_predict_board_alternates_are_ranked_and_never_repeat_the_primary():
    from scoring.draft_sim import predict_board
    pool = _pool()
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    board = predict_board(pool, S, slots, 4, taken, _adp_betas(slots.values()),
                          n_rollouts=30, seed=6, alternates=2)
    for pick, grp in board.groupby("overall_pick"):
        grp = grp.sort_values("alt_rank")
        assert grp["alt_rank"].tolist() == list(range(len(grp)))
        assert len(set(grp["player_id"])) == len(grp)
        alts = grp[grp["alt_rank"] > 0]["prob"].tolist()
        assert alts == sorted(alts, reverse=True)


def test_predict_board_is_deterministic_under_a_fixed_seed():
    from scoring.draft_sim import predict_board
    pool = _pool()
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    args = (pool, S, slots, 4, taken, _adp_betas(slots.values()))
    a = predict_board(*args, n_rollouts=12, seed=7)
    b = predict_board(*args, n_rollouts=12, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_predict_board_reports_already_made_picks_as_certain():
    from scoring.draft_sim import predict_board
    pool = _pool()
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    taken[:3] = True
    board = predict_board(pool, S, slots, 4, taken, _adp_betas(slots.values()),
                          n_rollouts=10, seed=8, taken_order=[0, 1, 2])
    made = board[board["certain"]]
    assert made["overall_pick"].tolist() == [1, 2, 3]
    assert made["player_id"].tolist() == [pool.player_id[i] for i in (0, 1, 2)]
    assert (made["prob"] == 1.0).all()
    assert (made["alt_rank"] == 0).all()
    # A certain pick has no alternates, and the predictions start after it.
    assert board[board["overall_pick"] <= 3]["alt_rank"].max() == 0
    assert not board[board["overall_pick"] == 4]["certain"].any()


def test_predict_board_skips_a_certain_pick_whose_player_fell_off_the_pool():
    """A `taken_order` entry can be None -- a player drafted, then dropped
    from the pool by a later refresh (see `_seed_rosters`'s docstring). His
    pick still consumed a turn, but there is no player_id left to report, so
    that overall_pick must produce no certain row while the turns around it
    still do.
    """
    from scoring.draft_sim import predict_board
    pool = _pool()
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    taken[[0, 2]] = True
    board = predict_board(pool, S, slots, 4, taken, _adp_betas(slots.values()),
                          n_rollouts=10, seed=9, taken_order=[0, None, 2])
    made = board[board["certain"]]
    assert made["overall_pick"].tolist() == [1, 3]
    assert made["player_id"].tolist() == [pool.player_id[0], pool.player_id[2]]
    # Pick 2's turn was consumed (predictions resume at pick 4, not pick 2),
    # but it reports no row of its own -- certain or otherwise.
    assert 2 not in board["overall_pick"].tolist()


def test_predict_board_drops_picks_past_the_end_of_the_draft():
    """`taken_order` can be longer than the draft has turns -- 8 x 15 is 120
    slots, and nothing stops the drafted table holding more rows than that.
    Those picks used to be emitted with `slot = 0`, and no column on the grid
    carries slot 0, so they disappeared off the page with no trace.
    `_seed_rosters` already breaks on exactly this condition; report the same
    set of picks it does.
    """
    from scoring.draft_sim import predict_board
    pool = _pool(150)                     # more players than teams x rounds
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    taken[:125] = True
    board = predict_board(pool, S, slots, 4, taken, _adp_betas(slots.values()),
                          n_rollouts=5, seed=3, taken_order=list(range(125)))
    assert (board["slot"] >= 1).all()
    assert board["overall_pick"].tolist() == list(range(1, 121))
    assert board["certain"].all()


def test_assign_primaries_never_invents_a_candidate_outside_the_cells_own_players():
    """Regression case from code review: a naive per-cell argmax fallback
    (`max(cell, key=...)`, with no notion of who else already claimed what)
    can hand two different picks the same primary even though a
    claimed-set-aware walk never repeats a player it doesn't have to.

    counts = {1: {10: 6, 11: 4}, 2: {10: 5, 11: 5}, 3: {10: 7, 11: 3}, 4: {12: 10}}
    has only 3 distinct recorded players (10, 11, 12) for 4 picks. Picks 1-3
    between them only ever recorded players 10 and 11 -- two candidates for
    three picks -- so by the pigeonhole principle no assignment that only
    ever uses a pick's own recorded candidates can give all three of them
    distinct primaries; some repeat is mathematically forced no matter the
    algorithm (draft-order greedy included -- this pins that the current
    implementation is one of the algorithms this reasoning covers).

    What IS a meaningful, checkable property -- and what actually
    distinguishes a correct implementation from a broken one -- is that the
    forced repeat is still one of that pick's own recorded candidates, never
    a fabricated one, and that exactly one repeat occurs here (not more):
    picks 1, 2 and 4 each come away with their own distinct winner (1 and 2
    are processed first and split the only two players anyone recorded for
    them), and pick 3 -- processed last among the trio, by which point both
    10 and 11 are already claimed -- reuses its own top candidate, 10,
    rather than being handed a player nobody ever recorded for it.
    """
    from scoring.draft_sim import _assign_primaries
    counts = {1: {10: 6, 11: 4}, 2: {10: 5, 11: 5}, 3: {10: 7, 11: 3}, 4: {12: 10}}
    primary = _assign_primaries(counts)

    assert set(primary) == set(counts)
    # Every primary is a player that pick's own rollouts actually produced.
    for overall_pick, chosen in primary.items():
        assert chosen in counts[overall_pick]
    # Exactly one repeat -- the mathematical minimum for this input, not
    # more (a sloppier fallback could, in principle, produce extra repeats
    # among the picks that need one; this pins that it does not).
    values = list(primary.values())
    assert len(values) - len(set(values)) == 1


def test_assign_primaries_is_deterministic():
    from scoring.draft_sim import _assign_primaries
    counts = {1: {10: 6, 11: 4}, 2: {10: 5, 11: 5}, 3: {10: 7, 11: 3}, 4: {12: 10}}
    a = _assign_primaries(counts)
    b = _assign_primaries(counts)
    assert a == b


def test_assign_primaries_gives_pick_one_its_most_likely_player():
    """The defect this fixes, from a real run: pick 1 showed a 12% player
    while a 16% player sat in the hover, because the global assignment
    'saved' him for pick 13 where he scored 40%."""
    from scoring.draft_sim import _assign_primaries
    counts = {1: {10: 6, 11: 8}, 13: {11: 20, 12: 5}}
    primary = _assign_primaries(counts)
    assert primary[1] == 11          # pick 1 gets its own most likely
    assert primary[13] == 12         # 11 is claimed, 13 takes the next


def test_assign_primaries_processes_picks_in_draft_order():
    # Pick 5's cell must carry a real second candidate (4): with only {1: 10}
    # as originally drafted, player 1 is pick 5's *only* recorded candidate,
    # so once pick 2 claims him first, primary[5] == 1 is forced by the
    # "keep your own best when every candidate is claimed" rule -- the same
    # rule test_assign_primaries_allows_a_duplicate_only_when_forced pins
    # elsewhere. `primary[5] != 1` cannot hold for that input under any
    # implementation of the stated rule; giving pick 5 a genuine fallback
    # candidate is what actually exercises "a later, non-priority pick falls
    # back to its own next-best rather than inheriting the earlier claim".
    from scoring.draft_sim import _assign_primaries
    counts = {5: {1: 10, 4: 1}, 2: {1: 9, 2: 3}, 9: {1: 30, 3: 2}}
    primary = _assign_primaries(counts)
    assert primary[2] == 1           # earliest pick wins the contested player
    assert primary[5] != 1 and primary[9] != 1


def test_assign_primaries_allows_a_duplicate_only_when_forced():
    from scoring.draft_sim import _assign_primaries
    counts = {1: {7: 5}, 2: {7: 5}}   # one candidate, two picks
    primary = _assign_primaries(counts)
    assert primary == {1: 7, 2: 7}


def _seed_board_tables(conn):
    """Minimal weekly/schedules/adp/espn tables so build_board returns rows."""
    import pandas as pd
    from pipeline.db import write_table
    write_table(conn, "weekly", pd.DataFrame([
        {"player_id": f"p{n}", "player_display_name": f"Player {n}",
         "position": ["QB", "RB", "WR", "TE"][n % 4], "recent_team": "DET",
         "opponent_team": "GB", "season": 2025, "week": w,
         "receptions": 5, "receiving_yards": 60, "targets": 7, "carries": 3}
        for n in range(1, 30) for w in range(1, 18)]))
    write_table(conn, "schedules", pd.DataFrame([
        {"home_team": "DET", "away_team": "GB", "week": 1,
         "total_line": 48.0, "spread_line": 2.0}]))
    write_table(conn, "adp", pd.DataFrame(
        columns=["adp_name", "position", "team", "adp"]))
    write_table(conn, "depth_charts", pd.DataFrame(
        columns=["gsis_id", "depth_team", "formation", "week", "position"]))
    write_table(conn, "espn_adp", pd.DataFrame(
        columns=["espn_id", "espn_name", "position", "espn_adp",
                 "espn_ppr_rank", "espn_proj"]))
    write_table(conn, "fp_ecr", pd.DataFrame(
        columns=["fp_name", "team", "position", "rank_ecr", "rank_ave",
                 "rank_std", "fp_tier"]))
    write_table(conn, "sleeper_ids", pd.DataFrame(
        columns=["gsis_id", "espn_id", "sleeper_name", "position", "team"]))


def test_run_sim_writes_sim_board(tmp_path, monkeypatch):
    import pandas as pd
    from pipeline.db import get_conn, write_table, read_table
    from scoring import draft_model, league
    conn = get_conn(str(tmp_path / "t.duckdb"))
    write_table(conn, "league", pd.DataFrame(
        [{"season": 2026, "settings_json": league.to_json(league.default_settings())}]))
    _seed_board_tables(conn)
    monkeypatch.setattr(draft_model, "fit_all", lambda *a, **k: {
        "__pooled__": _POOLED,
        **{f"m{i}": np.zeros(len(FEATURE_NAMES)) for i in range(1, 9)}})
    run_id = run_sim(conn, my_slot=1,
                     slot_managers={i: f"m{i}" for i in range(1, 9)},
                     n_rollouts=3, seed=0)
    board = read_table(conn, "sim_board")
    assert list(board.columns) == ["run_id", "overall_pick", "round",
                                   "round_pick", "slot", "alt_rank",
                                   "player_id", "prob", "certain"]
    assert not board.empty
    assert set(board["run_id"]) == {run_id}
    # No blanket "every primary is unique" check: with uniform-random
    # opponents (zero betas), a forced duplicate is a likely, legitimate
    # outcome of the draft-order-greedy rule (Task 7), not a bug. But the
    # weaker, real invariant -- any duplicate must be forced -- IS checkable
    # here without touching run_sim's interface: a cell's distinct
    # candidate count can never exceed n_rollouts (each rollout contributes
    # at most one entry per pick), and this run uses n_rollouts=3 against
    # `predict_board`'s default alternates=2, i.e. n_rollouts <= alternates
    # + 1. That bound means every cell's full candidate set (up to 3
    # distinct players) fits inside the 3 rows (1 primary + 2 alternates)
    # `sim_board` already carries -- nothing is truncated. Same walk as
    # test_predict_board_never_puts_one_player_in_two_cells_unless_forced.
    by_pick = {pick: grp for pick, grp in board.groupby("overall_pick")}
    claimed_by = {}
    for pick in sorted(by_pick):
        cell = by_pick[pick]
        chosen = cell.loc[cell["alt_rank"] == 0, "player_id"].iloc[0]
        if chosen in claimed_by:
            own_candidates = set(cell["player_id"])
            already_claimed = set(claimed_by)
            unclaimed = own_candidates - already_claimed
            assert not unclaimed, (
                f"pick {pick} repeated {chosen!r} while {unclaimed} of its "
                "own recorded candidates were still free")
        else:
            claimed_by[chosen] = pick


def test_live_features_matches_feature_matrix_on_the_new_columns():
    """The two implementations of one feature definition must agree. A
    mismatch silently invalidates every simulation while everything runs."""
    import numpy as np
    from scoring.draft_model import FEATURE_NAMES, PickObservation, feature_matrix
    from scoring.draft_sim import SimPool, _live_features
    import pandas as pd

    pool_df = pd.DataFrame({
        "norm": ["a", "b", "c", "d"],
        "position": ["RB", "WR", "QB", "TE"],
        "adp_rank": [1.0, 12.0, 40.0, 90.0],
        "market_rank": [2.0, 10.0, 55.0, 80.0],
        "hype": [-4.0, 7.0, np.nan, 25.0],
        "age": [23.0, 29.0, 34.0, np.nan],
        "no_track_record": [False, False, True, False],
        "trend": [2.5, -1.75, 0.0, 0.4]})
    obs = PickObservation(season=2026, overall_pick=9, manager="m", chosen=0,
                          pool=pool_df, roster={"RB": 1}, recent=["WR", "RB"])
    sim = SimPool(
        player_id=np.array(["a", "b", "c", "d"]),
        norm=pool_df["norm"].to_numpy(), position=pool_df["position"].to_numpy(),
        adp_rank=pool_df["adp_rank"].to_numpy(),
        points=np.array([300.0, 250.0, 380.0, 190.0]),
        availability=np.full(4, 90.0), vor=np.array([150.0, 120.0, 80.0, 60.0]),
        market_rank=pool_df["market_rank"].to_numpy(),
        age=pool_df["age"].to_numpy(),
        no_track_record=pool_df["no_track_record"].to_numpy(),
        hype=pool_df["hype"].to_numpy(), trend=pool_df["trend"].to_numpy())
    available = np.arange(4)
    np.testing.assert_allclose(
        _live_features(sim, available, 9, {"RB": 1}, ["WR", "RB"], S),
        feature_matrix(obs, S))
