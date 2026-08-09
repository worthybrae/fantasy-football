from itertools import combinations

import numpy as np
import pandas as pd
from pipeline.db import get_conn, write_table
from scoring import league
from scoring.draft_sim import (FLEX_POSITIONS, POSITION_FLOOR, best_lineup_points,
                               projections, roster_value)

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
