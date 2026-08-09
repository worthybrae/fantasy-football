from itertools import combinations

import numpy as np
import pandas as pd
from pipeline.db import get_conn, write_table
from scoring import league
from scoring.draft_sim import (FLEX_POSITIONS, POSITION_FLOOR, best_lineup_points,
                               build_pool, projections, roster_value)

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
    by_id = dict(zip(pool.player_id, pool.adp_rank))
    # Known market_rank players occupy dense ranks 1..3, in market order --
    # unaffected by p3's real rank value (500.0) exceeding len(board) (5).
    assert by_id["p2"] == 1.0     # market_rank 5.0
    assert by_id["p1"] == 2.0     # market_rank 20.0
    assert by_id["p3"] == 3.0     # market_rank 500.0 -- still ranked 3rd
    # Unranked players continue the sequence after the known ones, not
    # interleaved with them.
    assert sorted([by_id["p4"], by_id["p5"]]) == [4.0, 5.0]


from scoring.draft_model import FEATURE_NAMES
from scoring.draft_sim import SimPool, rollout, snake_slots


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
        durability=np.full(n, 90.0))


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
                    rng=np.random.default_rng(3))
    assert value > 0                     # runs out of players without crashing


def test_rollout_handles_taken_covering_the_whole_draft_without_crashing():
    """Locks in rollout's resume contract for the fully-drafted edge case.

    `taken`'s count of True values IS the number of picks already made
    (rollout's / _run_draft's docstring). If that count already meets or
    exceeds the total number of picks in the draft (teams x rounds), there
    is nothing left to simulate, and _run_draft returns explicitly rather
    than falling through a `slots[already:]` slice that happens to be empty.
    _run_draft has no per-team ownership record for players marked `taken`
    before it was called -- only a pool-wide "off the board" mask -- so 0.0
    is the correct, documented answer here, not a numeric regression: this
    test does not change under the Finding 1/2 fixes, it guards against a
    future refactor silently breaking the degenerate case.
    """
    pool = _pool(150)                     # more players than teams x rounds (120)
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.ones(len(pool.player_id), dtype=bool)   # already >= len(slots)
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
    """
    positions = np.array(["QB", "RB", "WR", "TE", "K", "DST",
                          "RB", "WR", "QB", "TE", "RB", "WR"])
    norms = np.array([f"player {i}" for i in range(len(positions))])
    adp_rank = np.arange(1, len(positions) + 1, dtype=float)
    taken = np.zeros(len(positions), dtype=bool)
    taken[[2, 7]] = True                      # both taken players are WR
    available = np.flatnonzero(~taken)

    pool = SimPool(
        player_id=np.array([f"p{i}" for i in range(len(positions))]),
        norm=norms, position=positions, adp_rank=adp_rank,
        points=np.linspace(300.0, 60.0, len(positions)),
        durability=np.full(len(positions), 90.0))

    roster = {"RB": 2, "WR": 1, "QB": 1}       # RB need false, others true
    recent = ["WR", "RB", "RB", "QB", "TE"]    # fills RUN_WINDOW exactly

    obs_pool = pd.DataFrame({
        "norm": norms[available], "position": positions[available],
        "adp_rank": adp_rank[available]})

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

from scoring.draft_sim import _roster_cap, _run_draft


def test_roster_cap_limits_kicker_defense_and_qb_regardless_of_starters():
    caps = _roster_cap(S)
    assert caps["K"] == 1
    assert caps["DST"] == 1
    assert caps["QB"] <= 3


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
        points=points, durability=np.full(n, 90.0))
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
