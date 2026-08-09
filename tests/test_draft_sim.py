from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from pipeline.db import get_conn, read_table, write_table
from scoring import league
from scoring.draft_sim import (DEFAULT_AVAILABILITY, FLEX_POSITIONS, POSITION_FLOOR,
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
        availability=np.full(n, 90.0),
        vor=np.linspace(300.0, 60.0, n))


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
        availability=np.full(len(positions), 90.0),
        vor=np.linspace(300.0, 60.0, len(positions)))

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
        points=points, availability=np.full(n, 90.0), vor=points)
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
                        lambda conn, settings=None: {"__pooled__": personal,
                                                     "m1": personal})
    monkeypatch.setattr(board_mod, "build_board",
                        lambda conn, weights=None, settings=None: pd.DataFrame())
    monkeypatch.setattr(draft_sim_mod, "build_pool",
                        lambda conn, board, settings: _pool(12))

    with pytest.raises(ValueError, match="pick_no"):
        run_sim(conn, my_slot=1, slot_managers={1: "m1"}, n_rollouts=2, seed=0)


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
                        lambda conn, settings=None: {"__pooled__": zeros, "m1": zeros})
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

    run_id = run_sim(conn, my_slot=1, slot_managers={1: "m1"}, n_rollouts=2, seed=0)

    # _pool(n) numbers players p0..p(n-1) in pool order, so the pool index of
    # "pK" is K -- pick order p5, p1, p9 must arrive as [5, 1, 9].
    assert captured["taken_order"] == [5, 1, 9]
    assert run_id.endswith("-3")               # three picks already made


from scoring.draft_sim import _candidate_indices, search_pick, survival


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
        vor=np.linspace(60.0, 300.0, n))


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
    "who can I wait on" for right now is answered as "everyone", not
    whatever the board would look like several picks later.
    """
    pool = _pool()
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    out = survival(pool, S, slots, 1, taken, _flat_betas(slots.values()),
                   n_rollouts=10, seed=0)
    assert (out["avail_pct"] == 1.0).all()


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
    pooled_beta = np.zeros(len(FEATURE_NAMES))
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

    run_sim(conn, my_slot=1, slot_managers={1: "reacher", 2: "average"},
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
    monkeypatch.setattr(draft_model_mod, "fit_all",
                        lambda conn, settings=None: {"__pooled__": np.zeros(len(FEATURE_NAMES))})
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
    assert not avail.empty
    assert (avail["run_id"] == run_id).all()


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
    fits = {"__pooled__": np.zeros(len(FEATURE_NAMES)), "some_manager": personal_beta}
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

    run_sim(conn, my_slot=1, slot_managers={1: "some_manager"}, n_rollouts=5, seed=0)

    # No manager_profiles row exists to say "yes, use the personal fit" --
    # must fall back to pooled (all zeros), not crash, and not silently use
    # the personal fit either.
    np.testing.assert_allclose(captured["betas"]["some_manager"],
                               np.zeros(len(FEATURE_NAMES)))
