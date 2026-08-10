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
from scoring.draft_sim import SimPool, _run_draft, rollout, snake_slots


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
        age=np.full(n, np.nan), ppg_std=np.zeros(n), missed_rate=np.zeros(n),
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
    ppg_std = np.array([2.0, 5.5, 3.25, 1.0, 0.0, 4.0,
                        6.5, 2.75, 0.5, 3.0, 1.5, 4.25])
    missed_rate = np.array([0.1, 0.0, 0.35, 0.05, 0.2, 0.0,
                            0.15, 0.4, 0.0, 0.25, 0.1, 0.3])
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
        market_rank=adp_rank, age=age, ppg_std=ppg_std,
        missed_rate=missed_rate, no_track_record=no_track_record,
        hype=hype, trend=trend)

    roster = {"RB": 2, "WR": 1, "QB": 1}       # RB need false, others true
    recent = ["WR", "RB", "RB", "QB", "TE"]    # fills RUN_WINDOW exactly

    obs_pool = pd.DataFrame({
        "norm": norms[available], "position": positions[available],
        "adp_rank": adp_rank[available], "market_rank": adp_rank[available],
        "age": age[available], "ppg_std": ppg_std[available],
        "missed_rate": missed_rate[available],
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
        points=points, availability=np.full(n, 90.0), vor=points,
        market_rank=np.arange(1, n + 1, dtype=float), age=np.full(n, np.nan),
        ppg_std=np.zeros(n), missed_rate=np.zeros(n),
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
                        lambda conn, settings=None: {"__pooled__": personal,
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
        ppg_std=np.zeros(n), missed_rate=np.zeros(n),
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
                        lambda conn, settings=None: {"__pooled__": zeros,
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


def test_run_sim_refuses_to_run_with_no_fitted_manager_models(tmp_path, monkeypatch):
    """No draft history means fit_all returns {} and every opponent's beta is
    zeros -- a uniform draw over roughly 500 available players. Measured on a
    500-player pool, the consensus number one's Avail% at slot 8 comes back
    100% and the top ten average 98.75%. That is a confidently wrong answer,
    not a degraded one, and the board merges it with no way to tell.
    """
    conn = get_conn(str(tmp_path / "t.duckdb"))
    monkeypatch.setattr(draft_model_mod, "fit_all", lambda conn, settings=None: {})
    monkeypatch.setattr(board_mod, "build_board",
                        lambda conn, weights=None, settings=None: pd.DataFrame())
    monkeypatch.setattr(draft_sim_mod, "build_pool",
                        lambda conn, board, settings: _pool(12))

    with pytest.raises(ValueError, match="fit-managers"):
        run_sim(conn, my_slot=1, slot_managers={i: "m1" for i in range(1, 9)}, n_rollouts=2, seed=0)
    assert read_table(conn, "sim_results").empty


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

    run_sim(conn, my_slot=1, slot_managers={i: "some_manager" for i in range(1, 9)}, n_rollouts=5, seed=0)

    # No manager_profiles row exists to say "yes, use the personal fit" --
    # must fall back to pooled (all zeros), not crash, and not silently use
    # the personal fit either.
    np.testing.assert_allclose(captured["betas"]["some_manager"],
                               np.zeros(len(FEATURE_NAMES)))


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
        "__pooled__": np.zeros(len(FEATURE_NAMES)),
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
    primary = _assign_primaries(counts, n_rollouts=10)

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
    a = _assign_primaries(counts, n_rollouts=10)
    b = _assign_primaries(counts, n_rollouts=10)
    assert a == b


def test_assign_primaries_gives_pick_one_its_most_likely_player():
    """The defect this fixes, from a real run: pick 1 showed a 12% player
    while a 16% player sat in the hover, because the global assignment
    'saved' him for pick 13 where he scored 40%."""
    from scoring.draft_sim import _assign_primaries
    counts = {1: {10: 6, 11: 8}, 13: {11: 20, 12: 5}}
    primary = _assign_primaries(counts, n_rollouts=50)
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
    primary = _assign_primaries(counts, n_rollouts=50)
    assert primary[2] == 1           # earliest pick wins the contested player
    assert primary[5] != 1 and primary[9] != 1


def test_assign_primaries_allows_a_duplicate_only_when_forced():
    from scoring.draft_sim import _assign_primaries
    counts = {1: {7: 5}, 2: {7: 5}}   # one candidate, two picks
    primary = _assign_primaries(counts, n_rollouts=10)
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
        "__pooled__": np.zeros(len(FEATURE_NAMES)),
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
        "ppg_std": [1.5, 6.0, 0.0, 3.25],
        "missed_rate": [0.0, 0.35, 0.1, 0.0],
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
        age=pool_df["age"].to_numpy(), ppg_std=pool_df["ppg_std"].to_numpy(),
        missed_rate=pool_df["missed_rate"].to_numpy(),
        no_track_record=pool_df["no_track_record"].to_numpy(),
        hype=pool_df["hype"].to_numpy(), trend=pool_df["trend"].to_numpy())
    available = np.arange(4)
    np.testing.assert_allclose(
        _live_features(sim, available, 9, {"RB": 1}, ["WR", "RB"], S),
        feature_matrix(obs, S))
