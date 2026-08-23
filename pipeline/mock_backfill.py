"""Harvest the ESPN mock drafts already sitting on disk into the cross-league
draft corpus.

WHY THIS EXISTS. The per-manager pick model's cold-start prior
(`scoring.draft_model.COLD_START_PRIOR`) is fitted on one league's own
imported history -- 696 picks, all from the same eight people. `data/leagues/`
already holds 45 completed ESPN mock drafts, each an 8-team room of total
strangers, worth roughly 5,760 more picks of "how does the market draft"
signal that has never reached `pipeline.draft_log`'s corpus. This module is
the one-time harvest -- and, because `draft_log.record` is idempotent by
`draft_id`, the safe-to-rerun harvest -- that turns those dormant files into
corpus rows a later refit can read.

WHERE THE FILES COME FROM AND WHAT THEY DO NOT CARRY. Each
`data/leagues/<id>.duckdb` is a per-league database (see `pipeline.db`'s
LEAGUE_TABLES / UNIVERSAL_TABLES split) that a mock-draft session provisioned
and played through ESPN's live mock-draft lobby rather than a real league
import. So it carries the universal reference tables (`weekly`, `adp`,
`espn_adp`, ...) but has no `league` row and an empty `draft_order` -- there
is no stored team count, no manager names, and no record of which seat was
the user. What it does carry is `drafted`: exactly (`player_id`, `pick_no`),
one row per pick ESPN actually made.

THE 8x16 ASSUMPTION. Nothing on disk says how many teams or rounds a given
mock ran. What IS on disk: 23 of the 45 files hold exactly 128 picks with
`max(pick_no) == 128`, and 128 = 8 teams x 16 rounds is independently
confirmed by `data/draft_room_trace.jsonl`, a recording of a real ESPN mock
that shows 128 SELECTED frames and team ids exactly 1..8 -- ESPN's own mock
lobby is an 8-team room. So a file is accepted only when its `drafted` table
is exactly complete at that count; anything else (an abandoned connect, a
lobby that never filled) is skipped rather than guessed at, and
`MOCK_TEAMS * MOCK_ROUNDS == 128` is asserted below so this assumption fails
loudly rather than silently if a differently-shaped draft ever shows up in
this directory.

WHAT THE ROSTER SHAPE ITSELF IS AN ASSUMPTION FOR. A pick count says nothing
about starters/flex/bench, and that is not recoverable either. `_mock_settings`
below picks one: the same starters `league.default_settings()` uses for this
project's own live league (QB/2RB/2WR/TE/2FLEX/K/DST), with the bench widened
from 5 to 6. That widening is necessary and nothing more -- `LeagueSettings.
rounds` is a derived property (`sum(starters) + flex_slots + bench`), the
live league's own 5-bench shape sums to 15, and only a 16-round shape can
absorb 128 picks over 8 teams. This is a guess about a shape nobody recorded,
not a measurement, which is why it rides along in `settings_json` on every
row this module writes rather than staying implicit in the code.

WHAT IS DELIBERATELY LEFT NULL. `my_slot`: nothing in these files records
which seat, if any, was a real user rather than another simulated stranger.
`autodrafted`: a `drafted` table is only ever (`player_id`, `pick_no`) --
whether ESPN's engine made a given pick rather than a person is not a fact
these files carry, so it is not a fact this module can report. Both are
NULL, never a guessed default (see `pipeline.draft_log.ensure_schema`'s
comment on the same column for the live-draft case where it IS known).
"""
import dataclasses
import glob
import os
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from pipeline import draft_log as dl
from scoring import league
from scoring.board import build_board
from scoring.config import CURRENT_SEASON
from scoring.draft_sim import build_pool, snake_slots

# Overridable by a caller (the test suite passes its own tmp_path files
# straight to `backfill`, so nothing here actually reads this constant in
# tests) -- kept as a module constant rather than a literal only so a real
# invocation and a docstring example can both point at one name.
LEAGUE_GLOB = "data/leagues/*.duckdb"

# See the module docstring's "THE 8x16 ASSUMPTION": the one shape a file is
# accepted at, confirmed independently against data/draft_room_trace.jsonl.
MOCK_TEAMS = 8
MOCK_ROUNDS = 16
assert MOCK_TEAMS * MOCK_ROUNDS == 128, (
    "the 128-pick completeness check below and the 8x16 shape it is read as "
    "have drifted apart -- fix one or the other before this runs")


def _mock_settings() -> "league.LeagueSettings":
    """8-team, 16-round settings for a draft whose real roster shape nothing
    on disk records.

    See the module docstring's "WHAT THE ROSTER SHAPE ITSELF IS AN
    ASSUMPTION FOR". Only `bench` is touched -- `LeagueSettings.rounds`
    cannot be set directly, it is derived from starters/flex/bench, so
    reaching 16 rounds from the live league's 15-round shape means widening
    the one field that does not change what a roster actually starts.
    """
    base = league.default_settings()
    settings = dataclasses.replace(base, bench=base.bench + 1)
    assert settings.teams == MOCK_TEAMS and settings.rounds == MOCK_ROUNDS, (
        f"_mock_settings produced {settings.teams}x{settings.rounds}, not "
        f"{MOCK_TEAMS}x{MOCK_ROUNDS} -- default_settings() must have changed "
        "shape out from under this")
    return settings


def _open_read_only(path: str):
    """`None` on any failure to open, rather than letting one bad file take
    the whole backfill down.

    These are the user's live league files. Always opened read-only, never
    written to. A file can fail to open for reasons that have nothing to do
    with whether the draft inside it is any good -- locked by another
    process still using it, or mid-write with only a `.wal` sidecar to show
    for it -- and neither is this module's business to resolve; skip it and
    let the caller count it.
    """
    try:
        return duckdb.connect(path, read_only=True)
    except Exception:
        return None


def _has_rows(conn, table: str) -> bool:
    tables = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
    if table not in tables:
        return False
    return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] > 0


def _pool_frame(conn, settings) -> pd.DataFrame:
    """This draft's pool snapshot, shaped for `_POOL_COLUMNS` (minus
    `draft_id`, which `draft_log.record` fills in).

    `build_board` then `build_pool` -- the expensive step this module pays
    for, measured at roughly 1.5-1.9s and 2.2-3.6s respectively against the
    real data, which is why it only ever runs for a file that has already
    cleared the completeness check below, not once per file scanned.

    `team` is not one of `SimPool`'s fields (see its own NamedTuple in
    scoring.draft_sim) even though it is one of `_POOL_COLUMNS`, so it is
    read back off `board` and joined on `player_id` rather than out of the
    pool the simulator built.
    """
    board = build_board(conn, settings=settings)
    pool = build_pool(conn, board, settings)
    pool_df = pd.DataFrame({
        "player_id": pool.player_id,
        "position": pool.position,
        "adp_rank": pool.adp_rank,
        "proj_points": pool.points,
    })
    team_by_player = (board.drop_duplicates("player_id", keep="first")
                      .set_index("player_id")["team"])
    pool_df["team"] = pool_df["player_id"].map(team_by_player)
    return pool_df


def _picks_frame(drafted: pd.DataFrame, draft_id: str,
                 pool_df: pd.DataFrame) -> tuple:
    """This draft's picks, shaped for `_PICK_COLUMNS`, and how many were
    dropped for a `player_id` the pool join could not find.

    Slot comes from `draft_sim.snake_slots(MOCK_TEAMS, MOCK_ROUNDS)` indexed
    by `pick_no - 1` -- the assumed shape applied to the one thing these
    files do record, pick order. `position`, `adp_rank` and `proj_points`
    come from THIS draft's own pool snapshot, joined on `player_id`: a pick
    the join misses has no position to store it under and is dropped rather
    than written half-filled. It is COUNTED, not silenced -- a silent drop
    here would look exactly like a smaller corpus rather than like the join
    failure it actually is, and nobody reading `draft_log.summary` later
    could tell the difference.
    """
    slots = snake_slots(MOCK_TEAMS, MOCK_ROUNDS)
    picks = drafted.sort_values("pick_no").reset_index(drop=True).copy()
    picks["round"] = ((picks["pick_no"] - 1) // MOCK_TEAMS + 1).astype(int)
    picks["slot"] = [slots[p - 1] for p in picks["pick_no"]]
    # Mock opponents are strangers with no identity that survives the
    # session -- draft_log's own module docstring on ANONYMOUS_PREFIX makes
    # the same argument for why they are stored anyway, flagged rather than
    # dropped: they can never build a personal profile, but they ARE the
    # population baseline this whole harvest exists to grow.
    picks["owner_key"] = [dl.anonymous_key(draft_id, s) for s in picks["slot"]]
    picks["is_anonymous"] = True
    picks = picks.merge(
        pool_df[["player_id", "position", "adp_rank", "proj_points"]],
        on="player_id", how="left")
    missing = picks["position"].isna()
    dropped = int(missing.sum())
    return picks[~missing].reset_index(drop=True), dropped


def _backfill_file(path: str, settings, corpus) -> dict:
    """Fold one league file's completed mock draft into `corpus`, or report
    why it was skipped.

    Never raises for an ordinary skip condition (unopenable file, no picks
    yet, an abandoned draft) -- those are expected outcomes for files in
    this directory, not bugs, and `backfill` needs to keep going past them.
    """
    conn = _open_read_only(path)
    if conn is None:
        return {"status": "open_failed"}
    try:
        if not _has_rows(conn, "drafted"):
            return {"status": "no_rows"}
        drafted = conn.execute(
            "SELECT player_id, pick_no FROM drafted ORDER BY pick_no").df()
        count, max_pick = len(drafted), int(drafted["pick_no"].max())
        # Accept only complete drafts -- do not guess at partial ones. See
        # the module docstring's "THE 8x16 ASSUMPTION" for why 128 and not
        # some other number.
        if count != max_pick or max_pick != MOCK_TEAMS * MOCK_ROUNDS:
            return {"status": "incomplete", "picks": count, "max_pick": max_pick}
        assert MOCK_TEAMS * MOCK_ROUNDS == max_pick, (
            f"{path}: a draft with {max_pick} picks passed the completeness "
            f"check above but does not match the assumed {MOCK_TEAMS}x"
            f"{MOCK_ROUNDS} shape -- that check and this assumption must "
            "have drifted apart")

        league_id = Path(path).stem
        # The file's mtime, not `datetime.now()`: two mocks played in the
        # same league on different nights must not collide onto one
        # draft_id, and mtime is the only timestamp these files carry at
        # all -- there is no `league` row with a start time, and `drafted`
        # itself carries none either.
        started_at = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
        draft_id = dl.draft_id_for(dl.SOURCE_MOCK, league_id, CURRENT_SEASON,
                                   started_at)

        pool_df = _pool_frame(conn, settings)
        picks_df, dropped = _picks_frame(drafted, draft_id, pool_df)

        dl.record(corpus, dl.DraftRecord(
            source=dl.SOURCE_MOCK, league_id=league_id, season=CURRENT_SEASON,
            teams=MOCK_TEAMS, rounds=MOCK_ROUNDS, my_slot=None,
            settings_json=league.to_json(settings), started_at=started_at,
            picks=picks_df, pool=pool_df, draft_id=draft_id))
        return {"status": "recorded", "picks": len(picks_df), "dropped": dropped}
    finally:
        conn.close()


def backfill(paths: list, corpus) -> dict:
    """Fold every complete mock draft among `paths` into `corpus`.

    One `_mock_settings()` for the whole run: the assumption is about the
    shape of a mock draft in general, not about any one file, so there is
    nothing to vary per file.
    """
    settings = _mock_settings()
    counts = {
        "files_scanned": 0, "open_failed": 0, "no_rows": 0, "incomplete": 0,
        "drafts_recorded": 0, "picks_recorded": 0, "picks_dropped": 0,
    }
    for path in paths:
        counts["files_scanned"] += 1
        result = _backfill_file(path, settings, corpus)
        status = result["status"]
        if status in ("open_failed", "no_rows", "incomplete"):
            counts[status] += 1
        elif status == "recorded":
            counts["drafts_recorded"] += 1
            counts["picks_recorded"] += result["picks"]
            counts["picks_dropped"] += result["dropped"]
    return counts


def _print_summary(counts: dict) -> None:
    print(f"files scanned:                 {counts['files_scanned']}")
    print(f"  open failed (locked/unreadable, skipped): {counts['open_failed']}")
    print(f"  no drafted rows (skipped):    {counts['no_rows']}")
    print(f"  incomplete draft (skipped):   {counts['incomplete']}")
    print(f"drafts recorded:               {counts['drafts_recorded']}")
    print(f"picks recorded:                {counts['picks_recorded']}")
    print(f"picks dropped (no pool match): {counts['picks_dropped']}")


def main(argv: list) -> int:
    """Harvest every complete ESPN mock draft in `data/leagues/*.duckdb`
    into the cross-league draft corpus, and report what it found.

    Run: python -m pipeline.mock_backfill [glob]
    `glob`, if given, overrides LEAGUE_GLOB.
    """
    pattern = argv[1] if len(argv) > 1 else LEAGUE_GLOB
    paths = sorted(glob.glob(pattern))
    corpus = dl.corpus_conn()
    try:
        counts = backfill(paths, corpus)
    finally:
        corpus.close()
    _print_summary(counts)
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv))
