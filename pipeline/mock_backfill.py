"""Harvest the ESPN mock drafts already sitting on disk into the cross-league
draft corpus.

WHY THIS EXISTS. The per-manager pick model's cold-start prior
(`scoring.draft_model.COLD_START_PRIOR`) was fitted on one league's own
imported history -- 696 picks, all from the same eight people. `data/leagues/`
already holds 45 completed ESPN mock drafts, each an 8-team room of total
strangers, worth roughly 5,760 more picks of "how does the market draft"
signal that has never reached `pipeline.draft_log`'s corpus. This module is
the one-time harvest -- and, because `draft_log.record` is idempotent by
`draft_id`, the safe-to-rerun harvest -- that turns those dormant files into
corpus rows a later refit can read.

THAT REFIT HAS NOW HAPPENED. `pipeline/fit_prior.py` (`make fit-prior`) fitted
a new prior on 26 of these drafts and it beat the incumbent on held-out top-1
by 3.98pp, so `scoring/mock_prior.py` holds coefficients fitted on the rows
this module writes. Every draft added here is evidence the next refit reads;
the numbers, and what they do not establish, are in
docs/superpowers/findings/2026-08-23-mock-corpus-features.md.

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
lobby is an 8-team room. So a file is accepted only when its `drafted`
table's `pick_no` column is EXACTLY the contiguous set `1..MOCK_TEAMS *
MOCK_ROUNDS`, once each -- not merely `count(*) == max(pick_no) == 128`,
which a duplicated `pick_no` with a compensating gap elsewhere (two rows at
50, none at 51) satisfies just as well. Anything that fails that -- an
abandoned connect, a lobby that never filled, a corrupted table -- is
skipped rather than guessed at. `MOCK_TEAMS * MOCK_ROUNDS == 128` is
asserted below at import time, separately, so the two facts this module
rests on -- the constant and the shape the data files were independently
confirmed to run -- cannot silently drift apart from each other.

DRAFT IDENTITY IS CONTENT, NOT A TIMESTAMP. An earlier version of this
module derived `draft_id` from the file's mtime, on the reasoning that two
mocks played in the same league on different nights must not collide onto
one id. That broke on real data: several of these files carry a `.wal`
sidecar with a LATER mtime than the main file (DuckDB replays the WAL on
open, so a read-only connection already sees the true, complete picks, but
the main file itself is not yet checkpointed), and the first thing that
opens such a file read-write and checkpoints it moves the mtime forward for
a draft that never changed. Since the old id folded that timestamp straight
into a hash, the very next backfill computed a NEW id for the SAME draft and
inserted its 128 picks a second time -- silent corpus duplication, exactly
what `draft_id_for`'s idempotency exists to prevent. `_content_draft_id`
below fixes this by hashing the picks themselves (sorted by `pick_no`) plus
the league id: a draft that has not changed produces the same id forever,
regardless of mtimes, checkpoints, or which machine ran the backfill.
`_reconcile_stale_ids` is the one-time-turned-permanent cleanup this
required -- see its own docstring for why it is safe to run on every pass
even with a second, concurrent process writing real drafts into the same
corpus.

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

THIS MODULE ALSO CONTAINS `corpus_report`, the read-only description of what
the harvest (and `pipeline.mock_farm`, writing the same tables) has produced
so far -- see that function's own docstring for why it exists and how it
reads `my_slot`. Backfilling grows the corpus; reporting is how anyone finds
out, before fitting anything on it, whether what grew is people or ESPN's
autodraft repeating its own ranking back at itself.
"""
import dataclasses
import glob
import hashlib
import shutil
import tempfile
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


def _content_draft_id(league_id: str, drafted: pd.DataFrame) -> str:
    """A draft's identity, derived from the picks themselves rather than
    from when a filesystem last touched the file.

    See the module docstring's "DRAFT IDENTITY IS CONTENT, NOT A TIMESTAMP"
    for why `os.path.getmtime` cannot be trusted for this: a WAL checkpoint
    alone can move a file's mtime forward for a draft that never changed,
    which silently duplicated 128 picks on the next run. Hashing the picks
    instead means a draft that has not changed produces the same id
    forever, on any machine, checkpointed or not.

    `league_id` is folded in, matching `draft_id_for`'s own reasoning for
    every other source: without it, two DIFFERENT leagues whose mocks
    happened to draft the same 128 players in the same order -- a real
    possibility when every room's bots lean on similar ADP -- would
    collide onto one id. `pick_no` is sorted first so row order in
    `drafted` (a SELECT with no guaranteed physical ordering) can never
    change the digest for the same actual draft.

    Same `{source}:{16 hex}` shape `draft_log.draft_id_for` produces, so
    nothing that reads a draft_id downstream needs to know this one was
    built differently.
    """
    ordered = drafted.sort_values("pick_no")
    payload = "|".join(f"{pid}:{pick}" for pid, pick in
                       zip(ordered["player_id"], ordered["pick_no"]))
    digest = hashlib.sha1(f"{league_id}|{payload}".encode()).hexdigest()[:16]
    return f"{dl.SOURCE_MOCK}:{digest}"


def _reconcile_stale_ids(corpus, league_id: str, keep_draft_id: str) -> int:
    """Delete backfilled rows left behind by a since-changed draft_id for
    this same league file, and report how many were removed.

    RUNS ON EVERY BACKFILL, not as a one-off migration script -- so a corpus
    holding rows written under a previous, less stable identity scheme (the
    mtime-derived one `_content_draft_id`'s docstring describes) self-heals
    the next time `make mock-backfill` touches that same file, and so does
    any future identity change, without anyone having to remember to run a
    separate cleanup.

    THE DISCRIMINATOR IS DELIBERATELY NARROW, because this corpus is not
    this module's alone: a separate, concurrent process farms live mock
    drafts into the very same `source='mock'` rows, and those are NOT
    reproducible from anything on disk here -- deleting one loses it
    permanently. So this never does a blanket delete of `source='mock'`; it
    deletes only rows matching ALL of:
      * `source = 'mock'`           -- never another source's history
      * `my_slot IS NULL`           -- a row THIS module writes never sets a
        real seat (see the module docstring's "WHAT IS DELIBERATELY LEFT
        NULL"); a live-farmed row does, by construction, which is exactly
        what keeps it out of reach of this delete
      * `league_id = league_id`     -- only the file being re-recorded right
        now, never another file's history
      * `draft_id != keep_draft_id` -- never the id this very call just
        wrote
    Deleted from all three corpus tables, mirroring `draft_log.record`'s own
    delete-then-insert -- and, like every writer in this module, never via
    DROP or CREATE OR REPLACE.
    """
    stale = corpus.execute(
        """SELECT draft_id FROM draft_log
           WHERE source = ? AND my_slot IS NULL AND league_id = ?
             AND draft_id != ?""",
        [dl.SOURCE_MOCK, league_id, keep_draft_id]).df()["draft_id"].tolist()
    if not stale:
        return 0
    placeholders = ",".join(["?"] * len(stale))
    for table in ("draft_log_pick", "draft_log_pool", "draft_log"):
        corpus.execute(
            f"DELETE FROM {table} WHERE draft_id IN ({placeholders})", stale)
    return len(stale)


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
        expected_picks = MOCK_TEAMS * MOCK_ROUNDS
        count = len(drafted)
        max_pick = int(drafted["pick_no"].max()) if count else 0
        # Accept only a draft whose pick_no is EXACTLY the contiguous set
        # 1..expected_picks, once each -- do not guess at anything else. See
        # the module docstring's "THE 8x16 ASSUMPTION" for why 128, and for
        # why `count(*) == max(pick_no) == 128` alone is not this check: a
        # duplicated pick_no with a compensating gap elsewhere satisfies
        # both those aggregates too. `count == expected_picks` together with
        # the pick_no SET equalling {1..expected_picks} rules that out --
        # expected_picks rows collapsing to exactly expected_picks distinct
        # required values leaves no room for a duplicate anywhere.
        complete = (count == expected_picks and
                   set(drafted["pick_no"]) == set(range(1, expected_picks + 1)))
        if not complete:
            return {"status": "incomplete", "picks": count, "max_pick": max_pick}

        league_id = Path(path).stem
        # Content, not the file's mtime -- see the module docstring's
        # "DRAFT IDENTITY IS CONTENT, NOT A TIMESTAMP" for the duplication
        # bug mtime caused and why this is the fix.
        draft_id = _content_draft_id(league_id, drafted)

        pool_df = _pool_frame(conn, settings)
        picks_df, dropped = _picks_frame(drafted, draft_id, pool_df)

        dl.record(corpus, dl.DraftRecord(
            source=dl.SOURCE_MOCK, league_id=league_id, season=CURRENT_SEASON,
            teams=MOCK_TEAMS, rounds=MOCK_ROUNDS, my_slot=None,
            # `settings_json` carries the roster SHAPE (starters/flex/bench);
            # it does not separately carry `rounds` -- `LeagueSettings.
            # rounds` is a `@property` derived from that shape, and
            # `league.to_json`'s `dataclasses.asdict` only serializes
            # declared fields, so there was never a `rounds` key for it to
            # drop. That is legibility, not loss: the literal value is
            # `draft_log.rounds` right here, a real column (alongside
            # `draft_log.teams`) on the very row this JSON rides along with.
            settings_json=league.to_json(settings),
            picks=picks_df, pool=pool_df, draft_id=draft_id))
        stale_removed = _reconcile_stale_ids(corpus, league_id, draft_id)
        return {"status": "recorded", "picks": len(picks_df), "dropped": dropped,
               "stale_removed": stale_removed}
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
        "stale_rows_removed": 0,
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
            counts["stale_rows_removed"] += result["stale_removed"]
    return counts


def _print_summary(counts: dict) -> None:
    print(f"files scanned:                 {counts['files_scanned']}")
    print(f"  open failed (locked/unreadable, skipped): {counts['open_failed']}")
    print(f"  no drafted rows (skipped):    {counts['no_rows']}")
    print(f"  incomplete draft (skipped):   {counts['incomplete']}")
    print(f"drafts recorded:               {counts['drafts_recorded']}")
    print(f"picks recorded:                {counts['picks_recorded']}")
    print(f"picks dropped (no pool match): {counts['picks_dropped']}")
    print(f"stale rows reconciled (old draft_id, same file): {counts['stale_rows_removed']}")


# Round-bucket boundaries, by ROUND NUMBER -- not by team count or by how
# many rounds any one draft ran. "Early" is the top of the board where
# consensus lives; "late" is where K/DST and pure preference take over. See
# `corpus_report`'s docstring for why the round NUMBER these buckets sort on
# is computed per draft from `draft_log.teams` rather than assumed.
_EARLY_MAX_ROUND = 3   # rounds 1-3
_MID_MAX_ROUND = 10     # rounds 4-10; anything above is "late" (11+)


def _round_bucket(round_no: int) -> str:
    """Classify an already-computed round number. Does not touch teams or
    pick_no itself -- see `corpus_report` for where round comes from."""
    if round_no <= _EARLY_MAX_ROUND:
        return "early"
    if round_no <= _MID_MAX_ROUND:
        return "mid"
    return "late"


def _deviation_stats(df: "pd.DataFrame") -> dict:
    """Mean and median |pick_no - adp_rank| over the rows in `df` that HAVE
    an adp_rank.

    A pick the ADP join never matched (see `_picks_frame`'s own "dropped"
    counting for the backfill-time version of this same problem) is simply
    excluded here, not coerced to a 0 deviation or a missing row silently
    dropped from the corpus -- `n` is returned alongside so a reader can see
    how much of the mean/median is actually resting on real matches. `df`
    empty, or nothing in it with a known adp_rank, reports `n=0` and `None`
    for both stats rather than letting pandas hand back a silent NaN.
    """
    known = df[df["adp_rank"].notna()]
    dev = (known["pick_no"] - known["adp_rank"]).abs()
    return {
        "n": int(len(dev)),
        "mean": float(dev.mean()) if len(dev) else None,
        "median": float(dev.median()) if len(dev) else None,
    }


def _first_round_positions(df: "pd.DataFrame") -> dict:
    """Position counts at round 1 -- the corpus-gate-finding's check that a
    broken ADP join could not fake: a near-even RB/WR split with QB
    essentially absent is a real strategic distribution, not a ranking read
    off in pick order. Unlike `_deviation_stats`, this does not filter on
    adp_rank -- a pick's position is known whether or not the ADP join
    matched it.
    """
    first = df[df["round"] == 1]
    return {pos: int(n) for pos, n in first["position"].value_counts().items()}


def _position_share_by_bucket(df: "pd.DataFrame") -> dict:
    """Position mix within each round bucket, as a percentage of THAT
    bucket's own picks -- not of the whole corpus, so early/mid/late can be
    compared on equal footing even though they hold different pick counts.

    Kickers and defenses living almost entirely in the late bucket is
    exactly the late-round structure real drafting produces and a corpus of
    pure autodraft would not (autodraft has no reason to wait on K/DST any
    more than on anyone else). An empty bucket reports `{}` rather than
    raising on `value_counts` of nothing.
    """
    out = {}
    for bucket in ("early", "mid", "late"):
        sub = df[df["bucket"] == bucket]
        if sub.empty:
            out[bucket] = {}
            continue
        share = sub["position"].value_counts(normalize=True) * 100
        out[bucket] = {pos: round(float(pct), 1) for pos, pct in share.items()}
    return out


def _population_stats(picks_df: "pd.DataFrame", heads_df: "pd.DataFrame") -> dict:
    """Every number this report computes, for ONE population of mock drafts
    (backfilled, live-farmed, or the two pooled) -- see `corpus_report` for
    why the population split matters and what `picks_df`/`heads_df` must
    already carry (`round`, `bucket` columns on `picks_df`; both frames
    already filtered to the one population).

    `heads_df` (rows of `draft_log`), not `picks_df["draft_id"].nunique()`,
    is the draft count -- deliberately: a recorded draft that somehow ended
    up with zero picks would silently vanish from a picks-derived count
    while still being a real row in the corpus.
    """
    n_picks = int(len(picks_df))
    adp_null = int(picks_df["adp_rank"].isna().sum())
    autodraft_known = picks_df["autodrafted"].notna()
    n_known = int(autodraft_known.sum())
    n_true = int((picks_df["autodrafted"] == True).sum())  # noqa: E712
    return {
        "drafts": int(heads_df["draft_id"].nunique()),
        "picks": n_picks,
        "adp_null": adp_null,
        "adp_known": n_picks - adp_null,
        "autodrafted_known": n_known,
        "autodrafted_unknown": n_picks - n_known,
        "autodrafted_true": n_true,
        "autodrafted_share": (n_true / n_known) if n_known else None,
        "deviation": _deviation_stats(picks_df),
        "by_bucket": {b: _deviation_stats(picks_df[picks_df["bucket"] == b])
                     for b in ("early", "mid", "late")},
        "first_round_positions": _first_round_positions(picks_df),
        "position_share_by_bucket": _position_share_by_bucket(picks_df),
    }


def corpus_report(conn) -> dict:
    """Describe what `conn` currently holds: drafts and picks by source
    (`pipeline.draft_log.summary`), then -- for the mock corpus specifically
    -- whether pick_no tracks adp_rank closely (ESPN's autodraft, following
    its own fixed ranking) or diverges from it the way a room of people
    does. This is the corpus-gate-finding's one-off analysis
    (`.superpowers/sdd/2026-08-23-mock-draft-corpus-plan/
    corpus-gate-finding.md`), turned into something that can be rerun as the
    corpus grows rather than computed by hand again.

    WHY THE MOCK CORPUS IS SPLIT INTO TWO POPULATIONS, NOT READ AS ONE.
    `pipeline.mock_backfill` and `pipeline.mock_farm` both write
    `source='mock'` rows into the same three tables, but they are not the
    same kind of draft. A backfilled draft (`my_slot IS NULL`) came from a
    `drafted` table that never recorded which picks, if any, were ESPN's
    engine rather than a person -- `autodrafted` is NULL on every one of its
    picks, unknown, not "no". A live-farmed draft (`my_slot IS NOT NULL`)
    is one this tool's own bot sat in, and ESPN told the listener, per pick,
    whether that pick was autodrafted -- ground truth, not inference.
    Averaging the two populations together would let a farmed draft's KNOWN
    autodraft picks and a backfilled draft's UNKNOWN ones blend into one
    number that looks more certain than either population actually is on
    its own, which is exactly the thing this report exists to keep visible
    rather than hide. So every stat below is computed three times --
    `backfilled`, `live_farmed`, and `combined` (the straight pool of both)
    -- where `combined` is what reproduces the original finding's headline
    number, and the population split is what tells a reader whether that
    number is stable across two different kinds of draft or an artifact of
    whichever one currently dominates the corpus.

    ROUND IS COMPUTED PER DRAFT FROM `draft_log.teams`, deliberately not
    hardcoded to `MOCK_TEAMS` (8). Every backfilled draft happens to be
    8-team (see this module's own "THE 8x16 ASSUMPTION"), but a live-farmed
    draft's team count is read off ESPN's own room settings by
    `pipeline.mock_farm` and can differ. A hardcoded 8 would silently
    mis-bucket every pick of a non-8-team farmed draft into the wrong round
    -- not fail loudly, just quietly lie about which bucket a pick belongs
    to -- which is worse than any error this function could raise instead.
    """
    # NOT `dl.summary(conn)` -- it calls `ensure_schema`, and DuckDB refuses
    # to run ANY statement of type CREATE against a read-only-attached
    # database, even a no-op `CREATE TABLE IF NOT EXISTS` against a schema
    # that already matches. A read-only connection is exactly what
    # `report_main` hands this function (see `_open_corpus_read_only`), so
    # this is `summary`'s own query, copied rather than called, with the
    # `ensure_schema` call it opens with left out.
    by_source = conn.execute("""
        SELECT d.source, count(DISTINCT d.draft_id) AS drafts,
               count(p.pick_no) AS picks,
               count(DISTINCT CASE WHEN NOT p.is_anonymous THEN p.owner_key END)
                   AS known_owners
        FROM draft_log d LEFT JOIN draft_log_pick p USING (draft_id)
        GROUP BY d.source ORDER BY d.source""").df()

    heads = conn.execute(
        "SELECT draft_id, teams, my_slot FROM draft_log WHERE source = ?",
        [dl.SOURCE_MOCK]).df()
    picks = conn.execute(
        """SELECT p.draft_id, p.pick_no, p.position, p.adp_rank,
                  p.autodrafted, d.teams, d.my_slot
           FROM draft_log_pick p JOIN draft_log d USING (draft_id)
           WHERE d.source = ?""", [dl.SOURCE_MOCK]).df()

    # `teams` should never be null on a mock row -- both writers set it (see
    # this function's own "ROUND IS COMPUTED..." above) -- but a round
    # computed against a null divisor would raise and take the whole report
    # down over one bad row, rather than reporting everything else. Excluded
    # and counted instead, the same discipline `_picks_frame` uses for a
    # pick the pool join can't match.
    unknown_teams = int(picks["teams"].isna().sum())
    picks = picks[picks["teams"].notna()].copy()
    picks["round"] = ((picks["pick_no"] - 1) // picks["teams"] + 1).astype(int)
    picks["bucket"] = picks["round"].map(_round_bucket)

    backfilled = picks["my_slot"].isna()
    heads_backfilled = heads["my_slot"].isna()

    return {
        "by_source": by_source,
        "unknown_teams_excluded": unknown_teams,
        "mock": {
            "combined": _population_stats(picks, heads),
            "backfilled": _population_stats(
                picks[backfilled], heads[heads_backfilled]),
            "live_farmed": _population_stats(
                picks[~backfilled], heads[~heads_backfilled]),
        },
    }


def _fmt_stat(stats: dict) -> str:
    if stats["n"] == 0:
        return "n=0 (no adp-matched picks)"
    return f"n={stats['n']}, mean={stats['mean']:.2f}, median={stats['median']:.2f}"


def _print_population(label: str, pop: dict) -> None:
    print(f"\n-- {label} --")
    print(f"drafts: {pop['drafts']}   picks: {pop['picks']}")
    if pop["picks"] == 0:
        print("  (no picks recorded for this population yet)")
        return
    null_share = pop["adp_null"] / pop["picks"] * 100
    print(f"adp_rank unknown: {pop['adp_null']} of {pop['picks']} picks "
          f"({null_share:.1f}%) -- excluded from every deviation number below")
    if pop["autodrafted_known"]:
        share = pop["autodrafted_share"] * 100
        print(f"autodrafted (known): {pop['autodrafted_true']} of "
              f"{pop['autodrafted_known']} flagged picks ({share:.1f}%); "
              f"{pop['autodrafted_unknown']} picks unknown")
    else:
        print(f"autodrafted: unknown for all {pop['autodrafted_unknown']} "
              f"picks (this population carries no per-pick flag)")
    print(f"mean |pick_no - adp_rank|: {_fmt_stat(pop['deviation'])}")
    print("  by round bucket:")
    for bucket in ("early", "mid", "late"):
        print(f"    {bucket:5s}: {_fmt_stat(pop['by_bucket'][bucket])}")
    if pop["first_round_positions"]:
        mix = ", ".join(f"{pos} {n}" for pos, n in
                        sorted(pop["first_round_positions"].items(),
                               key=lambda kv: -kv[1]))
        print(f"  first-round positions: {mix}")
    for bucket in ("early", "mid", "late"):
        share = pop["position_share_by_bucket"].get(bucket)
        if not share:
            continue
        row = ", ".join(f"{pos} {pct:.1f}%" for pos, pct in
                        sorted(share.items(), key=lambda kv: -kv[1]))
        print(f"  {bucket} position share: {row}")


def _print_report(report: dict) -> None:
    print("=== corpus contents, by source ===")
    print(report["by_source"].to_string(index=False))
    if report["unknown_teams_excluded"]:
        print(f"\n({report['unknown_teams_excluded']} mock picks excluded "
              f"from every round-bucketed stat below -- draft_log.teams was "
              f"NULL for their draft)")

    combined = report["mock"]["combined"]
    print("\n=== the headline number ===")
    print(f"mean |pick_no - adp_rank| across all mock picks: "
          f"{_fmt_stat(combined['deviation'])}")
    print(
        "\nHow to read that: ESPN's autodraft works down a fixed ranking, so "
        "a corpus dominated by it would sit near zero -- pick_no and "
        "adp_rank would coincide almost every time, and a prior fitted on "
        "it would just relearn the market instead of learning how people "
        "deviate from it. A large, round-shaped number -- tight near the "
        "top of the board where consensus is tight, wide by the late "
        "rounds where preference and sleepers pull picks away from rank -- "
        "is the signature of real drafting instead. Some part of a nonzero "
        "number is still source disagreement (ESPN's own autodraft ranking "
        "is not this corpus's adp_rank, not the same list); the early-round "
        "figure below, where market consensus is tightest, bounds how large "
        "that part can be without eliminating it as a possibility."
    )

    print("\n=== mock corpus, by population ===")
    print(
        "Backfilled and live-farmed mock drafts are different populations "
        "-- see corpus_report's own docstring -- so every number below is "
        "given for each separately as well as pooled. `combined` above and "
        "below is what reproduces the original one-off finding's headline "
        "figure."
    )
    _print_population("combined (backfilled + live-farmed)", combined)
    _print_population("backfilled (my_slot IS NULL, autodrafted unknown)",
                      report["mock"]["backfilled"])
    _print_population("live-farmed (my_slot IS NOT NULL, autodrafted known)",
                      report["mock"]["live_farmed"])


def _open_corpus_read_only(path: str | None = None, out=print):
    """A read-only connection to the draft corpus, for reporting only.

    Read-only is not a style preference here: `pipeline.draft_log`'s module
    docstring is explicit that this corpus is the one thing this tool
    cannot rebuild if damaged -- a mock draft that is not written down is
    gone for good -- and a report has no legitimate reason to hold write
    access to describe what is already there. `duckdb.connect(...,
    read_only=True)` enforces that at the connection level rather than
    trusting this module to simply never call INSERT/DELETE.

    THE FARM LOOP MAKES THIS CONTENTIOUS, NOT JUST CAUTIOUS. `pipeline.
    mock_farm` can hold a read-write connection to this same file for the
    length of a live draft, and DuckDB is single-writer per file -- it does
    not let a reader in while a writer holds the lock (see `mock_farm.
    open_board_db`'s own comment, hitting the identical problem against
    `data/nfl.duckdb`). A report that simply failed whenever the farm loop
    was mid-write would be unusable exactly while the corpus is most
    interesting to look at. So, like `open_board_db`, a lock error falls
    back to a read-only connection against a plain file copy -- the report
    may be a few picks stale, and it says so, but it runs.
    """
    target = path or dl.CORPUS_PATH
    try:
        return duckdb.connect(target, read_only=True)
    except Exception as exc:                    # noqa: BLE001
        if "lock" not in str(exc).lower():
            raise
        snapshot = str(Path(tempfile.mkdtemp(prefix="corpus-report-")) /
                       Path(target).name)
        out(f"{target} is locked (the farm loop is likely writing) -- "
            f"reading a snapshot copy at {snapshot} instead; this report "
            f"may be a few picks stale")
        shutil.copy2(target, snapshot)
        return duckdb.connect(snapshot, read_only=True)


def report_main(argv: list) -> int:
    """Print `corpus_report` for the real corpus (or `argv[0]` if given).

    A missing corpus file is reported plainly rather than as a traceback --
    it is the ordinary state of things before the first `make mock-backfill`
    or `make farm-mocks` has ever run, not a bug in this report.
    """
    target = argv[0] if argv else dl.CORPUS_PATH
    if not Path(target).exists():
        print(f"no corpus at {target} -- nothing to report yet (run "
              f"`make mock-backfill` or `make farm-mocks` first)")
        return 0
    conn = _open_corpus_read_only(target)
    try:
        report = corpus_report(conn)
    finally:
        conn.close()
    _print_report(report)
    return 0


def main(argv: list) -> int:
    """Harvest every complete ESPN mock draft in `data/leagues/*.duckdb`
    into the cross-league draft corpus, and report what it found.

    Run: python -m pipeline.mock_backfill [glob]
         python -m pipeline.mock_backfill --report [corpus_path]

    `--report` prints `corpus_report` instead of running the harvest -- a
    read-only description of what the corpus (this module's writes AND
    `pipeline.mock_farm`'s) currently holds, not a write of any kind. See
    `corpus_report`'s own docstring for what it describes.
    """
    if len(argv) > 1 and argv[1] == "--report":
        return report_main(argv[2:])
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
