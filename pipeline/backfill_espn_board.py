"""Give the drafts already in the corpus the ESPN board they were drafted from.

WHY THIS EXISTS. `draft_log_pool` gained `espn_rank`, `espn_proj` and `bye`
(see `draft_log.ensure_schema` for what they are for). The farm fills them at
record time from the board it already built for that draft; every draft
recorded BEFORE that landed has NULL in all three, and the new model reads
ESPN's rank as its primary reference. Sixty-six drafts of picks that the model
could not price against the list their drafters were reading is most of the
corpus.

WHY BACKFILLING THESE IS HONEST AND BACKFILLING TIME-TO-PICK WOULD NOT BE.
**The values written here are TODAY's board, not the board as of the draft.**
That is stated in the code, not just in a commit message, because it is the
one thing a later reader must know about these rows. It is defensible for
exactly these three columns and for no others:

  - `espn_rank` and `espn_proj` are ESPN's PRESEASON ranking and projection.
    They are published once and move very little through the preseason; the
    whole corpus was farmed inside a few days of it.
  - `bye` is a fixture-list fact for the season and does not move at all.
  - The measurement that motivated the redesign (docs/superpowers/specs/
    2026-08-23-best-opponent-model-design.md) already did this join, from
    today's board onto every recorded pick, and reported 95% coverage and
    23.1% top-1 for "take the top of ESPN's list". The numbers this backfill
    writes are the numbers that measurement was computed on.

`seconds_to_pick` gets no such treatment and never will: nothing on disk
records when a frame arrived for a draft that is already over, so those rows
stay NULL. A backfilled duration would be an invention, and the column exists
precisely to tell a fast pick from a slow one.

THE FIRST UPDATE THIS CORPUS HAS EVER HAD, and the care around it is
proportionate to that. Every other write to these tables is
delete-then-insert of one whole draft inside one transaction (see
`draft_log.record`), because the corpus is the one thing in this project that
cannot be rebuilt from a source that still exists. So:

  - SCOPED. `WHERE espn_rank IS NULL` -- a row that already has a rank is
    never rewritten, so a draft the farm recorded properly keeps the board it
    was actually played from rather than today's.
  - IDEMPOTENT. Re-running touches nothing: every row the reference can match
    now has a rank and no longer meets the scope. The rows it cannot match
    (ESPN publishes 500 players, the pool is larger) stay NULL and are
    counted and reported rather than filled with a guess.
  - ONE DRAFT PER CONNECTION, and the connection is closed between them.
    DuckDB is single-writer per FILE across processes and three farm sessions
    are playing rooms right now; each of them needs that write lock for the
    one INSERT at the end of a forty-minute draft. Holding it open across a
    whole backfill would make losing one of those drafts possible. Every
    open retries while the lock is held by somebody else, for the same
    reason and with the same patience as `mock_farm.record_draft`.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from pipeline import draft_log as dl

# Same pair of numbers, and the same argument for them, as
# `mock_farm.CORPUS_LOCK_WAIT_SECONDS`: a farm holds the corpus lock for the
# length of one INSERT, so anything measured in minutes is generous -- and
# giving up early on a shared file is how a job like this ends up half done.
LOCK_WAIT_SECONDS = 180.0
LOCK_RETRY_SECONDS = 5.0


def espn_reference(board_conn, settings=None) -> pd.DataFrame:
    """`player_id` -> (espn_rank, espn_proj, bye), from today's board.

    Built by `mock_farm.espn_pool_columns`, the SAME function the farm now
    writes these columns with, so a backfilled row and a farmed row mean the
    same thing in the same column. Two implementations of "ESPN's rank for
    this player" would be two different answers.

    Filtered to players ESPN actually ranks, and that filter is what makes
    the UPDATE idempotent: a row matched by this reference always comes out
    with a non-null `espn_rank` and therefore never meets the scope again. A
    player ESPN does not rank contributes no row at all rather than a row of
    nulls, so "we could not match him" stays visible in the count instead of
    being rewritten every run.
    """
    from scoring.board import build_board

    from pipeline.mock_farm import espn_pool_columns

    board = build_board(board_conn, settings=settings)
    ref = espn_pool_columns(board, board_conn)
    # The corpus stores `player_id` as VARCHAR; the board's is whatever
    # nflverse handed us. Cast on this side, once, so the join below is a
    # string join on both sides rather than a silent no-match.
    ref = ref.assign(player_id=ref["player_id"].astype(str))
    ref = ref[ref["espn_rank"].notna()]
    return ref.drop_duplicates("player_id", keep="first").reset_index(drop=True)


def _with_corpus(path: str, work, out=print, wait: float = LOCK_WAIT_SECONDS):
    """Open the corpus, do ONE short piece of work, close it again.

    The open is what blocks when somebody else holds the file (DuckDB refuses
    the connection outright rather than queueing on the statement), so the
    retry belongs here rather than around the UPDATE. Two different holders
    cause it and both are normal on this machine: a farm taking the write
    lock for the one INSERT that ends a draft, and a long-running READER --
    `fit_archetypes` keeps the corpus open across a multi-hour sweep, and a
    read lock blocks a writer just as firmly. Anything that is not a lock
    failure is re-raised immediately: a corrupt file or a missing table is
    not something waiting fixes.
    """
    import duckdb

    deadline = time.monotonic() + wait
    said = False
    while True:
        try:
            conn = duckdb.connect(path)
        except Exception as exc:                          # noqa: BLE001
            if "lock" not in str(exc).lower():
                raise
            if time.monotonic() > deadline:
                raise
            if not said:
                out(f"  {path} is locked by another process (a farm recording "
                    f"a draft, or a long reader) -- retrying every "
                    f"{LOCK_RETRY_SECONDS:.0f}s")
                said = True
            time.sleep(LOCK_RETRY_SECONDS)
            continue
        try:
            return work(conn)
        finally:
            conn.close()


def _pending_drafts(conn) -> list:
    """Every draft with at least one pool row still missing an ESPN rank.

    `ensure_schema` first, because on a corpus written before these columns
    existed the query below would fail on the column itself -- and the ALTER
    is idempotent DDL that is wanted whether or not anything needs filling.
    """
    dl.ensure_schema(conn)
    rows = conn.execute(
        "SELECT DISTINCT draft_id FROM draft_log_pool "
        "WHERE espn_rank IS NULL ORDER BY draft_id").fetchall()
    return [r[0] for r in rows]


def _fill_one(conn, draft_id: str, reference: pd.DataFrame) -> int:
    """Fill one draft's pool rows. Returns how many rows gained a rank.

    Wrapped in its own transaction even though it is a single statement, so
    that the failure mode is "this draft is unchanged" rather than "this
    draft is half filled" -- the same discipline `draft_log.record` applies,
    for the same reason.
    """
    conn.register("_espn_ref", reference)
    try:
        conn.execute("BEGIN TRANSACTION")
        n = conn.execute("""
            UPDATE draft_log_pool AS p
               SET espn_rank = r.espn_rank,
                   espn_proj = r.espn_proj,
                   bye = r.bye
              FROM _espn_ref AS r
             WHERE p.player_id = r.player_id
               AND p.draft_id = ?
               AND p.espn_rank IS NULL""", [draft_id]).fetchone()[0]
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.unregister("_espn_ref")
    return int(n)


def _still_null(conn) -> int:
    dl.ensure_schema(conn)
    return int(conn.execute("SELECT count(*) FROM draft_log_pool "
                            "WHERE espn_rank IS NULL").fetchone()[0])


def backfill(corpus_path: str, reference: pd.DataFrame, out=print) -> dict:
    """Fill `espn_rank`/`espn_proj`/`bye` on every pool row still missing them.

    Returns the two counts that matter and prints them: how many rows were
    filled, and how many could not be matched to a player ESPN ranks. The
    second number is not an error -- ESPN publishes 500 players and a draft
    pool is larger, so a floor of unmatched rows is expected -- but it is the
    number that would go wrong first if the join key ever drifted, so it is
    reported every run rather than inferred.
    """
    drafts = _with_corpus(corpus_path, _pending_drafts, out)
    if not drafts:
        out("  nothing to fill: every pool row already has an ESPN rank")
        return {"drafts": 0, "filled": 0, "unmatched": _with_corpus(
            corpus_path, _still_null, out)}

    out(f"  {len(drafts)} drafts have pool rows without an ESPN rank; "
        f"the reference board holds {len(reference)} ranked players")
    filled = 0
    for i, draft_id in enumerate(drafts, 1):
        n = _with_corpus(corpus_path,
                         lambda conn, d=draft_id: _fill_one(conn, d, reference),
                         out)
        filled += n
        out(f"  [{i}/{len(drafts)}] {draft_id}: filled {n} pool rows")
    unmatched = _with_corpus(corpus_path, _still_null, out)
    out(f"  filled {filled} pool rows across {len(drafts)} drafts; "
        f"{unmatched} rows left NULL (no ESPN ranking for that player)")
    return {"drafts": len(drafts), "filled": filled, "unmatched": unmatched}


def default_board_db() -> str | None:
    """The most recently written league database, or None if there are none.

    A LEAGUE file rather than `data/nfl.duckdb`, and that is not arbitrary:
    the dev API holds a read-write connection to the universal database for
    as long as it runs, and DuckDB will not let a second process in -- not
    even read-only. Every league file carries its own copy of the shared
    reference tables (see `pipeline.leagues.provision`), `espn_adp` among
    them, so any of them builds the same board. Newest first, because that
    copy is the one most likely to hold the latest refresh.
    """
    files = sorted(Path("data/leagues").glob("*.duckdb"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return str(files[0]) if files else None


def main(argv: list) -> int:
    """Backfill the corpus from today's board.

    Run: python -m pipeline.backfill_espn_board [league_db] [corpus_path]
    """
    from pipeline.mock_farm import open_board_db

    board_path = argv[1] if len(argv) > 1 else default_board_db()
    corpus_path = argv[2] if len(argv) > 2 else dl.CORPUS_PATH
    if not board_path:
        print("no league database under data/leagues/ to build a board from")
        return 1
    print(f"building today's board from {board_path}")
    # `open_board_db` rather than a bare connect: it falls back to a snapshot
    # copy when something else holds the file's lock, which is the normal
    # state of affairs on a machine running the API and three farms.
    board_conn = open_board_db(board_path)
    try:
        reference = espn_reference(board_conn)
    finally:
        board_conn.close()
    print(f"reference: {len(reference)} players with an ESPN rank")
    print(f"backfilling {corpus_path} -- THESE ARE TODAY'S VALUES, not the "
          "board as of each draft (see this module's docstring)")
    counts = backfill(corpus_path, reference)
    print(f"filled {counts['filled']} pool rows; "
          f"{counts['unmatched']} unmatched")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv))
