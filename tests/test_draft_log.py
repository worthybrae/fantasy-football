"""The draft corpus: every draft ever seen, kept as facts not conclusions.

A mock draft that is not written down when it happens is gone -- there is no
source to re-derive it from, unlike every other table in this project. So the
tests here are mostly about not losing things.
"""
import duckdb
import pandas as pd
import pytest

from pipeline import draft_log as dl


@pytest.fixture
def corpus(tmp_path):
    return dl.corpus_conn(str(tmp_path / "corpus.duckdb"))


def _picks(n=4, owner="espn:1:Amy", anon=False):
    return pd.DataFrame({
        "pick_no": range(1, n + 1),
        "round": [1] * n,
        "slot": range(1, n + 1),
        "owner_key": [owner] * n,
        "is_anonymous": [anon] * n,
        "player_id": [f"p{i}" for i in range(n)],
        "position": ["RB", "WR", "TE", "QB"][:n],
        "adp_rank": [1.0, 2.0, 3.0, 4.0][:n],
        "proj_points": [200.0] * n,
    })


def test_a_league_draft_keeps_its_identity_across_reimports(corpus):
    """Re-importing a season must land on the same row, not a second copy.
    A league's history gets re-imported whenever the user reconnects."""
    a = dl.DraftRecord(source=dl.SOURCE_HISTORY, league_id="1", season=2024,
                       picks=_picks())
    b = dl.DraftRecord(source=dl.SOURCE_HISTORY, league_id="1", season=2024,
                       picks=_picks())

    assert dl.record(corpus, a) == dl.record(corpus, b)
    assert len(dl.picks(corpus)) == 4, "the same draft was stored twice"


def test_two_mocks_on_one_day_are_two_drafts(corpus):
    """A mock has no league-and-season identity -- run three in an afternoon
    and they are three drafts. Collapsing them onto one id would silently
    discard all but the last, which is the corpus losing data it can never
    recover."""
    import datetime as dt
    one = dl.DraftRecord(source=dl.SOURCE_MOCK, league_id="1", season=2026,
                         started_at=dt.datetime(2026, 8, 21, 10, 0),
                         picks=_picks())
    two = dl.DraftRecord(source=dl.SOURCE_MOCK, league_id="1", season=2026,
                         started_at=dt.datetime(2026, 8, 21, 14, 0),
                         picks=_picks())
    dl.record(corpus, one)
    dl.record(corpus, two)

    assert one.resolved_id() != two.resolved_id()
    assert len(dl.picks(corpus)) == 8


def test_rerecording_a_draft_replaces_its_picks_rather_than_merging(corpus):
    """A draft recorded at pick 4 and again at pick 10 must end with ten
    picks, not fourteen. Reconnecting mid-draft does exactly this."""
    d = dl.DraftRecord(source=dl.SOURCE_LIVE, league_id="1", season=2026,
                       started_at="fixed", picks=_picks(4))
    dl.record(corpus, d)
    d.picks = pd.concat([_picks(4), _picks(4).assign(
        pick_no=[5, 6, 7, 8], player_id=[f"q{i}" for i in range(4)])])
    dl.record(corpus, d)

    got = dl.picks(corpus)
    assert len(got) == 8
    assert sorted(got["pick_no"]) == list(range(1, 9))


def test_an_anonymous_seat_still_reaches_the_corpus(corpus):
    """Mock opponents have no identity that survives the session. They can
    never build a personal profile, but their picks ARE the population
    baseline, which is what most drafts are useful for -- so they must be
    stored, flagged rather than dropped."""
    d = dl.DraftRecord(source=dl.SOURCE_MOCK, league_id=None, season=2026,
                       started_at="t", picks=_picks(owner="anon:x:1", anon=True))
    dl.record(corpus, d)

    got = dl.picks(corpus)
    assert len(got) == 4
    assert got["is_anonymous"].all()
    assert dl.summary(corpus)["known_owners"].iloc[0] == 0


def test_anonymous_keys_never_collide_across_drafts(corpus):
    """Seat 1 of one mock is not seat 1 of another. A key that collided would
    merge strangers into a single fictitious profile."""
    assert dl.anonymous_key("mock:aaa", 1) != dl.anonymous_key("mock:bbb", 1)
    assert dl.anonymous_key("mock:aaa", 1).startswith(dl.ANONYMOUS_PREFIX)


def test_recording_a_draft_with_no_picks_yet_still_registers_it(corpus):
    """A draft that has been joined but not started is a real row: the live
    recorder writes the header first and the picks as they land."""
    dl.record(corpus, dl.DraftRecord(source=dl.SOURCE_LIVE, league_id="9",
                                     season=2026, started_at="t"))
    assert len(dl.picks(corpus)) == 0
    assert corpus.execute("SELECT count(*) FROM draft_log").fetchone()[0] == 1


def test_the_pool_snapshot_is_stored_once_per_draft(corpus):
    """Availability at pick N is the pool minus the first N picks, so one
    snapshot replays the whole draft. Storing the available set per pick
    would be the same information 130 times over."""
    pool = pd.DataFrame({"player_id": ["p0", "p1"], "position": ["RB", "WR"],
                         "team": ["DET", "GB"], "adp_rank": [1.0, 2.0],
                         "proj_points": [250.0, 240.0]})
    dl.record(corpus, dl.DraftRecord(source=dl.SOURCE_MOCK, season=2026,
                                     started_at="t", picks=_picks(2), pool=pool))

    got = corpus.execute("SELECT * FROM draft_log_pool").df()
    assert len(got) == 2
    assert set(got["player_id"]) == {"p0", "p1"}


def test_ensure_schema_adds_autodrafted_without_losing_existing_rows(tmp_path):
    """`autodrafted` was added after `draft_log_pick` first shipped. A
    corpus file written before that column existed must gain it in place --
    the mock backfill (pipeline.mock_backfill) and every earlier recorder
    both write NULL for it, and a corpus that lost its rows getting there
    would be exactly the failure this module's docstring says never to
    risk."""
    path = str(tmp_path / "old_corpus.duckdb")
    conn = duckdb.connect(path)
    # The DDL ensure_schema ran before `autodrafted` existed, reproduced by
    # hand rather than imported, so this test still means the same thing
    # after that DDL gains the column.
    conn.execute("""CREATE TABLE draft_log_pick (
        draft_id VARCHAR, pick_no INTEGER, round INTEGER, slot INTEGER,
        owner_key VARCHAR, is_anonymous BOOLEAN, player_id VARCHAR,
        position VARCHAR, adp_rank DOUBLE, proj_points DOUBLE,
        PRIMARY KEY (draft_id, pick_no))""")
    conn.execute("INSERT INTO draft_log_pick VALUES "
                 "('d1', 1, 1, 1, 'anon:d1:1', true, 'p1', 'RB', 1.0, 200.0)")

    dl.ensure_schema(conn)

    cols = {r[1] for r in conn.execute("PRAGMA table_info('draft_log_pick')").fetchall()}
    assert "autodrafted" in cols
    row = conn.execute("SELECT * FROM draft_log_pick").df()
    assert len(row) == 1, "the pre-existing row must survive the ALTER"
    assert row["player_id"].iloc[0] == "p1"
    assert pd.isna(row["autodrafted"].iloc[0])


def test_a_failed_insert_leaves_the_previously_recorded_draft_intact(corpus):
    """The delete and the insert are one transaction, and this is why.

    `record` replaces a draft by deleting all three of its tables and
    inserting the new copy. Between those two steps the corpus holds nothing
    for that draft -- and the corpus is the one thing in this project that
    cannot be rebuilt from a source that still exists. The failure is not
    hypothetical: re-running `make mock-backfill` over an already-recorded
    file after a board change that makes the merge emit two rows for one
    pick_no gets the delete, and then the primary key rejects the insert.
    Unwrapped, that turns a re-record into a deletion.

    A duplicate `(draft_id, pick_no)` is exactly what is used here, because
    it is the real reachable case rather than an injected fault.
    """
    good = dl.DraftRecord(source=dl.SOURCE_MOCK, league_id="7", season=2026,
                          started_at="t", picks=_picks(4),
                          pool=pd.DataFrame({"player_id": ["p0"],
                                             "position": ["RB"],
                                             "team": ["DET"],
                                             "adp_rank": [1.0],
                                             "proj_points": [250.0]}))
    draft_id = dl.record(corpus, good)

    broken = _picks(4)
    broken.loc[1, "pick_no"] = 1          # two picks claiming pick 1
    with pytest.raises(Exception):
        dl.record(corpus, dl.DraftRecord(
            source=dl.SOURCE_MOCK, league_id="7", season=2026,
            started_at="t", picks=broken))

    # Everything the first record wrote is still there, in all three tables.
    stored = corpus.execute(
        "SELECT pick_no FROM draft_log_pick WHERE draft_id = ? "
        "ORDER BY pick_no", [draft_id]).df()
    assert list(stored["pick_no"]) == [1, 2, 3, 4]
    assert corpus.execute("SELECT count(*) FROM draft_log WHERE draft_id = ?",
                          [draft_id]).fetchone()[0] == 1
    assert corpus.execute(
        "SELECT count(*) FROM draft_log_pool WHERE draft_id = ?",
        [draft_id]).fetchone()[0] == 1


def test_a_rolled_back_record_leaves_the_connection_usable(corpus):
    """The rollback has to end the transaction, not just abandon it: the farm
    records one draft every forty minutes on a long-lived connection, and a
    process left inside a failed transaction would lose every draft after the
    first bad one rather than just that one."""
    broken = _picks(4)
    broken.loc[1, "pick_no"] = 1
    with pytest.raises(Exception):
        dl.record(corpus, dl.DraftRecord(source=dl.SOURCE_MOCK, season=2026,
                                         started_at="a", picks=broken))

    draft_id = dl.record(corpus, dl.DraftRecord(
        source=dl.SOURCE_MOCK, season=2026, started_at="b", picks=_picks(4)))
    assert len(dl.picks(corpus)) == 4
    assert corpus.execute("SELECT count(*) FROM draft_log WHERE draft_id = ?",
                          [draft_id]).fetchone()[0] == 1


def test_the_espn_and_timing_columns_land_on_a_corpus_that_already_has_rows():
    """The migration that matters most, because it ran against a live file.

    `espn_rank`/`espn_proj`/`bye` on the pool and `seconds_to_pick`/
    `clock_seconds` on the picks were added while three farm processes were
    writing drafts into the real corpus. ALTER ... ADD COLUMN IF NOT EXISTS
    is the only shape that is safe there: a rebuild would destroy the one
    copy of every draft ever recorded (see this module's docstring), and the
    existing rows must come out the other side unchanged, with NULL in the
    new columns rather than a default.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/old_corpus.duckdb"
        conn = duckdb.connect(path)
        # The DDL as it stood BEFORE this change, written by hand rather than
        # imported, so this test keeps meaning the same thing afterwards.
        conn.execute("""CREATE TABLE draft_log_pick (
            draft_id VARCHAR, pick_no INTEGER, round INTEGER, slot INTEGER,
            owner_key VARCHAR, is_anonymous BOOLEAN, player_id VARCHAR,
            position VARCHAR, adp_rank DOUBLE, proj_points DOUBLE,
            autodrafted BOOLEAN, had_owner BOOLEAN,
            PRIMARY KEY (draft_id, pick_no))""")
        conn.execute("""CREATE TABLE draft_log_pool (
            draft_id VARCHAR, player_id VARCHAR, position VARCHAR,
            team VARCHAR, adp_rank DOUBLE, proj_points DOUBLE,
            PRIMARY KEY (draft_id, player_id))""")
        conn.execute("INSERT INTO draft_log_pick VALUES "
                     "('d1', 1, 1, 1, 'anon:d1:1', true, 'p1', 'RB', 1.0, "
                     "200.0, false, true)")
        conn.execute("INSERT INTO draft_log_pool VALUES "
                     "('d1', 'p1', 'RB', 'DET', 1.0, 200.0)")

        dl.ensure_schema(conn)
        dl.ensure_schema(conn)         # idempotent: every open runs this

        pick = conn.execute("SELECT * FROM draft_log_pick").df()
        pool = conn.execute("SELECT * FROM draft_log_pool").df()
        assert len(pick) == 1 and len(pool) == 1, "existing rows must survive"
        assert pick["player_id"].iloc[0] == "p1"
        assert bool(pick["had_owner"].iloc[0]) is True
        # NULL on a draft recorded before the column existed, and for
        # `seconds_to_pick` that is permanent: nothing on disk holds the
        # frame arrival times of a draft that is already over.
        for col in ("seconds_to_pick", "clock_seconds"):
            assert pd.isna(pick[col].iloc[0]), col
        for col in ("espn_rank", "espn_proj", "bye"):
            assert pd.isna(pool[col].iloc[0]), col
        conn.close()


def test_a_writer_that_omits_the_new_columns_still_records_its_draft(corpus):
    """A farm process holding the OLD module keeps writing while the schema
    moves under it -- that is the normal way this change lands, not an edge
    case. `_shape` fills what the frame omits, and `record` names its columns
    so a drifted frame fails loudly instead of writing every value one column
    to the left."""
    old_pool = pd.DataFrame({"player_id": ["p0"], "position": ["RB"],
                             "team": ["DET"], "adp_rank": [1.0],
                             "proj_points": [250.0]})
    draft_id = dl.record(corpus, dl.DraftRecord(
        source=dl.SOURCE_MOCK, season=2026, started_at="t",
        picks=_picks(2), pool=old_pool))

    pool = corpus.execute("SELECT * FROM draft_log_pool WHERE draft_id = ?",
                          [draft_id]).df()
    picks = corpus.execute("SELECT * FROM draft_log_pick WHERE draft_id = ?",
                           [draft_id]).df()
    assert len(pool) == 1 and len(picks) == 2
    assert pool["proj_points"].iloc[0] == 250.0     # not shifted a column
    assert pd.isna(pool["espn_rank"].iloc[0])
    assert pd.isna(picks["seconds_to_pick"]).all()


# ---------------------------------------------------------------------------
# Shapes: what a draft was played under, and how much of each the corpus holds.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("blob,expected", [
    ('{"receptions": 1.0}', "ppr"),
    ('{"receptions": 0.5}', "half"),
    ('{"receptions": 0.0}', "std"),
    ('{"passing_tds": 4.0}', "std"),            # scored, and not receptions
    # The whole-settings blob, which is what every draft recorded before
    # `scoring_json` was written carries -- and the only thing on disk today.
    ('{"teams": 12, "scoring": {"receptions": 0.5}}', "half"),
    ('{"teams": 12, "scoring": {}}', "ppr"),    # nothing mapped: PPR, as ever
    (None, "ppr"),
    ("", "ppr"),
    ("not json at all", "ppr"),
    ("[1, 2, 3]", "ppr"),
])
def test_a_drafts_format_is_read_off_what_a_reception_was_worth(blob, expected):
    assert dl.draft_format(blob) == expected


def test_the_format_rule_is_the_leagues_own():
    """One rule, in `scoring.league`, read by both: the corpus is COUNTED by
    this function and QUERIED with the league's, so two copies of the
    thresholds that drifted would have a room looking up a shape the farm
    never recorded."""
    import dataclasses

    from scoring import league as league_mod

    base = league_mod.default_settings()
    for points, expected in ((1.0, "ppr"), (0.5, "half"), (0.0, "std")):
        settings = dataclasses.replace(
            base, scoring={**base.scoring, "receptions": points})
        assert league_mod.scoring_format(settings) == expected
        assert dl.draft_format(league_mod.to_json(settings)) == expected


def _head(corpus, teams, receptions, n, source=dl.SOURCE_MOCK):
    import json

    for i in range(n):
        dl.record(corpus, dl.DraftRecord(
            source=source, league_id="1", season=2026, teams=teams,
            rounds=16, started_at=f"{source}-{teams}-{receptions}-{i}",
            scoring_json=json.dumps({"receptions": receptions})))


def test_the_corpus_counts_its_drafts_by_shape(corpus, tmp_path):
    """What the farm's rotation reads: how many drafts of each (teams,
    format) are already recorded."""
    _head(corpus, 8, 1.0, 3)
    _head(corpus, 10, 1.0, 2)
    _head(corpus, 12, 0.0, 1)
    corpus.close()
    assert dl.shape_counts(str(tmp_path / "corpus.duckdb")) == {
        (8, "ppr"): 3, (10, "ppr"): 2, (12, "std"): 1}


def test_a_draft_with_no_team_count_is_left_out_rather_than_guessed(corpus,
                                                                    tmp_path):
    """Team count is the one field a shape cannot be assumed for -- see
    `mock_farm.play_draft`, which refuses a room the lobby and the league
    disagree about."""
    _head(corpus, 8, 1.0, 2)
    dl.record(corpus, dl.DraftRecord(source=dl.SOURCE_MOCK, league_id="1",
                                     season=2026, started_at="shapeless"))
    corpus.close()
    assert dl.shape_counts(str(tmp_path / "corpus.duckdb")) == {(8, "ppr"): 2}


def test_counting_an_unreadable_corpus_is_empty_rather_than_an_error(tmp_path):
    """The farm holds the write lock while it records, so a poll that cannot
    read the counts is ordinary -- and `{}` reads as "nothing recorded yet",
    which makes the rotation try every shape rather than none."""
    assert dl.shape_counts(str(tmp_path / "does-not-exist.duckdb")) == {}
    junk = tmp_path / "junk.duckdb"
    junk.write_text("not a database")
    assert dl.shape_counts(str(junk)) == {}


def test_the_rotations_counts_are_mocks_only(corpus, tmp_path):
    """An imported league history carries no scoring blob -- `backfill_history`
    writes picks and nothing else -- so every one of its seasons would read as
    PPR whatever the league actually scored. Six of them counted as six 8-team
    PPR drafts is enough to nudge a rotation that is deliberately keeping its
    shapes within a few drafts of each other."""
    _head(corpus, 8, 1.0, 2)
    _head(corpus, 12, 0.0, 3, source=dl.SOURCE_HISTORY)
    corpus.close()
    path = str(tmp_path / "corpus.duckdb")

    assert dl.shape_counts(path) == {(8, "ppr"): 2}
    # And a caller that wants the whole file can still have it.
    assert dl.shape_counts(path, source=None) == {(8, "ppr"): 2,
                                                  (12, "std"): 3}
