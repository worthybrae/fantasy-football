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
