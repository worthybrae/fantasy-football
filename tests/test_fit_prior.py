"""pipeline.fit_prior: refitting the cold-start prior on the mock corpus.

The subject of most of these tests is not the arithmetic -- `fit`,
`feature_matrix` and `_softmax` are `draft_model`'s and are tested there.
It is the DISCIPLINE around the arithmetic, which is where this task could
go wrong invisibly:

  * a losing refit must write nothing, and there must be no way to make it
  * the fold must be the draft, not the pick
  * the picks that are excluded must be the ones that were meant to be, and
    excluding a pick must not hand the next one a board that still holds a
    player who is gone
  * a pick the pool does not carry must be COUNTED, never silently skipped

Every corpus here is a throwaway file under `tmp_path`. Nothing in this file
opens, let alone writes, the real `data/draft_corpus.duckdb`.
"""
import numpy as np
import pandas as pd
import pytest

from pipeline import draft_log as dl
from pipeline import fit_prior as fp
from pipeline.db import get_conn, write_table
from scoring import league as league_mod
from scoring.draft_model import FEATURE_NAMES, LEGACY_FEATURE_NAMES

TEAMS = 4
ROUNDS = 3
POOL_SIZE = 24
# Positions cycle so every position dummy is exercised and so "the best
# available at a position" is a different player from "the best available".
_POSITIONS = ["RB", "WR", "QB", "TE"]


def _settings():
    """A 4x3 league. `LeagueSettings.rounds` is derived from the roster shape,
    so three rounds means three starters and no bench -- there is no `rounds`
    field to set directly."""
    return league_mod.LeagueSettings(
        season=2026, teams=TEAMS,
        starters={"QB": 1, "RB": 1, "WR": 1}, flex_slots=0, bench=0,
        scoring={"receptions": 1.0}, draft_type="SNAKE")


def _pool():
    return pd.DataFrame({
        "player_id": [f"p{i}" for i in range(POOL_SIZE)],
        "position": [_POSITIONS[i % len(_POSITIONS)] for i in range(POOL_SIZE)],
        "team": ["DET"] * POOL_SIZE,
        # Dense 1..N, which is what `draft_sim.build_pool` writes and what
        # `fit_prior.enrich_pool` reads back as `market_rank`.
        "adp_rank": [float(i + 1) for i in range(POOL_SIZE)],
        "proj_points": [float(POOL_SIZE - i) for i in range(POOL_SIZE)],
    })


def _picks(draft_id, taken, my_slot=None, autodrafted=None):
    """`taken` is one pool INDEX per pick, in pick order."""
    slots = []
    for r in range(ROUNDS):
        forward = list(range(1, TEAMS + 1))
        slots.extend(forward if r % 2 == 0 else forward[::-1])
    pool = _pool()
    n = len(taken)
    return pd.DataFrame({
        "pick_no": range(1, n + 1),
        "round": [(i // TEAMS) + 1 for i in range(n)],
        "slot": slots[:n],
        "owner_key": [dl.anonymous_key(draft_id, s) for s in slots[:n]],
        "is_anonymous": [True] * n,
        "player_id": [pool["player_id"][i] for i in taken],
        "position": [pool["position"][i] for i in taken],
        "adp_rank": [pool["adp_rank"][i] for i in taken],
        "proj_points": [None] * n,
        "autodrafted": (autodrafted if autodrafted is not None
                        else [None] * n),
    })


def _alternating_picks(n):
    """Best available on an even pick, second best on an odd one.

    Deliberately learnable but not separable. "Always the best available"
    would make the likelihood unbounded -- the fit would run off to infinity
    and report a convergence warning -- while pure noise would make a refit's
    win a coin flip. This gives a refit a real rule to find (top of the
    board) and a ceiling of about half the picks, which is a stable number to
    assert against.
    """
    taken, gone = [], set()
    for pick in range(n):
        available = [i for i in range(POOL_SIZE) if i not in gone]
        choice = available[0] if pick % 2 == 0 else available[1]
        taken.append(choice)
        gone.add(choice)
    return taken


def _write_corpus(path, drafts):
    """`drafts` is a list of (draft_id, picks_frame). Written, then closed --
    the corpus must be CLOSED before `open_corpus` can open the same file
    read-only from this process."""
    conn = dl.corpus_conn(str(path))
    settings_json = league_mod.to_json(_settings())
    for draft_id, picks in drafts:
        dl.record(conn, dl.DraftRecord(
            source=dl.SOURCE_MOCK, league_id=draft_id, season=2026,
            teams=TEAMS, rounds=ROUNDS,
            my_slot=picks.attrs.get("my_slot"),
            settings_json=settings_json, draft_id=draft_id,
            picks=picks, pool=_pool()))
    conn.close()
    return str(path)


def _corpus_of(tmp_path, n_drafts=3, **kwargs):
    n = TEAMS * ROUNDS
    drafts = []
    for d in range(n_drafts):
        draft_id = f"mock:fixture{d}"
        picks = _picks(draft_id, _alternating_picks(n), **kwargs)
        picks.attrs["my_slot"] = kwargs.get("my_slot")
        drafts.append((draft_id, picks))
    return _write_corpus(tmp_path / "corpus.duckdb", drafts)


def _empty_league(tmp_path):
    """A league database with no `weekly` at all.

    That is a real state -- a fresh provision before any refresh -- and it is
    the one that exercises the neutral-attribute path: every stat-profile
    column reaches `feature_matrix` as NaN and centres to 0.0, exactly as a
    player the join misses does.
    """
    path = str(tmp_path / "league.duckdb")
    get_conn(path).close()
    return path


@pytest.fixture
def corpus_and_league(tmp_path):
    corpus_path = _corpus_of(tmp_path)
    league_path = _empty_league(tmp_path)
    return corpus_path, league_path


# ------------------------------------------------------------- the decision

def _run(corpus_path, league_path, monkeypatch, tmp_path, incumbent=None):
    """`main` with the generated module redirected into tmp_path."""
    target = tmp_path / "generated_prior.py"
    monkeypatch.setattr(fp, "PRIOR_MODULE", target)
    if incumbent is not None:
        monkeypatch.setattr(fp, "COLD_START_PRIOR", incumbent)
    code = fp.main(["--corpus", corpus_path, "--league-db", league_path])
    return code, target


def _anti_market_prior():
    """An incumbent that predicts the exact opposite of the board.

    `reach` is `max(0, log1p(market_rank) - log1p(pick))`, so a large POSITIVE
    coefficient on it ranks the deepest available player first. Every pick in
    the fixture comes from the top two of the board, so this scores top-1 zero
    -- which makes "does a refit beat the incumbent" a question about the
    refit rather than about how good the real incumbent happens to be on a
    synthetic fixture.
    """
    beta = np.zeros(len(FEATURE_NAMES))
    beta[FEATURE_NAMES.index("reach")] = 50.0
    return beta


def test_a_winning_refit_writes_the_generated_module(corpus_and_league,
                                                     monkeypatch, tmp_path):
    corpus_path, league_path = corpus_and_league
    code, target = _run(corpus_path, league_path, monkeypatch, tmp_path,
                        incumbent=_anti_market_prior())

    assert code == 0
    assert target.exists()


def test_a_losing_refit_writes_nothing_at_all(corpus_and_league, monkeypatch,
                                              tmp_path):
    """The whole rule, in one assertion: a refit that does not beat the
    incumbent on top-1 leaves `scoring/mock_prior.py` untouched and exits
    non-zero. There is no flag anywhere that changes this.

    The seam is the scorer rather than the data because the thing under test
    is the DECISION. Constructing a corpus on which a refit genuinely loses
    would test the fixture; forcing the measurement to report a loss tests
    what `main` does with one.
    """
    corpus_path, league_path = corpus_and_league
    losing = {"top1": 0.0, "top5": 0.0, "logloss": 9.9, "n": 1,
              "by_draft": {}, "by_bucket": {}}
    monkeypatch.setattr(fp, "leave_one_draft_out",
                        lambda *a, **k: dict(losing))

    code, target = _run(corpus_path, league_path, monkeypatch, tmp_path)

    assert code == 1
    assert not target.exists(), "a losing refit wrote a prior"


def test_there_is_no_force_flag(corpus_and_league, monkeypatch, tmp_path):
    """Shipping a losing prior has to be a reviewable diff, not an argument.

    Passing the flags somebody would reach for changes nothing: `main` reads
    exactly two options and neither of them is an override.
    """
    corpus_path, league_path = corpus_and_league
    losing = {"top1": 0.0, "top5": 0.0, "logloss": 9.9, "n": 1,
              "by_draft": {}, "by_bucket": {}}
    monkeypatch.setattr(fp, "leave_one_draft_out", lambda *a, **k: dict(losing))
    target = tmp_path / "generated_prior.py"
    monkeypatch.setattr(fp, "PRIOR_MODULE", target)

    for flag in ("--force", "--yes", "--write"):
        assert fp.main(["--corpus", corpus_path, "--league-db", league_path,
                        flag]) == 1
        assert not target.exists()


# ----------------------------------------------------------- what it writes

def test_the_generated_module_has_one_coefficient_per_feature(corpus_and_league,
                                                              monkeypatch,
                                                              tmp_path):
    """The property `draft_model`'s import-time assertions depend on. A
    generated file one coefficient short would mis-score every cold-start
    opponent, silently, for every feature after the gap."""
    corpus_path, league_path = corpus_and_league
    _, target = _run(corpus_path, league_path, monkeypatch, tmp_path,
                     incumbent=_anti_market_prior())

    namespace = {}
    exec(compile(target.read_text(), str(target), "exec"), namespace)

    assert list(namespace["FEATURES"]) == FEATURE_NAMES
    assert len(namespace["PRIOR"]) == len(FEATURE_NAMES)
    assert np.isfinite(namespace["PRIOR"]).all()


def test_the_generated_module_records_the_drafts_it_was_fitted_on(
        corpus_and_league, monkeypatch, tmp_path):
    """The corpus GROWS while a fit runs -- the farm adds a draft every
    fifteen minutes -- so a fit that does not name its drafts is not
    reproducible."""
    corpus_path, league_path = corpus_and_league
    _, target = _run(corpus_path, league_path, monkeypatch, tmp_path,
                     incumbent=_anti_market_prior())

    namespace = {}
    exec(compile(target.read_text(), str(target), "exec"), namespace)

    assert namespace["DRAFT_IDS"] == ["mock:fixture0", "mock:fixture1",
                                      "mock:fixture2"]
    assert namespace["PROVENANCE"]["folds"] == "leave-one-draft-out"
    assert namespace["PROVENANCE"]["drafts"] == 3


def test_a_cut_feature_is_zero_and_says_so(tmp_path):
    """A 0.0 that was measured and rejected and a 0.0 that was never measured
    are different facts. Only the cut list can tell them apart, so the
    annotation is driven by that list and never by the value."""
    beta = np.arange(len(FEATURE_NAMES), dtype=float)
    beta[FEATURE_NAMES.index("usage")] = 0.0
    text = fp.render_module(beta, FEATURE_NAMES, ["mock:1"], {"drafts": 1},
                            "prose", cut=["usage"])

    assert "# usage" in text
    usage_line = [l for l in text.split("\n") if l.endswith("cut on delta_top1, not fitted")]
    assert len(usage_line) == 1 and "usage" in usage_line[0]


def test_the_shipped_feature_set_keeps_the_15_and_cuts_a_loser():
    """`delta_top1 > 0` earns a place; at or below zero is cut. Zero is a cut,
    not a keep -- "not shown to hurt" is not "shown to help"."""
    table = pd.DataFrame([
        {"dropped": "none", "delta_top1": 0.0},
        {"dropped": "held_at_pos", "delta_top1": 0.01},
        {"dropped": "first_at_pos", "delta_top1": 0.0},
        {"dropped": "usage", "delta_top1": -0.02},
    ])

    keep = fp.shipped_features(table)

    assert set(LEGACY_FEATURE_NAMES) <= set(keep)
    assert "held_at_pos" in keep
    assert "first_at_pos" not in keep, "a delta of exactly zero is a cut"
    assert "usage" not in keep
    # FEATURE_NAMES order, not the order the ablation reported them in: a
    # beta fitted against one column order and applied against another is
    # silently wrong.
    assert keep == [f for f in FEATURE_NAMES if f in set(keep)]


def test_a_cut_feature_has_no_coefficient_at_all(corpus_and_league):
    """Cut means the column is not in the fit, not that its coefficient came
    out small. Fitting all 23 and zeroing the losers afterwards would leave
    every surviving coefficient carrying mass it had been splitting with a
    column the shipped model does not have."""
    corpus_path, league_path = corpus_and_league
    corpus, league_conn = fp.open_corpus(corpus_path), fp.open_league(league_path)
    try:
        obs = fp.build_corpus_observations(
            corpus, league_conn, fp.snapshot_draft_ids(corpus))
        X_list, chosen, _, _, _ = fp.design(obs)
    finally:
        corpus.close()
        league_conn.close()

    beta = fp.full_beta(X_list, chosen, LEGACY_FEATURE_NAMES)

    assert len(beta) == len(FEATURE_NAMES)
    assert (beta[len(LEGACY_FEATURE_NAMES):] == 0.0).all()


# --------------------------------------------------------- which picks fit

def _observations(corpus_path, league_path):
    corpus, league_conn = fp.open_corpus(corpus_path), fp.open_league(league_path)
    try:
        return fp.build_corpus_observations(
            corpus, league_conn, fp.snapshot_draft_ids(corpus))
    finally:
        corpus.close()
        league_conn.close()


def test_our_own_seat_is_excluded_but_still_takes_its_player_off_the_board(
        tmp_path):
    """Our picks are part policy and part deliberate randomness, so they are
    nobody's choices. They still happened, though: the next pick must not be
    offered a player our seat already took."""
    n = TEAMS * ROUNDS
    taken = _alternating_picks(n)
    picks = _picks("mock:one", taken)
    picks.attrs["my_slot"] = 1
    corpus_path = _write_corpus(tmp_path / "corpus.duckdb", [("mock:one", picks)])
    conn = dl.corpus_conn(corpus_path)
    conn.execute("UPDATE draft_log SET my_slot = 1")
    conn.close()

    obs = _observations(corpus_path, _empty_league(tmp_path))

    assert obs.dropped["my_slot"] == ROUNDS, "one pick per round is ours"
    assert len(obs.observations) == n - ROUNDS
    # Pick 2 is not ours, and its choice set must already be missing the
    # player pick 1 (ours) took.
    second = obs.observations[0]
    assert second.overall_pick == 2
    first_player = picks["player_id"].iloc[0]
    assert first_player not in set(second.pool["player_id"])


def test_an_unknown_autodraft_flag_is_kept_and_a_true_one_is_not(tmp_path):
    """NULL means UNKNOWN, not True. Every backfilled pick carries NULL --
    a `drafted` table never had the column -- so excluding NULL would throw
    away essentially the whole corpus. This is a standing ruling."""
    n = TEAMS * ROUNDS
    flags = [None] * n
    flags[0] = True
    flags[1] = False
    picks = _picks("mock:one", _alternating_picks(n), autodrafted=flags)
    corpus_path = _write_corpus(tmp_path / "corpus.duckdb", [("mock:one", picks)])

    obs = _observations(corpus_path, _empty_league(tmp_path))

    assert obs.dropped["autodrafted"] == 1
    assert len(obs.observations) == n - 1
    assert fp._is_autodrafted(None) is False
    assert fp._is_autodrafted(pd.NA) is False
    assert fp._is_autodrafted(False) is False
    assert fp._is_autodrafted(True) is True


def test_a_pick_the_pool_does_not_carry_is_counted_not_silenced(tmp_path):
    """A silent drop looks exactly like a smaller corpus rather than like the
    join failure it is, and nobody reading the totals afterwards could tell
    the difference."""
    n = TEAMS * ROUNDS
    picks = _picks("mock:one", _alternating_picks(n))
    picks.loc[3, "player_id"] = "not_in_this_pool"
    corpus_path = _write_corpus(tmp_path / "corpus.duckdb", [("mock:one", picks)])

    obs = _observations(corpus_path, _empty_league(tmp_path))

    assert obs.dropped["not_in_pool"] == 1
    assert len(obs.observations) == n - 1
    assert obs.picks_seen == n


def test_a_pick_that_does_not_fit_still_counts_toward_the_teams_history(
        tmp_path):
    """Roster counts, the room's recent picks and this team's last pick at
    each position are bookkeeping about what HAPPENED, not about what was
    fitted. An autodrafted pick still filled a roster slot."""
    n = TEAMS * ROUNDS
    flags = [None] * n
    flags[0] = True                       # slot 1's first-round pick
    picks = _picks("mock:one", _alternating_picks(n), autodrafted=flags)
    corpus_path = _write_corpus(tmp_path / "corpus.duckdb", [("mock:one", picks)])

    obs = _observations(corpus_path, _empty_league(tmp_path))

    # Slot 1 picks again at pick 8 (the snake turns), and by then it must
    # know it already holds the position pick 1 took.
    eighth = [o for o in obs.observations if o.overall_pick == 8][0]
    assert eighth.roster.get(picks["position"].iloc[0]) == 1
    assert eighth.last_pick_at_pos.get(picks["position"].iloc[0]) == 1


def test_market_rank_is_the_board_that_was_stored(corpus_and_league):
    """The pool snapshot IS the board the room drafted against. Re-deriving
    one today from a live ADP feed would score every pick against a market
    that did not exist when it was made."""
    corpus_path, league_path = corpus_and_league
    obs = _observations(corpus_path, league_path)

    first = obs.observations[0]
    assert list(first.pool["market_rank"]) == sorted(first.pool["market_rank"])
    assert first.pool["market_rank"].iloc[0] == 1.0
    assert "adp_rank" not in first.pool.columns, (
        "market_rank must be the stored rank renamed, not a second column "
        "that could disagree with it")


def test_the_choice_set_shrinks_by_exactly_one_player_per_pick(
        corpus_and_league):
    corpus_path, league_path = corpus_and_league
    obs = _observations(corpus_path, league_path)

    first_draft = [o for o, d in zip(obs.observations, obs.draft_ids)
                   if d == "mock:fixture0"]
    sizes = [len(o.pool) for o in first_draft]
    assert sizes == [POOL_SIZE - (o.overall_pick - 1) for o in first_draft]
    for o in first_draft:
        assert 0 <= o.chosen < len(o.pool)


# ---------------------------------------------------------------- the folds

def test_the_fold_is_the_draft_and_a_holdout_is_never_fitted_on(
        corpus_and_league, monkeypatch):
    """Leave-one-SEASON-out is degenerate here: every mock draft is 2026, so
    it would be one fold with nothing to train on. The unit of independence
    is the draft -- one room, one board, eight strangers -- and a random
    split over picks would put the first half of a draft in training and the
    second half in test, where the model has already been told who is gone.
    """
    corpus_path, league_path = corpus_and_league
    obs = _observations(corpus_path, league_path)
    X_list, chosen, groups, _, _ = fp.design(obs)
    # Each fold's training set, recorded by identity: a matrix is in the
    # training set for a fold iff `fit` was handed that exact object.
    seen = []
    real_fit = fp.fit
    monkeypatch.setattr(fp, "fit",
                        lambda X, c, **kw: (seen.append({id(x) for x in X}),
                                            real_fit(X, c, **kw))[1])

    report = fp.leave_one_draft_out(X_list, chosen, groups)

    assert sorted(report["by_draft"]) == ["mock:fixture0", "mock:fixture1",
                                          "mock:fixture2"]
    assert len(seen) == 3
    for train_ids, holdout in zip(seen, sorted(set(groups))):
        held = {id(X) for X, g in zip(X_list, groups) if g == holdout}
        assert not (train_ids & held), f"{holdout} was fitted on itself"
        assert len(train_ids) == len(X_list) - len(held)


def test_the_clustered_standard_error_is_over_drafts_not_picks():
    """sqrt(p(1-p)/n) over three thousand picks is about 0.8pp and it is
    wrong: those picks come from 25 rooms and picks inside a room share a
    board, a set of opponents, and everything already taken."""
    assert fp.cluster_se({"a": 0.2, "b": 0.2, "c": 0.2}) == pytest.approx(0.0)
    spread = fp.cluster_se({"a": 0.1, "b": 0.3})
    assert spread == pytest.approx(np.std([0.1, 0.3], ddof=1) / np.sqrt(2))
    assert np.isnan(fp.cluster_se({"a": 0.25}))


def test_the_decisions_uncertainty_is_paired_by_draft():
    """Two vectors scored on the same picks in the same rooms share whatever
    made those rooms hard, so the difference is measured per draft and the
    spread taken over drafts. Two unpaired standard errors would charge the
    comparison for variation common to both sides.
    """
    # Same spread, but one is a constant edge and the other is noise. The
    # unpaired standard errors are identical; only the paired one can tell
    # them apart, and telling them apart is the whole reason it exists.
    steady = fp.paired_se({"a": 0.3, "b": 0.4}, {"a": 0.2, "b": 0.3})
    noisy = fp.paired_se({"a": 0.3, "b": 0.4}, {"a": 0.4, "b": 0.2})

    assert steady == pytest.approx(0.0)
    assert noisy > steady
    # Only drafts both sides scored take part, and one draft cannot have a
    # spread at all.
    assert np.isnan(fp.paired_se({"a": 0.3}, {"a": 0.2}))
    assert np.isnan(fp.paired_se({"a": 0.3, "b": 0.4}, {"a": 0.2}))


def test_both_deltas_are_reported_for_every_candidate(corpus_and_league):
    """Reporting delta_top1 and leaving top-5 out invites reading the second
    one only when it agrees with the first."""
    corpus_path, league_path = corpus_and_league
    obs = _observations(corpus_path, league_path)
    X_list, chosen, groups, _, _ = fp.design(obs)

    table = fp.ablate(X_list, chosen, groups, candidates=["usage"])

    assert list(table["dropped"]) == ["none", "usage"]
    assert {"delta_top1", "delta_top5", "logloss"} <= set(table.columns)


# ------------------------------------------------------------- read-only-ness

def test_the_corpus_is_opened_read_only(corpus_and_league):
    """The corpus is the one thing here that cannot be rebuilt from a source
    that still exists. A mock draft not written down when it happens is gone,
    so this module must not be able to touch it even by accident."""
    corpus_path, _ = corpus_and_league
    conn = fp.open_corpus(corpus_path)
    try:
        with pytest.raises(Exception):
            conn.execute("DELETE FROM draft_log_pick")
    finally:
        conn.close()


def test_no_write_against_draft_log_anywhere_in_the_module():
    """A grep, as a test. `fit_prior` is a fitting step that happens to read
    the corpus; a CREATE OR REPLACE or a DROP against these tables would
    destroy the only copy of drafts nobody can replay."""
    import pathlib
    source = pathlib.Path(fp.__file__).read_text().upper()
    for statement in ("CREATE OR REPLACE", "DROP TABLE", "DELETE FROM",
                      "INSERT INTO", "UPDATE DRAFT_LOG", "ALTER TABLE"):
        assert statement not in source, f"{statement} in pipeline/fit_prior.py"


# ------------------------------------------------------------- attributes

def test_a_players_history_lands_on_his_own_pool_row(tmp_path):
    """`attributes_as_of` is keyed by `adp_match_key`, which needs a NAME.
    A corpus pool stores `player_id` and no name at all, so the key has to be
    rebuilt -- and a rebuild that put one player's production on another's
    row would be invisible in every number downstream."""
    league_path = str(tmp_path / "league.duckdb")
    conn = get_conn(league_path)
    write_table(conn, "weekly", pd.DataFrame([
        {"player_id": "p0", "player_display_name": "Alpha Back",
         "position": "RB", "recent_team": "DET", "season": 2025, "week": w,
         "receptions": 4, "receiving_yards": 40, "targets": 6, "carries": 10}
        for w in range(1, 5)
    ] + [
        {"player_id": "p1", "player_display_name": "Beta Wideout",
         "position": "WR", "recent_team": "GB", "season": 2025, "week": w,
         "receptions": 1, "receiving_yards": 10, "targets": 2, "carries": 0}
        for w in range(1, 5)]))
    conn.close()
    league_conn = fp.open_league(league_path)
    try:
        attrs = fp.attributes_by_player_id(league_conn, 2026)
    finally:
        league_conn.close()

    by_id = attrs.set_index("player_id")
    assert set(by_id.index) == {"p0", "p1"}
    # The high-volume back must carry the higher usage; a mis-keyed rebuild
    # would swap them and nothing else in the pipeline would notice.
    assert by_id.loc["p0", "usage"] > by_id.loc["p1", "usage"]
    assert bool(by_id.loc["p0", "no_track_record"]) is False


def test_a_pool_row_with_no_history_reaches_the_matrix_neutral(
        corpus_and_league):
    """An empty league database is a real state -- a fresh provision before
    any refresh. Every stat-profile column must come through as the same
    neutral zero a missed join produces, not as a hole that crashes or a
    number that means something."""
    corpus_path, league_path = corpus_and_league
    obs = _observations(corpus_path, league_path)
    X_list, _, _, _, _ = fp.design(obs)

    X = X_list[0]
    assert np.isfinite(X).all()
    for name in ("usage", "efficiency", "played_share", "peak_gap"):
        assert (X[:, FEATURE_NAMES.index(name)] == 0.0).all()
    assert (X[:, FEATURE_NAMES.index("no_track_record")] == 1.0).all()
    # Same rule for the board columns the corpus does not store: an empty
    # league builds no board, so all four are unknown and unknown is 0.0.
    for name in ("vor", "durability", "proj_change", "last_of_tier"):
        assert (X[:, FEATURE_NAMES.index(name)] == 0.0).all()


def test_the_pool_signals_the_corpus_does_carry_are_built_from_it(
        corpus_and_league):
    """`dropoff_at_pos` and `slots_left_at_pos` need no board at all --
    `proj_points` is stored per pool row and the roster shape comes off
    `settings_json` -- so they must be live even against a league database
    with nothing in it. A column that is 0.0 for every candidate cancels out
    of the softmax exactly, so "present but always neutral" is the same thing
    as absent and this is what tells the two apart."""
    corpus_path, league_path = corpus_and_league
    obs = _observations(corpus_path, league_path)
    X_list, _, _, _, _ = fp.design(obs)

    X = X_list[0]
    for name in ("dropoff_at_pos", "slots_left_at_pos"):
        column = X[:, FEATURE_NAMES.index(name)]
        assert len(np.unique(column)) > 1, f"{name} is constant: {column}"


def test_the_board_columns_land_on_the_right_pool_row(tmp_path):
    """The corpus stores `adp_rank` and `proj_points` per pool row and
    nothing else, so `vor`, `durability`, `proj_change` and `tier` are joined
    from the board on `player_id`. A join that put one player's board row on
    another's pool row would be invisible in every number downstream -- the
    same failure `attributes_by_player_id` is guarded against above."""
    pool = _pool()
    signals = pd.DataFrame({
        "player_id": ["p0", "p2"],
        "vor": [120.0, -30.0], "durability": [88.0, 12.0],
        "proj_change": [2.5, -1.0], "tier": [1.0, 6.0]})
    enriched = fp.enrich_pool(pool, pd.DataFrame(), signals)

    by_id = enriched.set_index("player_id")
    assert by_id.loc["p0", "vor"] == 120.0
    assert by_id.loc["p2", "durability"] == 12.0
    # Everyone the board does not carry -- which is every DST, whose corpus
    # player_id is a synthesized `adp_<team>_defense` no board row has --
    # stays unknown rather than picking up a neighbour's numbers.
    assert pd.isna(by_id.loc["p1", "tier"])
    # And the join must not duplicate or drop a pool row: one pick removes
    # exactly one of them.
    assert len(enriched) == len(pool)


def test_a_duplicated_board_player_id_is_dropped_rather_than_duplicating_a_row(
        tmp_path):
    """`board._add_adp_only_players` can synthesize one `player_id` for two
    board rows (the same name at two positions in the ADP feed). A duplicate
    on the right of a left merge DUPLICATES the pool row, which breaks "one
    pick removes exactly one pool row" far downstream of here."""
    pool = _pool()
    signals = pd.DataFrame({
        "player_id": ["p0", "p0", "p1"],
        "vor": [120.0, 55.0, 10.0], "durability": [88.0, 40.0, 60.0],
        "proj_change": [2.5, 0.5, -1.0], "tier": [1.0, 4.0, 2.0]})
    # The guard lives in `board_signals_by_player_id`, so apply it there and
    # then join, which is the order production runs in.
    deduped = signals.drop_duplicates("player_id", keep=False)
    enriched = fp.enrich_pool(pool, pd.DataFrame(), deduped)

    assert len(enriched) == len(pool)
    by_id = enriched.set_index("player_id")
    assert pd.isna(by_id.loc["p0", "vor"])       # both sides of the collision
    assert by_id.loc["p1", "vor"] == 10.0


# ---------------------------------------------------------------------------
# One shape per fit.
# ---------------------------------------------------------------------------


def _odd_shape_draft(path, draft_id, teams, receptions):
    """One more draft in the same corpus file, of a different shape."""
    import json

    conn = dl.corpus_conn(str(path))
    try:
        dl.record(conn, dl.DraftRecord(
            source=dl.SOURCE_MOCK, league_id=draft_id, season=2026,
            teams=teams, rounds=ROUNDS, draft_id=draft_id,
            scoring_json=json.dumps({"receptions": receptions}),
            settings_json=league_mod.to_json(_settings()),
            picks=_picks(draft_id, _alternating_picks(TEAMS * ROUNDS)),
            pool=_pool()))
    finally:
        conn.close()


def test_the_fit_takes_one_shape_and_says_what_it_left_out(tmp_path):
    """Every pick in the fit is priced off ONE board, built once under one
    draft's rules, and every roster-shape feature is defined against a team
    count. That was exactly true while the corpus was all 8-team PPR mocks.
    A 12-team standard draft in the same fit would be scored against a board
    priced for somebody else's league, in a room with four more seats than
    the features assume.
    """
    path = _corpus_of(tmp_path, n_drafts=3)              # 4-team, PPR
    _odd_shape_draft(path, "mock:twelve", teams=12, receptions=1.0)
    _odd_shape_draft(path, "mock:standard", teams=TEAMS, receptions=0.0)

    corpus = fp.open_corpus(path)
    said = []
    try:
        ids = fp.snapshot_draft_ids(corpus, out=said.append)
    finally:
        corpus.close()

    assert ids == ["mock:fixture0", "mock:fixture1", "mock:fixture2"]
    assert len(said) == 1
    assert "2 draft(s) of other shapes left out" in said[0]
    assert f"{TEAMS} teams, ppr" in said[0]


def test_a_corpus_of_one_shape_says_nothing(tmp_path):
    """Which is every corpus this project has had until now: a line about
    exclusions on every run would be noise."""
    corpus = fp.open_corpus(_corpus_of(tmp_path, n_drafts=3))
    said = []
    try:
        ids = fp.snapshot_draft_ids(corpus, out=said.append)
    finally:
        corpus.close()

    assert len(ids) == 3 and said == []


def test_the_biggest_shape_wins_and_ties_are_broken_the_same_way_everywhere(
        tmp_path):
    """The archive pages and the prior have to agree about what the corpus
    IS, so both read `draft_log.dominant_shape`."""
    path = _corpus_of(tmp_path, n_drafts=2)              # 4-team PPR x2
    _odd_shape_draft(path, "mock:std1", teams=TEAMS, receptions=0.0)
    _odd_shape_draft(path, "mock:std2", teams=TEAMS, receptions=0.0)
    _odd_shape_draft(path, "mock:std3", teams=TEAMS, receptions=0.0)

    corpus = fp.open_corpus(path)
    try:
        ids = fp.snapshot_draft_ids(corpus, out=lambda *a: None)
    finally:
        corpus.close()

    assert ids == ["mock:std1", "mock:std2", "mock:std3"]
    # And on a tie the same rule the pages use: PPR first.
    assert dl.dominant_shape({(4, "std"): 2, (4, "ppr"): 2}) == (4, "ppr")
