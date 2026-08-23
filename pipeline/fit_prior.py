"""Refit the cold-start prior on the mock-draft corpus. Run: make fit-prior

WHAT THIS DECIDES. `scoring.draft_model.COLD_START_PRIOR` is the entire model
for a league with no draft history -- which is every mock draft, where all
eight opponents are strangers the tool has never seen. Until now that vector
was fifteen coefficients fitted on ONE league's 696 picks, by the same eight
people, six times. `pipeline.draft_log`'s corpus now holds real 8-team mock
drafts from ESPN's public lobby, and Task 3 added eight candidate features
that nothing has ever measured. This module fits a new vector on that corpus,
measures it honestly against the incumbent, and ships it only if it wins.

THE RULE, WHICH IS NOT THIS MODULE'S TO BEND. `delta_top1` decides. The new
prior is written if and only if it beats the incumbent on held-out top-1
accuracy. Top-5 and log-loss are reported -- both, always, so neither can be
reached for only when it flatters the answer -- and neither decides.
`scoring.draft_model` states this rule where the feature ablation lives and
`docs/superpowers/findings/2026-08-11-stat-profile-vs-position-dummies.md`
records what happened the one time it was tempting to reach past it. There is
deliberately NO `--force` flag: shipping a losing prior should require editing
the rule below, in a diff someone can review, not passing an argument.

A LOSING FIT IS A COMPLETE RESULT. If the refit loses, this prints the
numbers, writes nothing, and exits 1. `scoring/mock_prior.py` is left byte
for byte as it was.

LEAVE-ONE-DRAFT-OUT, NOT LEAVE-ONE-SEASON-OUT. `draft_model.backtest` holds
out a season, because a league's history is six drafts in six seasons and
picks inside one draft are not independent of each other. Every draft in this
corpus is 2026, so that split is degenerate here -- one fold, nothing to train
on. The unit of independence is the DRAFT: one room, one board, eight
strangers. So every draft is held out in turn, the pooled fit is rebuilt on
the other N-1, and the held-out draft's picks are scored by a vector that
never saw them.

WHICH PICKS ARE FITTED, AND WHY EACH EXCLUSION IS NARROW.

  * Our own seat (`draft_log.my_slot`) is excluded. Those picks are part
    policy and part deliberate epsilon-greedy randomness (see
    `pipeline.mock_farm`), so they are nobody's choices and modelling them
    would be fitting our own exploration noise.
  * `autodrafted IS TRUE` is excluded: ESPN's engine made that pick because
    nobody was sitting in that seat, which is not a decision. THIS IS NOW THE
    LOAD-BEARING FILTER, and the reason is that the flag finally means what
    it says. It used to be filled from AUTODRAFT frames alone, and ESPN sends
    no such frame for a seat that was never human -- so the seats ESPN pads a
    thin room out with, the very picks this exclusion is for, came back NULL
    and were fitted on. `pipeline.mock_farm` now seeds each slot's state from
    the room's own `?view=mTeam` owner census (no `owners` entry, no person,
    autodrafting from pick one) and transitions it on the AUTODRAFT
    broadcasts, so a True here covers both the seat that was never human and
    the human who drafted three rounds and wandered off.
  * `autodrafted IS NULL` IS KEPT. NULL means UNKNOWN, not True. Every
    backfilled pick carries NULL (`pipeline.mock_backfill` reads a `drafted`
    table that never had the column), so excluding NULL would discard
    essentially the whole corpus. This is a standing ruling; do not reverse
    it by "tightening" the filter. What DID change is how much NULL there is:
    for drafts recorded before the owner census, NULL is the usual answer and
    those picks are a mixture of people and ESPN's engine; for drafts
    recorded after it, NULL is rare and genuinely means unknown (the mTeam
    read failed, or the socket joined a draft already in progress).
  * `had_owner` is recorded per pick and `human_seats` per draft, and NEITHER
    is filtered on here. `had_owner IS FALSE` is a strict subset of
    `autodrafted IS TRUE` -- a seat with no owner autodrafts from pick one
    and nothing can flip it back, because a room's roster is fixed once the
    draft opens -- so a second exclusion would drop nothing and only add a
    way to get the filter wrong. They are stored so a later question can be
    asked of the corpus without re-deriving it ("fit only on rooms that had
    at least three people in them"; "is the 5am lobby emptier than the 8pm
    one"), which is a query, not a rule.

Every pick excluded for any reason is COUNTED and printed. A silent drop looks
exactly like a smaller corpus rather than like the join failure or filter it
actually is.

READ-ONLY, BOTH DATABASES. The corpus is the one thing in this project that
cannot be rebuilt from a source that still exists -- a mock draft that is not
written down when it happens is gone. This module opens it read-only and
never issues a write of any kind against `draft_log*`. The league database is
opened read-only too; all it is asked for is `weekly`/`players`, the universal
reference tables `scoring.player_history` reads.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from pipeline import db as db_mod
from pipeline import draft_log as dl
from scoring import league as league_mod
from scoring.board import adp_match_key
from scoring.draft_model import (ADP_BASELINE_TEMPERATURE, COLD_START_PRIOR,
                                 FEATURE_NAMES, LEGACY_FEATURE_NAMES,
                                 PickObservation, RUN_WINDOW,
                                 UNMEASURED_FEATURES, _ATTRIBUTE_DEFAULTS,
                                 _round_bucket, _ROUND_BUCKET_ORDER, _softmax,
                                 feature_matrix, fit)
from scoring.player_history import (assert_no_column_collision,
                                    attributes_as_of)

# Where the generated coefficient module lands. One place, so the writer, the
# "would have written" message on a loss, and the tests all name the same file.
PRIOR_MODULE = Path(__file__).resolve().parent.parent / "scoring" / "mock_prior.py"

# Where the write-up of a fit lives, by convention: findings are date-prefixed
# and this project has one document per measured question. The DATE IS FILLED
# IN FROM THE RUN, not hardcoded -- `make fit-prior` rewrites mock_prior.py
# with new coefficients every time a refit wins, and a pointer baked in at
# the time this line was written would send the reader of those new numbers
# to the write-up of a different fit on a different corpus. The generated
# module has to name the analysis of the numbers it is actually carrying.
FINDINGS_TEMPLATE = "docs/superpowers/findings/{date}-mock-corpus-features.md"

# `_ATTRIBUTE_DEFAULTS` above is imported, not repeated: it is the neutral
# value for every attribute a player's history does not supply, and
# `_enrich_pool` and `build_pool` already fill a missed join from it. A fourth
# copy of that table is a fourth thing to keep in step.


def open_corpus(path: str | None = None):
    """The corpus, READ-ONLY.

    Not `draft_log.corpus_conn`, which opens read-write and runs
    `ensure_schema` -- this module has no business creating or altering a
    table in the one database that cannot be rebuilt. Read-only also means a
    concurrently running farm loop (writing a new draft every ~15 minutes)
    cannot be blocked by this process.
    """
    return duckdb.connect(path or dl.CORPUS_PATH, read_only=True)


def open_league(path: str | None = None):
    """The league database, READ-ONLY, for its universal reference tables.

    `attributes_as_of` needs `weekly` and `players`. Those are UNIVERSAL
    tables in `pipeline.db`'s split, so every per-league file in
    `data/leagues/` carries the same copy of them and any one of them answers
    the question. That matters in practice: the deployment's own
    `data/nfl.duckdb` is frequently held by a running API process, and DuckDB
    will not open a file read-only while another process holds it read-write.
    Pass `--league-db data/leagues/<id>.duckdb` when that happens.
    """
    return duckdb.connect(path or db_mod.DEFAULT_PATH, read_only=True)


def snapshot_draft_ids(corpus) -> list:
    """Every draft this fit may use, fixed at the top of the run.

    THE FARM IS WRITING TO THIS DATABASE WHILE THIS RUNS. `make farm-mocks`
    joins live ESPN lobbies and adds roughly one draft every fifteen minutes,
    so "SELECT the drafts" issued twice inside one run can return two
    different corpora, and a backtest whose folds and whose training set
    disagree about what the corpus is is not a measurement. This is called
    once; every query afterwards is restricted to the ids it returned, and
    the list is printed so the run can be reproduced exactly.

    Only drafts that carry a POOL snapshot qualify. Reconstructing what was
    available at pick N means subtracting the first N-1 picks from the board
    as it stood at the start, and a draft recorded without that board (every
    `espn_history` row -- `draft_log.backfill_history` writes picks and no
    pool) has no choice set to offer. Sorted by id so the order is stable
    across runs rather than following physical row order.
    """
    return corpus.execute(
        """SELECT d.draft_id FROM draft_log d
           WHERE EXISTS (SELECT 1 FROM draft_log_pool o
                         WHERE o.draft_id = d.draft_id)
             AND EXISTS (SELECT 1 FROM draft_log_pick p
                         WHERE p.draft_id = d.draft_id)
           ORDER BY d.draft_id""").df()["draft_id"].tolist()


def attributes_by_player_id(league_conn, season: int) -> pd.DataFrame:
    """`attributes_as_of`, re-keyed from `adp_match_key` onto `player_id`.

    WHY A RE-KEY IS NEEDED AT ALL. `attributes_as_of` returns its rows keyed
    by `adp_match_key(name, position, team)`, because the two pools it was
    written for -- the historical fit pool and the live board -- both carry a
    player's NAME. A corpus pool does not: `draft_log_pool` stores
    `player_id`, `position`, `team`, `adp_rank`, `proj_points` and no name at
    all, since a name is the one thing a pick is never identified by
    downstream.

    WHY NOT JUST ADD `player_id` TO `attributes_as_of`'s OUTPUT. Because both
    of its existing callers merge it onto a frame that already has a
    `player_id` column, and pandas resolves a name held by both sides by
    suffixing them `_x`/`_y` -- silently. `assert_no_column_collision` exists
    to catch exactly that class of bug; widening `COLUMNS` would walk into it
    in `draft_sim.build_pool`, three lines from where the simulator reads
    `ranked["player_id"]`.

    So the map is rebuilt here, from the same table and by the same recipe
    `attributes_as_of` uses internally: a player's key is built from the LAST
    season he appears in before `season`, taking that season's last-recorded
    name, position and team. Written out rather than imported because it is
    four lines inside a hundred-line function; a test pins the two agreeing.

    The collision guard is the same one both production callers apply and for
    the same reason: `adp_match_key` carries no team for a non-DST position,
    so two different past players can share (position, normalized name), and
    guessing which one's numbers belong to the player actually drafted would
    put a wrong value on a `no_track_record: False` row. Both sides of a
    collision are dropped, so the pool row falls through to "unknown".

    A defense never matches, by construction and correctly: `draft_log_pool`
    identifies a DST by a synthesized `adp_<team>_defense` id that appears in
    no `weekly` row, so every DST reaches the feature matrix with the neutral
    attributes it would have reached it with anyway.
    """
    attrs = attributes_as_of(league_conn, season)
    if attrs.empty:
        return pd.DataFrame(columns=["player_id"] + [
            c for c in attrs.columns if c != "key"])
    attrs = attrs.drop_duplicates("key", keep=False)

    weekly = db_mod.read_table(league_conn, "weekly")
    prior = weekly[weekly["season"] < season] if not weekly.empty else weekly
    if prior.empty:
        return pd.DataFrame(columns=["player_id"] + [
            c for c in attrs.columns if c != "key"])
    per_season = prior.groupby(["player_id", "season"], as_index=False).agg(
        name=("player_display_name", "last"), position=("position", "last"),
        team=("recent_team", "last"))
    latest = per_season.sort_values("season").groupby(
        "player_id", as_index=False).last()
    latest["key"] = [adp_match_key(n, p, t) for n, p, t
                     in zip(latest["name"], latest["position"], latest["team"])]
    latest = latest[latest["key"].notna()][["player_id", "key"]]
    return latest.merge(attrs, on="key", how="inner").drop(columns=["key"])


def enrich_pool(pool_rows: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    """One draft's stored board, shaped into the pool `feature_matrix` reads.

    `market_rank` IS the stored `adp_rank`, renamed and not recomputed. That
    is not a convenience, it is the whole reason this fit is possible: the
    corpus pool was written by `draft_sim.build_pool`, whose `adp_rank` and
    `market_rank` are the SAME dense 1..k re-ranking of the cheat-sheet /
    FFC blend that `draft_model._enrich_pool` computes for the historical fit
    (`FFC_BLEND_WEIGHT`, imported by both so the weight cannot drift). So the
    board stored per draft is already on the scale the incumbent `reach` and
    `fall` coefficients were fitted against, and both the incumbent and the
    refit are read against one board rather than two. Re-deriving it today
    from a live ADP feed would produce a different board -- ADP moves daily
    -- and would silently score every pick against a market that did not
    exist when it was made.

    Everything else is the `_enrich_pool` treatment: attributes left-joined,
    misses filled with the neutral defaults, `no_track_record` True when the
    join found nothing, and `hype = prod_rank - market_rank` against the same
    dense rank both other implementations subtract.
    """
    pool = pool_rows.sort_values("adp_rank").reset_index(drop=True)
    pool = pool.rename(columns={"adp_rank": "market_rank"})
    if attrs.empty:
        for col, default in _ATTRIBUTE_DEFAULTS.items():
            pool[col] = default
    else:
        # The same guard both other callers run before this merge: pandas
        # resolves a column name held by both frames by suffixing them
        # `_x`/`_y` rather than failing, so a future pool column called
        # `trend` would silently stop `feature_matrix` reading the attribute
        # of that name. A corpus pool carries none of them today; this is
        # what keeps that true.
        assert_no_column_collision(pool)
        pool = pool.merge(attrs, on="player_id", how="left")
        pool["no_track_record"] = pool["no_track_record"].fillna(True).astype(bool)
        for col, default in _ATTRIBUTE_DEFAULTS.items():
            if col not in ("no_track_record", "prod_rank"):
                pool[col] = pool[col].fillna(default)
    pool["hype"] = pool["prod_rank"] - pool["market_rank"]
    return pool


def _is_autodrafted(value) -> bool:
    """True only for a recorded True. NULL is unknown and is NOT excluded.

    Three states reach this column and they are three different facts: True
    (ESPN's engine took the pick because a clock expired), False (a seat
    picked), and NULL (nobody recorded either way -- every backfilled pick,
    because a `drafted` table is only ever player_id and pick_no). Treating
    NULL as True would throw away 96 of every 128 picks in this corpus. See
    the module docstring.
    """
    return (not pd.isna(value)) and bool(value)


class CorpusObservations:
    """The fittable picks, their design matrices, and what was left out.

    One object rather than a tuple of six parallel lists, because every count
    in `dropped` has to travel with the observations it was dropped from --
    a report that says "3,162 picks" without saying what the other 38 were is
    the silent-drop failure this whole module is written against.
    """

    def __init__(self):
        self.observations = []
        self.draft_ids = []          # one draft_id per observation
        self.settings = {}           # draft_id -> LeagueSettings
        self.dropped = {"my_slot": 0, "autodrafted": 0, "not_in_pool": 0,
                        "already_taken": 0}
        self.picks_seen = 0


def build_corpus_observations(corpus, league_conn, draft_ids: list,
                              season: int | None = None) -> CorpusObservations:
    """Replay each draft into one `PickObservation` per fittable pick.

    THE POOL SNAPSHOT IS WHY NO REPLAY OF THE BOARD IS NEEDED. `draft_log`
    stores the board once per draft rather than the available set per pick --
    250 rows instead of 250 x 128 -- precisely because availability at pick N
    is the pool minus the first N-1 picks. That subtraction is this loop.

    BOOKKEEPING RUNS FOR EVERY PICK, FITTABLE OR NOT. A pick at our own seat,
    an autodrafted pick, and a pick whose player the pool does not carry all
    still removed a player from the board, still filled a roster slot on that
    team, and still counted toward the room's positional run. Skipping the
    bookkeeping for them would hand the NEXT pick a board that still contains
    a player who is gone. `draft_model.build_observations` follows the same
    rule for its own unmatched picks and says so.

    Seats, not managers. Every mock opponent is a stranger with no identity
    that survives the session (`draft_log.ANONYMOUS_PREFIX`), so the roster
    and pick history are tracked per SLOT within a draft and the owner key
    rides along only as a label. Nothing here fits a per-seat model: the
    cold-start prior is pooled by definition, since a league with no history
    has nothing to tell its opponents apart with.
    """
    out = CorpusObservations()
    if not draft_ids:
        return out

    heads = corpus.execute(
        "SELECT * FROM draft_log WHERE draft_id IN ({})".format(
            ",".join(["?"] * len(draft_ids))), draft_ids).df()
    pools = corpus.execute(
        "SELECT * FROM draft_log_pool WHERE draft_id IN ({})".format(
            ",".join(["?"] * len(draft_ids))), draft_ids).df()
    picks = corpus.execute(
        "SELECT * FROM draft_log_pick WHERE draft_id IN ({})".format(
            ",".join(["?"] * len(draft_ids))), draft_ids).df()

    # One attribute table for the whole corpus, not one per draft. Every draft
    # here is the same season, and `attributes_as_of` reads strictly earlier
    # seasons -- which are complete and final -- so a second call would return
    # the same frame at the cost of another second of groupby. If the corpus
    # ever spans seasons this becomes a per-season cache; the season is read
    # off the draft rather than assumed for that reason.
    seasons = sorted({int(s) for s in heads["season"].dropna()})
    attrs_by_season = {s: attributes_by_player_id(league_conn, s) for s in seasons}
    fallback_season = season or (seasons[0] if seasons else None)

    for draft_id in draft_ids:
        head = heads[heads["draft_id"] == draft_id]
        if head.empty:
            continue
        head = head.iloc[0]
        draft_season = (int(head["season"]) if not pd.isna(head["season"])
                        else fallback_season)
        settings = _settings_for(head)
        out.settings[draft_id] = settings

        pool = enrich_pool(pools[pools["draft_id"] == draft_id],
                           attrs_by_season.get(draft_season, pd.DataFrame()))
        row_of = {pid: i for i, pid in enumerate(pool["player_id"])}
        available = np.ones(len(pool), dtype=bool)

        my_slot = None if pd.isna(head["my_slot"]) else int(head["my_slot"])
        rosters, recent, last_pick_at = {}, [], {}

        draft_picks = picks[picks["draft_id"] == draft_id].sort_values("pick_no")
        for _, pick in draft_picks.iterrows():
            out.picks_seen += 1
            slot = int(pick["slot"]) if not pd.isna(pick["slot"]) else -1
            row = row_of.get(pick["player_id"])

            if row is None:
                # Counted, never silenced: the pool is this draft's own board
                # and a pick it does not contain is a join failure in the
                # recording, not a smaller corpus.
                out.dropped["not_in_pool"] += 1
            elif not available[row]:
                out.dropped["already_taken"] += 1
            elif my_slot is not None and slot == my_slot:
                out.dropped["my_slot"] += 1
            elif _is_autodrafted(pick["autodrafted"]):
                out.dropped["autodrafted"] += 1
            else:
                where = np.flatnonzero(available)
                out.observations.append(PickObservation(
                    season=draft_season, overall_pick=int(pick["pick_no"]),
                    manager=pick["owner_key"],
                    # `available` is a mask over the pool in board order, so
                    # the chosen player's index INSIDE the choice set is his
                    # position in that mask's true entries.
                    chosen=int(np.searchsorted(where, row)),
                    pool=pool.iloc[where].reset_index(drop=True),
                    roster=dict(rosters.get(slot, {})),
                    recent=list(recent[:RUN_WINDOW]),
                    last_pick_at_pos=dict(last_pick_at.get(slot, {}))))
                out.draft_ids.append(draft_id)

            if row is not None:
                available[row] = False
            position = pick["position"]
            rosters.setdefault(slot, {})
            rosters[slot][position] = rosters[slot].get(position, 0) + 1
            last_pick_at.setdefault(slot, {})[position] = int(pick["pick_no"])
            recent.insert(0, position)
    return out


def _settings_for(head) -> "league_mod.LeagueSettings":
    """This draft's roster shape, from the row it was recorded with.

    `settings_json` rides along on every corpus draft precisely so a later
    fit does not have to assume one (see `mock_backfill._mock_settings` for
    what is assumption and what is measurement in it). `feature_matrix` reads
    `teams`, `rounds` and `starters` from it: `need` is defined against the
    starter counts, and two of the roster-shape columns are divided by
    `rounds` so they mean "how far through the draft" in a 15-round league
    and a 16-round one alike.
    """
    blob = head.get("settings_json")
    if isinstance(blob, str) and blob:
        return league_mod.from_json(blob)
    return league_mod.default_settings()


def design(corpus_obs: CorpusObservations) -> tuple:
    """Feature matrices, chosen indices, fold labels, and each pick's board.

    `feature_matrix` is called -- not reimplemented -- so the vector this
    module fits is the vector `draft_sim._live_features` serves. Task 3's fix
    round made those two one definition with two callers; a third
    implementation here would undo it.

    Two things ride along beside the matrices. The board (`market_rank` per
    candidate) is for the ADP baseline, which must be scored on the SAME
    reference the model reads or the comparison confounds a better model with
    a worse yardstick. The round bucket is for the by-round table, which is
    where a headline win gets checked: a refit that gains only in the rounds
    where a kicker and a defense have to be taken has won something much
    smaller than its overall number says.
    """
    X_list, chosen, boards, buckets = [], [], [], []
    for obs, draft_id in zip(corpus_obs.observations, corpus_obs.draft_ids):
        settings = corpus_obs.settings[draft_id]
        X_list.append(feature_matrix(obs, settings))
        chosen.append(obs.chosen)
        boards.append(obs.pool["market_rank"].to_numpy(dtype=float))
        # `draft_model._round_bucket`, not a second set of boundaries. The
        # corpus report draws early/mid/late at 3/10 and the model draws them
        # at 3/8; this table is read next to the model's own backtest, so it
        # uses the model's split.
        buckets.append(_round_bucket(obs.overall_pick, settings.teams))
    return X_list, chosen, list(corpus_obs.draft_ids), boards, buckets


def score(beta, X_list, chosen, groups=None, buckets=None) -> dict:
    """Top-1, top-5 and log-loss of one coefficient vector over these picks.

    All three, always. A single reported metric is a metric that was chosen
    after the fact; the rule is that top-1 decides and the other two are
    published beside it so that reaching for one of them is visible.

    `groups`, when given, additionally returns each fold's own top-1, which
    is what the clustered standard error is computed from -- picks inside one
    draft are not independent, so the binomial standard error over 3,000
    picks understates the real uncertainty. `buckets`, when given, returns
    raw [top-1 hits, top-5 hits, n] per round bucket; raw counts rather than
    rates because the caller adds folds together.
    """
    beta = np.asarray(beta, dtype=float)
    hits1 = hits5 = 0
    ll = 0.0
    per_group, per_bucket = {}, {}
    for i, (X, k) in enumerate(zip(X_list, chosen)):
        probs = _softmax(X @ beta)
        order = np.argsort(-probs)
        hit1 = int(order[0] == k)
        hit5 = int(k in order[:5])
        hits1 += hit1
        hits5 += hit5
        ll += np.log(max(probs[k], 1e-12))
        if groups is not None:
            tally = per_group.setdefault(groups[i], [0, 0])
            tally[0] += hit1
            tally[1] += 1
        if buckets is not None:
            tally = per_bucket.setdefault(buckets[i], [0, 0, 0])
            tally[0] += hit1
            tally[1] += hit5
            tally[2] += 1
    n = len(chosen)
    if not n:
        return {"top1": 0.0, "top5": 0.0, "logloss": float("inf"), "n": 0,
                "hits1": 0, "hits5": 0, "ll": 0.0}
    # Raw counts ride along beside the rates so `leave_one_draft_out` can add
    # folds up exactly rather than by re-multiplying a rounded rate back out
    # by its fold size.
    report = {"top1": hits1 / n, "top5": hits5 / n, "logloss": -ll / n, "n": n,
              "hits1": hits1, "hits5": hits5, "ll": ll}
    if groups is not None:
        report["by_draft"] = {g: h / t for g, (h, t) in per_group.items()}
    if buckets is not None:
        report["by_bucket"] = per_bucket
    return report


def adp_baseline(boards, chosen) -> dict:
    """What the market alone predicts, as the floor any model has to clear.

    Identical in construction to `draft_model.backtest`'s baseline: a softmax
    over minus the player's 0-based place on `market_rank`, which is a real
    probability distribution over "who does the board think goes next". A
    uniform distribution is not a baseline -- it would give the board's #1 and
    its #250 the same probability, so "beats the market" would be true by
    construction.
    """
    hits1 = 0
    ll = 0.0
    for board, k in zip(boards, chosen):
        rankpos = np.argsort(np.argsort(board))
        hits1 += int(rankpos[k] == 0)
        probs = _softmax(-ADP_BASELINE_TEMPERATURE * rankpos)
        ll += np.log(max(probs[k], 1e-12))
    n = len(chosen)
    if not n:
        return {"top1": 0.0, "logloss": float("inf")}
    return {"top1": hits1 / n, "logloss": -ll / n}


def _sliced(X_list, keep_idx):
    """The design matrices restricted to `keep_idx` columns.

    A dropped feature has no coefficient at all and genuinely cannot
    influence a prediction -- it is not merely omitted from the report. Same
    discipline as `backtest(features=...)`, which is what makes an ablation a
    measurement rather than a relabelling.
    """
    if keep_idx is None:
        return X_list
    return [X[:, keep_idx] for X in X_list]


def leave_one_draft_out(X_list, chosen, groups, boards=None, keep_idx=None,
                        buckets=None) -> dict:
    """Refit without each draft in turn; score that draft with the result.

    The fold is the DRAFT because that is the unit of independence here: one
    room, one board, eight strangers, 128 picks that all see each other. A
    random split over picks would put the first half of a draft in training
    and the second half in test, where the model has already been told which
    players are gone.

    The fitted vector is the plain pooled maximum-likelihood fit -- no ridge,
    no prior -- which is what the incumbent `COLD_START_PRIOR` is and
    therefore the only thing comparable to it. `fit_all` produces its pooled
    vector the same way. Shrinking toward the incumbent would measure a blend
    rather than a refit, and shrinking toward zero would need a lambda chosen
    by a search this sample does not need: 23 coefficients against ~3,000
    picks is 130 observations per parameter.
    """
    X_list = _sliced(X_list, keep_idx)
    order = sorted(set(groups))
    hits1 = hits5 = scored = 0
    ll = 0.0
    per_draft, per_bucket = {}, {}
    for holdout in order:
        train = [i for i, g in enumerate(groups) if g != holdout]
        test = [i for i, g in enumerate(groups) if g == holdout]
        if not train or not test:
            continue
        beta = fit([X_list[i] for i in train], [chosen[i] for i in train])
        fold = score(beta, [X_list[i] for i in test], [chosen[i] for i in test],
                     buckets=None if buckets is None
                     else [buckets[i] for i in test])
        hits1 += fold["hits1"]
        hits5 += fold["hits5"]
        ll += fold["ll"]
        scored += fold["n"]
        per_draft[holdout] = fold["top1"]
        for bucket, tally in fold.get("by_bucket", {}).items():
            running = per_bucket.setdefault(bucket, [0, 0, 0])
            for j in range(3):
                running[j] += tally[j]
    if not scored:
        return {"top1": 0.0, "top5": 0.0, "logloss": float("inf"), "n": 0,
                "by_draft": {}}
    report = {"top1": hits1 / scored, "top5": hits5 / scored,
              "logloss": -ll / scored, "n": scored, "by_draft": per_draft}
    if buckets is not None:
        report["by_bucket"] = per_bucket
    if boards is not None:
        report["adp"] = adp_baseline(boards, chosen)
    return report


def cluster_se(by_draft: dict) -> float:
    """Standard error of top-1, treating each draft as one observation.

    sqrt(p(1-p)/n) over 3,000 picks is about 0.8pp, and it is wrong: those
    picks come from 25 rooms, and picks inside a room share a board, a set of
    opponents and everything already taken. The spread of per-draft top-1
    divided by sqrt(number of drafts) is the honest scale for judging whether
    a difference of half a point means anything.
    """
    values = np.array(list(by_draft.values()), dtype=float)
    if len(values) < 2:
        return float("nan")
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def paired_se(a: dict, b: dict) -> float:
    """Standard error of the DIFFERENCE in top-1, paired by draft.

    The right uncertainty for the decision, and it is not either vector's own
    standard error. Both are scored on the same 3,000 picks in the same 26
    rooms, so most of what makes one draft harder than another -- how deep the
    board ran, how much the room reached -- hits both sides equally and
    cancels in the difference. Comparing two unpaired standard errors instead
    would charge the comparison for variation that is common to both.
    """
    shared = sorted(set(a) & set(b))
    if len(shared) < 2:
        return float("nan")
    diffs = np.array([a[g] - b[g] for g in shared], dtype=float)
    return float(diffs.std(ddof=1) / np.sqrt(len(diffs)))


def feature_indices(names) -> list:
    """Column indices for `names`, in FEATURE_NAMES order.

    Order matters and is not the caller's to choose: a beta fitted with the
    columns in one order and applied with them in another is silently wrong.
    """
    wanted = set(names)
    return [i for i, name in enumerate(FEATURE_NAMES) if name in wanted]


def ablate(X_list, chosen, groups, candidates=None, full=None) -> pd.DataFrame:
    """Each candidate feature's contribution, one dropped at a time.

    `candidates` defaults to `UNMEASURED_FEATURES` -- the eight columns Task 3
    added and nothing has ever measured. This is the whole point of running
    the ablation here rather than in `make fit-managers`, whose own
    `ablation()` defaults to the four older features and would report a clean
    table that tested none of these eight.

    Read `delta_top1` as "top-1 accuracy the full model has that the model
    without this feature does not". Positive earns a place; at or below zero
    is cut. `delta_top5` is reported beside it, always, and does not decide.
    """
    candidates = list(UNMEASURED_FEATURES if candidates is None else candidates)
    # `full` is the all-columns row. The caller may pass one it already has:
    # each row here is a complete leave-one-draft-out backtest, and computing
    # the same one twice in a run is 25 needless fits.
    full = full if full is not None else leave_one_draft_out(X_list, chosen, groups)
    rows = [{"dropped": "none", "top1": full["top1"], "top5": full["top5"],
             "logloss": full["logloss"], "delta_top1": 0.0, "delta_top5": 0.0}]
    for feature in candidates:
        keep = feature_indices([f for f in FEATURE_NAMES if f != feature])
        cut = leave_one_draft_out(X_list, chosen, groups, keep_idx=keep)
        rows.append({"dropped": feature, "top1": cut["top1"],
                     "top5": cut["top5"], "logloss": cut["logloss"],
                     "delta_top1": full["top1"] - cut["top1"],
                     "delta_top5": full["top5"] - cut["top5"]})
    return pd.DataFrame(rows)


def shipped_features(ablation: pd.DataFrame) -> list:
    """The feature set a winning refit would ship: the 15, plus the winners.

    A candidate whose `delta_top1` is at or below zero is CUT, and cut means
    its coefficient is 0.0 in the shipped vector -- not "fitted and small".
    That is the same rule `_NEW_FEATURES` was decided by and the same rule
    that rejected the stat profile at n=696, applied to a larger sample
    rather than bent for one.

    The 15 legacy columns are not re-litigated here. They are what every
    number on record was measured against, this corpus is one season of mock
    drafts rather than six seasons of a real league, and a feature set is not
    something to re-derive from scratch on every new sample.
    """
    kept = ablation[(ablation["dropped"] != "none")
                    & (ablation["delta_top1"] > 0)]["dropped"].tolist()
    return [f for f in FEATURE_NAMES
            if f in set(LEGACY_FEATURE_NAMES) | set(kept)]


def full_beta(X_list, chosen, keep_names) -> np.ndarray:
    """The vector that ships: pooled over every draft, zero where cut.

    Fitted on the selected columns only, then scattered back into a
    full-length vector with 0.0 in the cut positions. Fitting all 23 and then
    zeroing the losers would ship coefficients that were fitted in the
    presence of columns the shipped model does not have -- every remaining
    coefficient would carry mass that the cut ones were splitting with it.

    Full length, always, so `len(COLD_START_PRIOR) == len(FEATURE_NAMES)`
    holds and nothing downstream needs to know which features were cut.
    """
    keep_idx = feature_indices(keep_names)
    gamma = fit(_sliced(X_list, keep_idx), chosen)
    beta = np.zeros(len(FEATURE_NAMES))
    beta[keep_idx] = gamma
    return beta


# ---------------------------------------------------------------- the module

_MODULE_TEMPLATE = '''"""The cold-start prior's coefficients. GENERATED -- do not edit by hand.

`scoring.draft_model` imports `PRIOR` from here and binds it to
`COLD_START_PRIOR`, the coefficient vector every opponent in a league with no
draft history is simulated with. In a mock draft, where all eight opponents
are strangers, this vector IS the entire model.

Written by `pipeline/fit_prior.py` (`make fit-prior`), and ONLY when the fit
it describes beat the incumbent on held-out top-1 accuracy. There is no way
to write a losing vector here short of editing that rule, which is the point.
Hand-editing a number here would launder a guess as a measurement; refit
instead.

The previous vector, and the provenance of its numbers, is in this file's git
history.

PROVENANCE OF THESE NUMBERS

{provenance_prose}
"""
import numpy as np

# What was fitted, when, and against what. Machine-readable so a reader does
# not have to trust the prose above and a test can assert the two agree.
PROVENANCE = {provenance_dict}

# The feature order these coefficients were fitted in, written out rather
# than assumed. `draft_model` asserts this equals `FEATURE_NAMES`: a
# coefficient vector applied to columns in a different order than it was
# fitted in is silently wrong, and the length check alone cannot see it.
FEATURES = {features}

# The draft_ids this was fitted on, recorded because the corpus GROWS -- a
# farm loop adds roughly one draft every fifteen minutes -- so "the corpus"
# is not a reproducible input without the list.
DRAFT_IDS = {draft_ids}

PRIOR = np.array([
{coefficients}])

assert len(PRIOR) == len(FEATURES), (
    "the generated prior and its feature list disagree -- regenerate with "
    "`make fit-prior` rather than editing either by hand")
'''


def render_module(beta, features, draft_ids, provenance,
                  provenance_prose: str, cut=()) -> str:
    """The text of `scoring/mock_prior.py`.

    Each coefficient is written on its own line with its feature name beside
    it, in `FEATURE_NAMES` order, because the one thing anyone ever wants
    from this file is to read what a named coefficient came out as -- and
    because a diff between two generations should show which coefficients
    moved, not one very long line.
    """
    cut = set(cut)
    width = max(len(name) for name in features)
    lines = []
    for name, value in zip(features, beta):
        # A 0.0 that was CUT and a 0.0 that happened to fit to zero are
        # different facts, and only the cut list can tell them apart -- so the
        # note is driven by that list, never by the value.
        note = "   <- cut on delta_top1, not fitted" if name in cut else ""
        lines.append(f"    {value:+.6f},  # {name:<{width}}{note}".rstrip())
    return _MODULE_TEMPLATE.format(
        provenance_prose=provenance_prose,
        provenance_dict=_pretty_dict(provenance),
        features=_pretty_list(features),
        draft_ids=_pretty_list(draft_ids),
        coefficients="\n".join(lines) + "\n")


def _pretty_list(values, indent: str = "    ") -> str:
    body = ",\n".join(f"{indent}{v!r}" for v in values)
    return "[\n" + body + ",\n]"


def _pretty_dict(values, indent: str = "    ") -> str:
    body = ",\n".join(f"{indent}{k!r}: {v!r}" for k, v in values.items())
    return "{\n" + body + ",\n}"


def write_prior(beta, features, draft_ids, provenance, provenance_prose,
                path: Path | None = None, cut=()) -> Path:
    """Write the generated module. Callers must have checked the rule first.

    Deliberately dumb: it writes what it is given. The decision lives in
    `main`, in one place, where it can be read next to the numbers that make
    it -- rather than in a flag on a writer, where a future caller could pass
    a different one.
    """
    target = Path(path or PRIOR_MODULE)
    target.write_text(render_module(beta, features, draft_ids, provenance,
                                    provenance_prose, cut=cut))
    return target


# ------------------------------------------------------------------ the run

def _prose(refit, incumbent, keep, draft_ids) -> str:
    cut = [f for f in UNMEASURED_FEATURES if f not in keep]
    return f"""
Fitted on {len(draft_ids)} mock drafts from the cross-league draft corpus
({dl.CORPUS_PATH}), {refit['n']} picks, on {_now()}.

Leave-one-DRAFT-out for the refit -- every draft in this corpus is one
season, so leave-one-season-out is degenerate here. The incumbent is not
refitted at all: it was fitted on a different league's six seasons and has
never seen one pick of this corpus, so every pick is already held out for it.
Both rows score the same picks.

    incumbent  top-1 {incumbent['top1']:.4f}  top-5 {incumbent['top5']:.4f}  log-loss {incumbent['logloss']:.4f}
    refit      top-1 {refit['top1']:.4f}  top-5 {refit['top5']:.4f}  log-loss {refit['logloss']:.4f}

top-1 is the decision metric and it is the one this had to win on. The other
two are recorded because they were measured, not because they decided.

Of the {len(UNMEASURED_FEATURES)} features this fit measured for the first time, {len(keep) - len(LEGACY_FEATURE_NAMES)} earned a place
on delta_top1 and {len(cut)} did not. The cut ones carry 0.0 here, which means
MEASURED AND REJECTED on this corpus -- a different fact from the 0.0 they
carried before, which meant not yet measured.

    kept: {', '.join(f for f in keep if f in UNMEASURED_FEATURES) or '(none)'}
    cut:  {', '.join(cut) or '(none)'}

The numbers and what they do not establish, written up beside this fit:
{FINDINGS_TEMPLATE.format(date=_now())}
""".strip("\n")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _print_metrics(label, report) -> None:
    se = (f"  +/-{cluster_se(report['by_draft']):.4f} (clustered by draft)"
          if report.get("by_draft") else "")
    print(f"  {label:<22} top-1 {report['top1']:.4f}  top-5 {report['top5']:.4f}"
          f"  log-loss {report['logloss']:.4f}{se}")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    corpus_path = _option(argv, "--corpus")
    league_path = _option(argv, "--league-db")

    corpus = open_corpus(corpus_path)
    league_conn = open_league(league_path)
    try:
        draft_ids = snapshot_draft_ids(corpus)
        if not draft_ids:
            print("The corpus holds no draft with a pool snapshot -- run "
                  "`make mock-backfill` or `make farm-mocks` first.")
            return 1
        print(f"Corpus snapshot: {len(draft_ids)} drafts, taken at the start "
              "of this run.\n  " + "\n  ".join(draft_ids))

        corpus_obs = build_corpus_observations(corpus, league_conn, draft_ids)
        X_list, chosen, groups, boards, buckets = design(corpus_obs)
        print(f"\n{corpus_obs.picks_seen} picks in those drafts; "
              f"{len(chosen)} fitted. Excluded:")
        for reason, n in corpus_obs.dropped.items():
            print(f"  {reason:<14} {n}")
        if not chosen:
            print("No fittable picks. Nothing to measure.")
            return 1

        # The incumbent is scored on every pick with no folds and no refit,
        # which is not a head start: it was fitted on a different league's six
        # seasons and has never seen one pick of this corpus. Every pick is
        # already held out for it.
        incumbent = score(COLD_START_PRIOR, X_list, chosen, groups=groups,
                          buckets=buckets)
        market = adp_baseline(boards, chosen)

        print("\nLeave-one-draft-out. The fold is the draft because that is the "
              "unit of\nindependence: one room, one board, eight strangers. The "
              "incumbent row is not\nrefitted -- it never saw this corpus, so "
              "every pick is already held out for it.")
        _print_metrics("incumbent prior", incumbent)
        print(f"  {'ADP baseline':<22} top-1 {market['top1']:.4f}"
              f"                    log-loss {market['logloss']:.4f}")

        all_23 = leave_one_draft_out(X_list, chosen, groups)
        legacy = leave_one_draft_out(X_list, chosen, groups,
                                     keep_idx=feature_indices(LEGACY_FEATURE_NAMES))
        _print_metrics("refit, legacy 15", legacy)
        _print_metrics("refit, all 23", all_23)

        print("\nPer-feature ablation over the eight columns Task 3 added and "
              "nothing has\never measured. delta_top1 > 0 earns a place; at or "
              "below zero is cut.")
        table = ablate(X_list, chosen, groups, full=all_23)
        print(table.to_string(index=False))

        keep = shipped_features(table)
        # Re-run even when the shipped set is all 23, so this row carries the
        # by-round table `all_23` was not asked for. Same folds, same picks.
        refit = leave_one_draft_out(X_list, chosen, groups, buckets=buckets,
                                    keep_idx=feature_indices(keep))
        print("\nThe configuration a win would ship: 15 legacy + "
              f"{len(keep) - len(LEGACY_FEATURE_NAMES)} of 8 new.")
        _print_metrics("refit, shipped set", refit)

        # WHERE the accuracy moved, not just whether it did. The 2026-08-11
        # finding turned on exactly this table: the stat profile lost nine of
        # its eleven picks in the late rounds, where the position dummies
        # carry the K/DST structure `market_rank` cannot. A refit that wins
        # only in the rounds where a kicker and a defense have to be taken has
        # won something much smaller than the headline number suggests.
        print("\nBy round bucket (early 1-3, mid 4-8, late 9+), incumbent "
              "against the\nshipped refit, on the same held-out picks:")
        print(f"  {'bucket':<7} {'n':>5}  {'incumbent top-1':>15} "
              f"{'refit top-1':>12}  {'incumbent top-5':>15} {'refit top-5':>12}")
        for bucket in _ROUND_BUCKET_ORDER:
            a = incumbent["by_bucket"].get(bucket)
            b = refit["by_bucket"].get(bucket)
            if not a or not b:
                continue
            print(f"  {bucket:<7} {a[2]:>5}  {a[0] / a[2]:>15.4f} "
                  f"{b[0] / b[2]:>12.4f}  {a[1] / a[2]:>15.4f} "
                  f"{b[1] / b[2]:>12.4f}")

        # THE RULE. delta_top1 against the incumbent, and nothing else. It is
        # written as one expression, once, so that changing it is a one-line
        # diff somebody has to justify rather than a flag somebody can pass.
        delta_top1 = refit["top1"] - incumbent["top1"]
        se = paired_se(refit["by_draft"], incumbent["by_draft"])
        print(f"\ndelta_top1 (shipped refit - incumbent): {delta_top1:+.4f} "
              f"+/-{se:.4f}\n(paired by draft: both vectors scored the same "
              "picks in the same rooms, so what\nmakes one room harder than "
              "another cancels in the difference)")
        if delta_top1 <= 0:
            print("\nThe refit does NOT beat the incumbent on top-1. Nothing is "
                  "written;\nscoring/mock_prior.py is unchanged. A negative "
                  "result is a complete result --\nwrite it up rather than "
                  "tuning until something wins.")
            return 1

        beta = full_beta(X_list, chosen, keep)
        provenance = {
            "fitted_at": _now(),
            "corpus": dl.CORPUS_PATH,
            "drafts": len(draft_ids),
            "picks_fitted": len(chosen),
            "picks_excluded": dict(corpus_obs.dropped),
            "folds": "leave-one-draft-out",
            "incumbent_top1": round(incumbent["top1"], 6),
            "refit_top1": round(refit["top1"], 6),
            "refit_top5": round(refit["top5"], 6),
            "refit_logloss": round(refit["logloss"], 6),
            "features_cut": [f for f in UNMEASURED_FEATURES if f not in keep],
        }
        target = write_prior(beta, FEATURE_NAMES, draft_ids, provenance,
                             _prose(refit, incumbent, keep, draft_ids),
                             cut=provenance["features_cut"])
        print(f"\nThe refit wins on top-1. Wrote {target}.")
        return 0
    finally:
        corpus.close()
        league_conn.close()


def _option(argv: list, name: str, default=None):
    """`--name value`, without pulling argparse in for two options."""
    if name in argv:
        i = argv.index(name)
        if i + 1 < len(argv):
            return argv[i + 1]
    return default


if __name__ == "__main__":
    sys.exit(main())
