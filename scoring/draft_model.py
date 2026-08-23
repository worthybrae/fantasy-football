"""Per-manager draft-pick model.

Each historical pick is treated as one choice from the set of players who were
available at that moment. That set is reconstructable: we know the full pick
order and that season's ADP pool, so subtracting everything taken before pick
N gives the pool the manager was actually choosing from.

Picks whose player has no ADP row that season are dropped from fitting. The
model's core feature is a player's position relative to market rank, which is
undefined without one -- and the import step reports how many picks this
removes per season so a bad join surfaces as a number, not a silent shrug.
"""
import json
import warnings
from itertools import repeat
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pipeline.db import read_table, write_table
from scoring import league as league_mod
from scoring import mock_prior
from scoring.board import (GAMES, _ADP_POSITION_ALIASES, _norm_name,
                          adp_match_key)
from scoring.player_history import assert_no_column_collision, attributes_as_of

RUN_WINDOW = 5

# How much of the market reference comes from FFC ADP rather than ESPN's
# preseason cheat sheet. Fitted here and applied in `draft_sim.build_pool`,
# which MUST use the same value: reach/fall are learned against this board
# and mean something else against any other. See the table in `_enrich_pool`
# for the measurement that chose it.
FFC_BLEND_WEIGHT = 0.25

_ATTRIBUTE_DEFAULTS = {"age": np.nan, "no_track_record": True,
                       "prod_rank": np.nan, "trend": 0.0,
                       # The stat-profile columns. They were carried on the
                       # pool and read by no feature for a year, precisely so
                       # that re-measuring them would be a change to
                       # FEATURE_NAMES rather than a rebuild of the join.
                       # That is what has now happened -- see
                       # _STAT_PROFILE_FEATURES below.
                       #
                       # The 2026-08-11 finding
                       # (docs/superpowers/findings/2026-08-11-stat-profile
                       # -vs-position-dummies.md) MEASURED THEM AND REJECTED
                       # THEM, and nothing here contradicts it: at n=696
                       # picks they cost top-1 accuracy against the position
                       # dummies, and that remains the answer for that
                       # sample. They are back in the matrix because the
                       # corpus is no longer n=696 -- see
                       # _STAT_PROFILE_FEATURES for the sample that licenses
                       # re-opening the question and for who decides it.
                       "usage": np.nan, "efficiency": np.nan,
                       "played_share": np.nan, "peak_gap": np.nan}


def _enrich_pool(conn, pool: pd.DataFrame, season: int,
                 cheatsheet: pd.DataFrame = None) -> pd.DataFrame:
    """Attach the market reference and the player-at-pick-time attributes.

    `market_rank` is ESPN's printable preseason cheat sheet (`PPR300`),
    dense-ranked 1..N per season, falling back to the Fantasy Football
    Calculator `adp_rank` for any season or player it does not cover. This
    is the same reference `draft_sim.build_pool` ranks the live board on,
    and that identity is the point: `reach`/`fall` are fitted here and
    applied there, so the two must mean the same thing.

    The cheat sheet is what the room actually saw -- the league drafts off
    ESPN -- and, unlike ESPN's `kona_player_info` API, it is a genuine
    *preseason* snapshot. The API is deliberately NOT the reference here
    (Task 9 measured this; do not re-litigate it without re-measuring):

    - It serves no `draftRanksByRankType.PPR.rank` at all for 2020-2022, so
      half of a six-season fit would silently fall back to FFC anyway and
      `market_rank` would mean two different things across the training set.
    - Its 2023 board is contaminated with hindsight: it correlates 0.905
      with FFC's *2024* ADP against 0.678 with 2023's own, 9 of the top 10
      against 5, and ranks Kyren Williams 7th when he was a late-round
      flier that preseason. The players are correctly 2023; the ranks are
      not. 2024 does not show this, so the contamination is specific to
      2023 rather than uniform across ESPN's history.

    Using the API ranks cost ~7 points of top-1 accuracy, which is why the
    `historic_espn` table they were imported into is gone: nothing read it
    once this became the reference, and six network calls per import to a
    discredited board is not provenance worth paying for. The live board's
    Mkt column is unaffected -- `scoring.market` reads `espn_adp`, which
    `pipeline.refresh` writes from the current-season API. That is a
    different table and a legitimate use: "where does ESPN have this player
    right now" makes no claim about any past preseason.
    """
    pool = pool.copy()
    if cheatsheet is None or cheatsheet.empty:
        pool["market_rank"] = pool["adp_rank"]
    else:
        season_cs = cheatsheet[cheatsheet["season"] == season].copy()
        season_cs["key"] = _match_keys(season_cs, "cs_name")
        season_cs = season_cs.dropna(subset=["key"]).sort_values(
            "cs_rank").drop_duplicates("key", keep="first")
        ranks = season_cs.set_index("key")["cs_rank"]
        mapped = pool["key"].map(ranks)
        # Dense 1..N over the pool, cheat-sheet order first and the players
        # it never ranked continuing the sequence behind them in FFC order.
        # Re-ranking rather than passing `cs_rank` through is what keeps this
        # on `adp_rank`'s scale: the cheat sheet stops at 300 while a
        # season's ADP pool is longer, so a raw cs_rank of 300 and an
        # adp_rank of 300 would otherwise describe different depths.
        order = pool.assign(_cs=mapped).sort_values(
            ["_cs", "adp_rank"], na_position="last").index
        cs_dense = pd.Series(
            np.arange(1, len(pool) + 1, dtype=float), index=order).reindex(pool.index)
        # Blend in the FFC ADP order. The cheat sheet is what the room reads,
        # and alone it beats FFC ADP outright (top-1 0.2356 to 0.2011), but
        # it is one publication's opinion and it is occasionally idiosyncratic
        # -- in 2026 it has Jeremiyah Love 13th where 5,789 real PPR mock
        # drafts have him 25.9, and Chase Brown 21st where they have him 12.2.
        # FFC ADP is the opposite kind of evidence: not an opinion at all, but
        # a record of what thousands of drafters actually did.
        #
        # Weighted 3:1 toward the cheat sheet, which is where the measurement
        # puts the optimum. Six-season LOSO, n=696:
        #
        #     w_ffc   top-1     top-5     log-loss
        #     0.00    0.2356    0.5848    2.8681      <- cheat sheet alone
        #     0.25    0.2514    0.6250    2.7903      <- here
        #     0.50    0.2270    0.5977    2.7957
        #     0.75    0.2155    0.5776    2.8802
        #     1.00    0.2011    0.5560    2.9819      <- FFC alone
        #
        # All three metrics peak together, which noise-chasing does not
        # usually do. The weight was still chosen by looking at the number it
        # is reported against, so it was re-checked by holding each season out
        # of the selection set: 0.25 beats 0 on top-1 and top-5 in 6 of 6.
        #
        # Re-dense after blending so "rank k" keeps meaning "the kth player on
        # this board" -- reach/fall are log-rank differences and a weighted
        # average of two ranks is not itself a rank.
        ffc_dense = pool["adp_rank"].rank(method="first")
        pool["market_rank"] = (
            (1.0 - FFC_BLEND_WEIGHT) * cs_dense
            + FFC_BLEND_WEIGHT * ffc_dense).rank(method="first")

    attrs = attributes_as_of(conn, season)
    if not attrs.empty:
        # `adp_match_key` carries no team for a non-DST position, so two
        # different past players can share (position, normalized name) --
        # a real risk with common names, not a contrived one. Guessing
        # which one's numbers belong to the player actually drafted would
        # put a wrong value on a "no_track_record: False" row; left-joining
        # against a key that appears twice would also duplicate the pool
        # row, breaking "one pick removes exactly one pool row". Drop both
        # sides of the collision so it falls through to "unknown" instead.
        attrs = attrs.drop_duplicates("key", keep=False)
    if attrs.empty:
        for col, default in _ATTRIBUTE_DEFAULTS.items():
            pool[col] = default
    else:
        assert_no_column_collision(pool)
        pool = pool.merge(attrs, on="key", how="left")
        pool["no_track_record"] = pool["no_track_record"].fillna(True).astype(bool)
        for col, default in _ATTRIBUTE_DEFAULTS.items():
            if col not in ("no_track_record", "prod_rank"):
                pool[col] = pool[col].fillna(default)

    # Positive means the market is ahead of what the player has actually
    # done -- taking him is a leap of faith. NaN when he has no production
    # to rank, which `feature_matrix` reads as neutral rather than as zero.
    pool["hype"] = pool["prod_rank"] - pool["market_rank"]
    return pool


def _match_keys(frame: pd.DataFrame, name_col: str, team_col: str = "team") -> list:
    """`board.adp_match_key` for every row.

    `team_col` differs by table: `historic_adp` carries `team`, `draft_picks`
    carries `nfl_team`. A missing column (a historic_adp written before it
    carried one) yields no team, which makes DSTs unmatchable rather than
    wrongly matched.
    """
    teams = (frame[team_col] if team_col in frame.columns
             else pd.Series([None] * len(frame), index=frame.index))
    return [adp_match_key(name, position, team) for name, position, team
            in zip(frame[name_col], frame["position"], teams)]


class PickObservation(NamedTuple):
    season: int
    overall_pick: int
    manager: str
    chosen: int
    pool: pd.DataFrame
    roster: dict
    recent: list
    # position -> `overall_pick` of the PICKING TEAM's most recent pick at
    # that position, before this one. `roster` says how many they hold and
    # `recent` says what the whole room just took; neither can answer "how
    # long has this manager left running back alone", which is what
    # `rounds_since_pos` is. Stored as an overall pick number rather than
    # a round so `build_observations` stays free of `settings` --
    # `feature_matrix`
    # already derives the round from `overall_pick` and `settings.teams`,
    # and deriving it in one place is what keeps the two agreeing.
    #
    # Defaulted so every existing construction site (and every test fixture)
    # keeps working and reads as "this team has taken nothing yet", which is
    # the neutral answer rather than a wrong one. The shared default dict is
    # never mutated: `build_observations` copies before storing and
    # `feature_matrix` only reads.
    last_pick_at_pos: dict = {}


def build_observations(conn) -> list:
    picks = read_table(conn, "draft_picks")
    teams = read_table(conn, "draft_teams")
    adp = read_table(conn, "historic_adp")
    cheatsheet = read_table(conn, "historic_espn_cs")
    if picks.empty or teams.empty or adp.empty:
        return []

    picks = picks.merge(teams[["season", "team_id", "manager"]],
                        on=["season", "team_id"], how="left")
    picks = picks.assign(norm=picks["player_name"].map(_norm_name))
    # Defensive: a historic_adp written before import_league normalized it
    # still says "PK" for kickers, which would leave pos_K an all-zero,
    # unidentified column in the feature matrix even once the join is keyed
    # correctly. Same alias table the board applies to the live ADP feed.
    adp = adp.assign(norm=adp["adp_name"].map(_norm_name),
                     position=adp["position"].replace(_ADP_POSITION_ALIASES))
    adp = adp.assign(key=_match_keys(adp, "adp_name"))
    picks = picks.assign(key=_match_keys(picks, "player_name", "nfl_team"))

    out = []
    for season, season_picks in picks.groupby("season"):
        pool = adp[adp["season"] == season][["norm", "position", "adp_rank", "key"]]
        # A DST with no team has no join key at all (see adp_match_key); it
        # is unmatchable, and leaving it in would let two of them match each
        # other.
        pool = pool[pool["key"].map(lambda k: k is not None)]
        # historic_adp has no uniqueness guarantee on the join key --
        # parse_adp builds it straight from the external payload, and
        # import_league writes it without deduping. Collapse to the
        # best-known (lowest) adp_rank per player before anything else, so
        # "one pick removes exactly one pool row" holds structurally rather
        # than by accident of the fixture. Same precedent as
        # scoring.board._dedupe_adp.
        pool = pool.sort_values("adp_rank").drop_duplicates(
            "key", keep="first").reset_index(drop=True)
        pool = _enrich_pool(conn, pool, season, cheatsheet)
        available = pool.copy()
        rosters, recent = {}, []
        # manager -> {position: overall_pick of their last pick there}. One
        # extra dict of at most six entries per manager, updated in O(1) per
        # pick and shallow-copied onto the observation exactly like `roster`
        # already is -- `build_observations` is ~10s against the owner's own
        # league and this must not be what makes it eleven.
        last_pick_at = {}
        for _, pick in season_picks.sort_values("overall_pick").iterrows():
            key = pick["key"]
            match = (available.index[available["key"] == key]
                     if key is not None else available.index[[]])
            if len(match):
                reset = available.reset_index(drop=True)
                chosen = int(reset.index[reset["key"] == key][0])
                out.append(PickObservation(
                    season=int(season), overall_pick=int(pick["overall_pick"]),
                    manager=pick["manager"], chosen=chosen, pool=reset,
                    roster=dict(rosters.get(pick["manager"], {})),
                    recent=list(recent[:RUN_WINDOW]),
                    last_pick_at_pos=dict(last_pick_at.get(pick["manager"], {}))))
                # Belt and braces: the pool is deduped above so `match`
                # should never carry more than one label, but drop only the
                # first to guarantee the pool never shrinks by more than one
                # row for a single pick even if that invariant is ever
                # violated upstream.
                available = available.drop(index=match[:1])
            # Bookkeeping runs unconditionally, matched or not: a pick with
            # no ADP row that season produces no observation, but it still
            # consumed a roster spot and still counts toward positional runs.
            rosters.setdefault(pick["manager"], {})
            rosters[pick["manager"]][pick["position"]] = (
                rosters[pick["manager"]].get(pick["position"], 0) + 1)
            last_pick_at.setdefault(pick["manager"], {})[pick["position"]] = (
                int(pick["overall_pick"]))
            recent.insert(0, pick["position"])
    return out


# QB is the dropped baseline: with a full set of position dummies plus an
# intercept-free softmax the columns would be collinear.
_POSITION_DUMMIES = ["RB", "WR", "TE", "K", "DST"]
# These four are what survived `ablation()`. Five were proposed; `volatility`
# measured at -0.0029 against the cheat sheets and was cut. Re-measured on a
# clean re-import (Task 9, six-season LOSO, n=696), every survivor's
# delta_top1 is positive, in picks recovered out of 696:
#
#     age +1.29pp (9 picks)   no_track_record +1.29pp (9)
#     hype +0.72pp (5)        trend +0.29pp (2)
#
# Read those with the standard error in mind -- se(top-1) is 1.6pp at this
# n, so `trend` in particular is "not shown to hurt" rather than "shown to
# help", and the same feature flipped sign across the three references this
# was measured against (contaminated ESPN API, FFC, cheat sheets).
#
# THE RULE, stated once so it cannot be applied selectively: delta_top1 is
# the decision metric and a feature is cut at or below zero on it. delta_top5
# is reported alongside because a feature can trade one against the other,
# but it is not a tiebreak -- reaching for it only when it defends the
# incumbent is how a table stops being a measurement.
#
# Re-measured after LAMBDA_GRID stopped being truncated (the grid feeds
# `backtest`'s per-manager fits, so the ablation moves with it), n=696:
#
#     dropped           delta_top1        delta_top5
#     age               +0.29pp (2)       -1.29pp (9)
#     no_track_record    0.00pp (0)       +0.43pp (3)
#     hype              +0.29pp (2)       -0.43pp (3)
#     trend             +0.29pp (2)       -0.14pp (1)
#
# Two of these are now uncomfortable and both are left in place, deliberately
# and with the reason stated rather than by omission:
#
# - `no_track_record` sits exactly ON the cut line on top-1. "At zero" at a
#   2-pick resolution is not the same finding as "below zero", and cutting on
#   a number that cannot tell the two apart is a coin flip dressed as a rule.
# - `age` is the mirror image: it is the only survivor top-5 actively argues
#   against (dropping it recovers 9 picks of top-5 while costing 2 of top-1).
#   The rule says top-1 decides, so it stays -- but it is not the clean
#   "+1.29pp, earns its place" the earlier note claimed.
#
# Both are the first things to re-measure when a seventh season lands.
_NEW_FEATURES = ["age", "no_track_record", "hype", "trend"]

# What the PICKING TEAM has already built, always crossed with the candidate.
#
# THE CONSTRAINT THAT SHAPES ALL FOUR, stated here because it is the one a
# well-meaning edit breaks: this is a conditional logit. The softmax runs over
# the candidates inside ONE choice set, so a column holding the same value for
# every candidate in that set cancels out of the likelihood exactly -- not
# approximately, not weakly, but algebraically, because a constant added to
# every score divides straight back out of the softmax. "It is round 6" and
# "this team already holds three running backs" are facts about the PICK.
# Neither can enter the model on its own, however predictive it sounds.
#
# Each therefore enters as an interaction with the candidate under
# consideration. Not "the team holds three RBs" but "this candidate is an RB
# and the team already holds three". `qb_early`/`te_early` above are the same
# pattern -- a round fact multiplied by a position dummy -- and these follow
# their shape. `tests/test_draft_model.py` asserts the cancellation property
# directly against `log_likelihood`, because nothing else in the suite does.
#
# `need` was already this shape and stays: these generalize it rather than
# replace it. `need` is binary and clips at the starter count, so it cannot
# tell a manager's second running back from his fifth.
_ROSTER_SHAPE_FEATURES = ["held_at_pos", "first_at_pos", "rounds_since_pos",
                          "first_at_pos_round"]

# The stat profile, re-enabled for re-measurement. See `_ATTRIBUTE_DEFAULTS`.
#
# These four were measured against the position dummies on 2026-08-11 at
# n=696 picks and REJECTED on `delta_top1`, the decision metric stated below.
# That finding stands and is not being re-litigated by argument: it is being
# re-measured on a sample it explicitly licenses re-measurement against. The
# mock corpus is 23 drafts / 2,944 picks -- four times the evidence, and
# eight strangers per draft rather than the same eight managers six times --
# and its position mix by round bucket (early RB 55.6% / WR 36.4% / TE 6.9% /
# QB 1.1%; late WR 22.3% / RB 18.3% / DST 16.8% / K 16.7% / QB 15.2% /
# TE 10.8%) is a different distribution from the one the rejection was
# measured on.
#
# THE ANSWER, from `make fit-prior` on 2026-08-23, n=3,267: they split.
# `usage` is the single most valuable of Task 3's eight columns
# (delta_top1 +0.0077, 25 picks) and ships in the cold-start prior.
# `efficiency`, `played_share` and `peak_gap` are cut -- the same three the
# 2026-08-11 finding lost on, now rejected on four times the sample as well.
# All four stay in `FEATURE_NAMES`: a cut feature is one whose PRIOR
# coefficient is 0.0, not a column removed from the matrix, and the per-manager
# fit is still free to find something in it. See
# docs/superpowers/findings/2026-08-23-mock-corpus-features.md.
_STAT_PROFILE_FEATURES = ["usage", "efficiency", "played_share", "peak_gap"]

# WHAT THE BOARD ALREADY COMPUTES AND THE MODEL COULD NOT SEE.
#
# The shipped prior is roughly "who is next on the board, with small
# corrections": `reach` carries -11.24 and everything describing the player
# himself sums to about 1.6. Meanwhile `build_board` computes 35 columns per
# player and `feature_matrix` reads ten of them. These six are the gap --
# five facts the board (or the pool, or `settings`) already knows, plus one
# summary of the pool itself.
#
# ALL SIX ARE CANDIDATE-SPECIFIC, which is not a stylistic preference but the
# entry requirement stated above `_ROSTER_SHAPE_FEATURES`: a column holding
# one value for every candidate in a choice set divides straight back out of
# the softmax and contributes exactly nothing. Two of them earn that the hard
# way -- `last_of_tier` is a fact about the POOL made candidate-specific by
# asking it of the candidate's own tier, and `slots_left_at_pos` is a fact
# about the PICKING TEAM indexed by the candidate's own position, the same
# shape `need` and the `_ROSTER_SHAPE_FEATURES` block already have.
#
# `dropoff_at_pos` IS THE SPEC'S `pos_dropoff`, RENAMED, and the rename is
# not cosmetic. `SUMMARY_FEATURES` below excludes every `pos_`-prefixed name
# because a position dummy cannot be shown one bar at a time, and
# `tests/test_api.py` asserts `/api/managers`'s `shown` against its own copy
# of that predicate. A feature called `pos_dropoff` would be silently dropped
# from the manager card by a rule written for something else. `held_at_pos`
# and `rounds_since_pos` were renamed off `pos_count`/`pos_gap` for exactly
# this reason and the note there says to do the same for the next one; this
# is the next one. `_at_pos` also matches `slots_left_at_pos` beside it.
#
# THE ONE DEFINITION THAT HAD TO BE PINNED DOWN is `dropoff_at_pos`. The
# tempting version -- his points over the best player expected to SURVIVE to
# this team's next turn -- is circular: survival comes out of `draft_sim`'s
# rollouts, which run this model, so the feature would be fitted on the
# output of the thing being fitted and no backtest of it would mean anything.
# What ships is pure pool state: his projection minus the projection of the
# next-best player still AVAILABLE at his position, at that moment. No model
# output, no rollout, nothing imported from `draft_sim`.
#
# EXPECT MOST OF THESE TO LOSE. Five of the eight above lost.  `vor` and
# `dropoff_at_pos` are both correlated with market rank, and a collinear
# addition to a model with a -11.24 coefficient on `reach` tends to split a
# coefficient rather than add signal. They are listed in
# `UNMEASURED_FEATURES` so `ablation(candidates=...)` decides, on
# `delta_top1`, by the rule stated above `_NEW_FEATURES`.
_POOL_SIGNAL_FEATURES = ["dropoff_at_pos", "vor", "durability", "proj_change",
                         "last_of_tier", "slots_left_at_pos"]

# The FLEX-eligible positions, defined here rather than in `draft_sim` (which
# now imports this name) because `slots_left_at_pos` needs them on the FITTING
# side and `draft_sim` importing `draft_model` is the direction the dependency
# already runs. One definition, two callers, for the same reason
# `roster_shape_features` has one.
FLEX_POSITIONS = ("RB", "WR", "TE")

# Everything added since the last measurement of THIS LEAGUE's fit, in one
# list, so the thing that has to be measured is enumerable rather than
# remembered. `ablation` takes it as `candidates`.
#
# THE NAME IS STILL ACCURATE, AND IT IS WORTH SAYING WHY. `make fit-prior`
# measured the first eight on the mock corpus on 2026-08-23 and cut five,
# so `COLD_START_PRIOR` now carries three fitted values and five zeros that
# mean MEASURED AND REJECTED
# (docs/superpowers/findings/2026-08-23-mock-corpus-features.md). That is a
# result about 26 rooms of strangers drafting an 8-team PPR mock. It is NOT a
# result about the eight managers in this league, whose per-manager fits come
# from `fit_all` against `draft_picks` and still fit all 23 coefficients with
# no evidence at all about these eight. `ablation()`'s default is still
# `_NEW_FEATURES` for that reason; pass this list to change it.
#
# The six `_POOL_SIGNAL_FEATURES` are unmeasured EVERYWHERE -- no corpus fit
# and no league fit has seen them -- so they carry 0.0 in `COLD_START_PRIOR`
# and that zero means NOT YET MEASURED rather than MEASURED AND REJECTED.
# Being on this list is what gets them measured: `pipeline/fit_prior.ablate`
# takes it as its default `candidates`, so a column appended to
# `FEATURE_NAMES` and left off this list would ride into the matrix on the
# next refit without ever being asked to justify itself.
UNMEASURED_FEATURES = (_ROSTER_SHAPE_FEATURES + _STAT_PROFILE_FEATURES
                       + _POOL_SIGNAL_FEATURES)

# The 15 features every backtest number on record was measured against.
# Appending to FEATURE_NAMES rather than inserting is not cosmetic: `backtest`
# and `fit_subset` slice the design matrix by INDEX, so the old columns have
# to keep the indices they had for a `features=` mask to mean the same thing
# it did before. A test pins this prefix for exactly that reason.
LEGACY_FEATURE_NAMES = (["reach", "fall"]
                        + [f"pos_{p}" for p in _POSITION_DUMMIES]
                        + ["qb_early", "te_early", "need", "run"]
                        + _NEW_FEATURES)
FEATURE_NAMES = LEGACY_FEATURE_NAMES + UNMEASURED_FEATURES
EARLY_ROUNDS = 3

# Coefficients the rail's three-bar manager card may show. Everything except
# the position dummies, which only mean anything relative to each other and
# so cannot be summarized honestly one bar at a time.
#
# Derived from FEATURE_NAMES, not hand-listed, and served per-coefficient by
# /api/managers as `shown` so the web client has nothing to keep in sync. The
# hand-kept copy in DraftRail.tsx went stale the moment this branch added
# `age`/`hype`/`trend`/`no_track_record`: the filter runs before the top-3
# slice, so a manager whose strongest deviation was on a new feature got bars
# for weaker ones instead. `_PHRASES` above had the identical bug.
#
# THE `pos_` PREFIX IS RESERVED FOR THE POSITION DUMMIES, and this predicate
# is why. A `pos_`-prefixed name is excluded from the card because a dummy
# only means anything relative to the other four, so it cannot be shown one
# bar at a time -- the prefix is load-bearing, not decorative.
#
# `tests/test_api.py` asserts `/api/managers`'s `shown` against its own copy
# of this same predicate. Widening it here to let a non-dummy through would
# drift the two apart, which is exactly the staleness the DraftRail.tsx note
# above describes happening once already. So a new feature that describes the
# ROSTER rather than being a position dummy must be NAMED so it does not
# collide: `held_at_pos` and `rounds_since_pos` were drafted as `pos_count`
# and `pos_gap` and renamed for this reason. Do the same for the next one.
SUMMARY_FEATURES = [f for f in FEATURE_NAMES if not f.startswith("pos_")]

# Divisor that puts `hype` on roughly the same scale as the other columns,
# so no coefficient has to be tiny or huge to matter. Not fitted -- changing
# it just rescales the coefficient -- but keeping the columns comparable
# makes the ridge penalty treat them even-handedly.
HYPE_SCALE = 50.0

# WHAT SCALE THE COLUMNS ADDED IN TASK 3 SIT ON, and what was done about it.
#
# Ridge penalizes every coefficient by the same lambda, so a column on a
# wildly different scale is effectively regularized differently: a feature
# measured in rounds needs a coefficient ten times smaller than one measured
# in dummies to say the same thing, and the penalty charges it for that.
# HYPE_SCALE exists for exactly this reason. So each new column was checked
# against the band the shipped ones already occupy rather than assumed into
# it. Standard deviation after `_centre_within_position`, measured over
# `attributes_as_of` for 2025 and 2026 (QB/RB/WR/TE, n=1,598 and 1,721):
#
#     age  4.44 / 4.67      <- shipped, unscaled, the widest incumbent
#     trend  1.95 / 1.85    <- shipped, unscaled
#     usage  5.56 / 5.41    peak_gap  3.88 / 4.01
#     efficiency  0.79 / 0.73    played_share  0.31 / 0.31
#
# So all four stat-profile columns land inside the band `age` and `trend`
# already define, and they are passed through unscaled. Dividing them would
# have put them on a DIFFERENT scale from the centred column they are most
# directly comparable against, which is the opposite of the goal.
#
# The roster-shape columns are handled per column:
#
# - `held_at_pos` is a raw count. It is bounded by the rounds already played
#   and in practice reaches about 5 on the deepest position of a 16-round
#   roster, so it sits inside the band above unscaled.
# - `first_at_pos` is a 0/1 dummy like the five position dummies.
# - `rounds_since_pos` is the ONLY one that needed a divisor. Measured in
#   rounds it spans 0-15 -- an order of magnitude wider than a dummy and
#   wider than anything above -- so it is divided by `settings.rounds` to
#   land in 0-1.
# - `first_at_pos_round` is divided by `settings.rounds` for the same reason,
#   which is also what the brief specifies.
#
# Dividing by `settings.rounds` rather than by a constant keeps the column
# meaning "how far through the draft", which is the same quantity in a
# 15-round league and a 16-round one. A flat divisor would not be.


def _log_rank_features(market_rank, pick_no):
    """Rounds-of-reach on a log axis.

    Linear rank made the gap from rank 1 to 5 (0.5 units) ten times smaller
    than the gap from 100 to 140 (5.0), so the fit spent its range on
    deep-bench noise and left the entire top of the board within ~2x of
    itself in probability -- a coin flip for the first pick of the draft.
    Log rank inverts that, matching how drafts actually behave. log1p so
    rank 1 and pick 1 are finite; no division by `teams`, which was a
    rounds-based scaling that means nothing on a log axis.
    """
    delta = np.log1p(market_rank) - np.log1p(pick_no)
    return np.maximum(0.0, delta), np.maximum(0.0, -delta)


class PositionGroups(NamedTuple):
    """`positions` reduced to integer group codes, computed once per call.

    `labels` are the distinct positions; `codes[i]` is candidate i's index
    into them, or -1 for a candidate with no position at all.
    """
    labels: np.ndarray
    codes: np.ndarray


def position_groups(positions) -> PositionGroups:
    """Group `positions` once so several within-position columns can share it.

    This is a hot-path structure. `draft_sim._live_features` runs inside every
    rollout step and now centres five columns (`age` plus the four
    stat-profile ones) on top of the roster-shape block. Grouped per column
    with a boolean mask per position, those eight new columns cost +87us on
    a 158us call over a 250-candidate pool -- the function more than doubled.
    Grouping ONCE here and reducing with `bincount` instead, they cost +4.8us
    on 98.2us (1.05x); at 60 candidates, +11.9us on 71.5us (1.20x). Nothing
    was approximated to get there: the numbers are identical either way.

    `pd.factorize`, not `np.unique(return_inverse=True)`: 3x faster on the
    object-dtype arrays a pool actually carries, and it TOLERATES a missing
    position (code -1) where `np.unique` raises comparing a float NaN against
    a string. Not sorted, which nothing here needs.

    Derived from `positions` itself rather than from a fixed vocabulary, so a
    position the vocabulary does not know about still gets its own group
    instead of silently falling into the zeros.
    """
    codes, labels = pd.factorize(np.asarray(positions))
    return PositionGroups(np.asarray(labels), codes)


def _by_group(groups: PositionGroups, per_label, default: float = 0.0):
    """One value per label, scattered back out to one value per candidate.

    The `default` bucket is prepended and the codes shifted by one, so a
    candidate with no position (code -1) lands on it instead of silently
    wrapping around to the last label.
    """
    table = np.concatenate(([default], np.asarray(per_label, dtype=float)))
    return table[groups.codes + 1]


def _centre_within_position(values, positions, groups=None):
    """Age relative to typical for the position, so the coefficient reads as
    'younger than his peers' rather than tracking that tight ends last
    longer than running backs. Unknown ages are neutral, not young.

    `groups` is an optional pre-built `position_groups(positions)`; passing
    it is purely a performance hoist and cannot change the result.

    Two `bincount`s rather than a gather per position group: same answer,
    a fixed handful of O(n) passes instead of one per group, and measured
    5.6x faster on a 250-candidate pool. A candidate with a non-finite value
    or no position contributes to no mean and comes out at 0.0, which is the
    same neutral the masked version produced.
    """
    values = np.asarray(values, dtype=float)
    if groups is None:
        groups = position_groups(positions)
    codes = groups.codes
    known = np.isfinite(values) & (codes >= 0)
    n_labels = len(groups.labels)
    sums = np.bincount(codes[known], weights=values[known], minlength=n_labels)
    counts = np.bincount(codes[known], minlength=n_labels)
    # `maximum(counts, 1)` only guards a division by zero for a group with no
    # finite value at all; every row in such a group is excluded by `known`
    # below, so the 0.0 mean it produces is never actually subtracted.
    means = _by_group(groups, sums / np.maximum(counts, 1))
    return np.where(known, values - means, 0.0)


def roster_shape_features(positions, roster, last_pick_at_pos, round_no,
                          teams, rounds, groups=None) -> tuple:
    """The four `_ROSTER_SHAPE_FEATURES` columns, in `FEATURE_NAMES` order.

    ONE IMPLEMENTATION, TWO CALLERS, AND THAT IS THE WHOLE POINT.
    `feature_matrix` is what the model is FITTED on; `draft_sim._live_features`
    is what it is SERVED from. If those two disagree about what a column
    means, the coefficient learned for `held_at_pos` gets applied at draft
    time to a number that is not `held_at_pos`, and every prediction the live
    room makes is computed from a feature vector the fit never saw. That
    failure is silent -- the shapes still line up and nothing raises -- so it
    is designed out here rather than policed by a test alone.
    `tests/test_draft_sim.py` still asserts the two agree end to end.

    Both callers arrive with the same four things and nothing else is
    needed: the candidates' positions, the picking team's position counts,
    the pick number of that team's last pick at each position, and where in
    the draft we are.

    Looked up once per POSITION and gathered out to the candidates, not
    computed per candidate: `_live_features` runs inside every rollout step,
    and `_legal_mask`'s docstring records what a per-element comprehension
    costs there. At most six dict lookups plus two O(n) gathers.

    Returns `(held, first_at_pos, rounds_since, first_at_pos_round)`.
    """
    if groups is None:
        groups = position_groups(positions)
    held = _by_group(groups, [float(roster.get(pos, 0))
                              for pos in groups.labels])
    # Same `(pick - 1) // teams + 1` the caller derived `round_no` with, so
    # there is one definition of "round" rather than two.
    rounds_since = _by_group(groups, [
        0.0 if last_pick_at_pos.get(pos) is None
        else (round_no - ((last_pick_at_pos[pos] - 1) // teams + 1)) / rounds
        for pos in groups.labels])
    # `first_at_pos` is a nonlinear transform of `held_at_pos`, not a linear
    # one, so the pair is identifiable and the fit is well posed -- but the
    # two ARE entangled and will trade coefficient mass, and so will
    # `rounds_since_pos`, which is 0.0 on exactly the candidates where
    # `first_at_pos` is 1.0. Read no one of the three alone.
    #
    # `delta_top1` settled it on the mock corpus (2026-08-23, n=3,267) and cut
    # two of the three: `held_at_pos` (the count) survives, `first_at_pos`
    # (the flag) and `rounds_since_pos` (the wait) do not, and what survives of
    # "this position is empty" is `first_at_pos_round` -- the flag crossed
    # with how far through the draft it is, the only one of them that says
    # WHEN an empty slot starts to matter. All four columns stay here: a cut
    # feature is one whose cold-start coefficient is 0.0, not a column removed
    # from the matrix.
    first_at_pos = (held == 0.0).astype(float)
    return held, first_at_pos, rounds_since, first_at_pos * (round_no / rounds)


# WHAT SCALE THE `_POOL_SIGNAL_FEATURES` COLUMNS SIT ON, measured against the
# band the note above defines (std 0.31 to 5.56 after centring). Two of the
# six arrive an order of magnitude outside it and are divided down; the rest
# already land inside and are passed through, because dividing a column that
# is already comparable just makes its coefficient less readable.
#
# Measured over `data/draft_corpus.duckdb` (60 drafts, 14,654 within-position
# adjacent pairs) and a `build_board` for the current season (250 rows):
#
#     proj_points gap  std 9.53   -> / GAMES = 0.56   <- dropoff_at_pos
#     vor              std 75.1   -> / GAMES = 4.42
#     durability       std 19.9   -> (x - 50)/50 = 0.40
#     proj_change      std 3.98      passed through
#     last_of_tier     0/1 dummy
#     slots_left_at_pos  0-4, the same shape as `held_at_pos`
#
# GAMES IS THE RIGHT DIVISOR FOR THE FIRST TWO, not an arbitrary constant.
# `proj_points` and `vor` are both SEASON totals, and dividing either by the
# games in a season turns it into the per-game quantity the design document
# actually names ("his projected points per game minus..."). A flat divisor
# chosen to hit the band would land on the same scale by accident; this one
# lands there and still means something out loud.
#
# `durability` is a within-position PERCENTILE (see
# `factors.normalize_within_position`), so it averages 50 by construction.
# Subtracting the midpoint before scaling is what makes 0.0 read as "median
# durability" -- which matters because 0.0 is also what an unknown player
# gets, and "unknown" must not be served to the model as "worst in the
# league". Note this is NOT an availability rate: `draft_sim.roster_value`
# needs the rate and reads a different quantity for exactly that reason.
DURABILITY_MIDPOINT = 50.0
DURABILITY_SCALE = 50.0

# Where `_last_of_tier` parks a candidate whose tier is unknown, and the
# widest integer tier its counting fast path will bincount over. -1.0 sits
# below every real tier (they are ranks, so 1 and up) and the bound is far
# past the ~15 a real board draws, so both exist to make the fast path's
# assumption checkable rather than to constrain anything real.
_UNKNOWN_TIER = -1.0
_MAX_TIER_BUCKETS = 1024


def _pool_signal(pool: pd.DataFrame, name: str) -> np.ndarray:
    """One `_POOL_SIGNAL_FEATURES` input column, or all-NaN if the pool has none.

    UNLIKE every other column `feature_matrix` reads, these five inputs
    (`proj_points`, `vor`, `durability`, `proj_change`, `tier`) are genuinely
    absent from one of the three pools this model is fitted on, and that is a
    fact about the data rather than a fixture convenience:

    - the mock corpus carries `proj_points` on every pool row and joins the
      other four off the current board (`fit_prior.enrich_pool`);
    - `draft_sim.build_pool` carries all five, straight off `build_board`;
    - THIS LEAGUE's own history (`build_observations` -> `_enrich_pool`) has
      none of them. Its pools are reconstructed from `historic_adp` for
      seasons 2020-2025, and no preseason board for those seasons exists to
      join -- rebuilding one today from current data would be hindsight, which
      is the same objection `_enrich_pool` records against ESPN's API ranks.

    So the honest answer for those seasons is "unknown", and NaN is how this
    model already spells it: every consumer below reads a non-finite value as
    the neutral 0.0. A column that is 0.0 for every candidate cancels out of
    the softmax exactly (see `_ROSTER_SHAPE_FEATURES`), so those six seasons
    contribute no evidence about these six coefficients rather than six
    seasons of fabricated evidence.
    """
    if name in pool.columns:
        return pool[name].to_numpy(dtype=float)
    return np.full(len(pool), np.nan)


def _finite_or_neutral(values: np.ndarray) -> np.ndarray:
    """Every non-finite entry replaced by 0.0, the neutral this model spells
    "nothing known" with.

    `np.where(isfinite)` rather than `np.nan_to_num`, which is 3x slower here
    (6.7us against 2.1us on 250 candidates, measured) because it walks the
    array once per kind of non-finite value. Three of these run on every
    `_live_features` call, which runs once per pick per rollout.

    +/-inf is not hypothetical: `draft_sim.build_pool` fills an unranked
    player's `vor` with -inf as a sentinel for its candidate SELECTION, and
    that value reaching a softmax would make every other candidate's
    probability zero.
    """
    return np.where(np.isfinite(values), values, 0.0)


def _next_best_gap(points, groups: PositionGroups) -> np.ndarray:
    """Each candidate's projection minus the next-best AVAILABLE one at his
    position -- the drop from him to the man behind him.

    Not "minus the best player expected to survive to my next turn". That
    definition needs `draft_sim.survival`, which runs rollouts through this
    very model, so the feature would be a function of the fit being measured
    and no backtest of it could mean anything. This one is pure pool state:
    the choice set, the positions in it, and the projections on it.

    The last available player at a position has nobody behind him and gets
    0.0, which is the same neutral an unknown projection gets -- correctly,
    since "there is no drop after him" and "the drop is unknown" are both
    "this column has nothing to say about him".

    ONE lexsort and three O(n) passes, no per-candidate Python. `_live_features`
    runs inside every rollout step (`_legal_mask`'s docstring records what a
    comprehension costs there), and a groupby-per-position would be one pass
    per position on the hot path. A NaN projection sorts to the end of its own
    group, so it can only zero out its own entry and its immediate
    predecessor's rather than corrupting the whole group.
    """
    points = np.asarray(points, dtype=float)
    codes = groups.codes
    n = len(points)
    if n == 0:
        return np.zeros(0)
    # Position groups first, best projection first inside each group.
    order = np.lexsort((-points, codes))
    ranked = points[order]
    ranked_codes = codes[order]
    gap = np.zeros(n)
    if n > 1:
        same_group = ranked_codes[:-1] == ranked_codes[1:]
        gap[:-1] = np.where(same_group, ranked[:-1] - ranked[1:], 0.0)
    out = np.empty(n)
    out[order] = gap
    # One pass for both exclusions rather than a `nan_to_num` and a `where`.
    # A non-finite gap is an unknown projection somewhere in the pair, and a
    # candidate with no position at all (code -1) was grouped with the other
    # positionless rows by the sort above, which would have him "dropping off"
    # against players he shares nothing with. Both are neutral.
    return np.where(np.isfinite(out) & (codes >= 0), out, 0.0)


def _last_of_tier(tier) -> np.ndarray:
    """1.0 for a candidate who is the only one left in his market tier.

    Scarcity as the board already draws it. `build_board`'s `tier` is a
    global break over the whole board rather than a per-position one, so
    "the last of tier 4" is a claim about the market's own grouping and not
    a restatement of `dropoff_at_pos`.

    A candidate whose tier is unknown gets 0.0 -- "not known to be the last
    one" rather than a guess in either direction -- and unknown tiers are
    never counted as a tier of their own, which would make two players with
    no tier look like a pair.
    """
    tier = np.asarray(tier, dtype=float)
    if len(tier) == 0:
        return np.zeros(0)
    known = np.isfinite(tier)
    # Counted over the WHOLE array and masked afterwards, rather than over a
    # boolean-indexed copy: same answer, one fewer allocation per call on a
    # function that runs once per pick per rollout. The unknown rows are all
    # parked on one sentinel and `known` removes them either way, which is
    # what stops two players with no tier from reading as a pair.
    parked = np.where(known, tier, _UNKNOWN_TIER)
    codes = parked.astype(np.intp) + 1
    if not (codes.min() >= 0 and codes.max() < _MAX_TIER_BUCKETS
            and np.array_equal(codes - 1.0, parked)):
        # A tier that is not a small non-negative integer. `build_board` never
        # produces one -- `tier` is a rank -- so this is the fixture and
        # future-proofing path, and it is a fallback rather than the default
        # because the general version is twice the cost: 18.9us against 9.2us
        # on 250 candidates, measured. Same answer either way; the guard above
        # is what makes that true rather than assumed.
        _, codes = np.unique(parked, return_inverse=True)
    counts = np.bincount(codes)
    return ((counts[codes] == 1) & known).astype(float)


def pool_signal_features(positions, points, vor, durability, proj_change,
                         tier, roster, starters, flex_slots: int,
                         groups=None) -> tuple:
    """The six `_POOL_SIGNAL_FEATURES` columns, in `FEATURE_NAMES` order.

    ONE IMPLEMENTATION, TWO CALLERS, for the reason `roster_shape_features`
    states at length: `feature_matrix` is what the model is FITTED on and
    `draft_sim._live_features` is what it is SERVED from, and a disagreement
    between them applies a fitted coefficient to a different quantity at
    draft time without anything raising. Every scaling decision -- the
    `GAMES` divisors, the durability midpoint, what an unknown value becomes
    -- therefore lives HERE, once, rather than being written out twice and
    kept in step by hand.

    `points`, `vor`, `durability`, `proj_change` and `tier` are the raw board
    quantities; non-finite entries are the "unknown" both pools spell with
    NaN and all of them come out at the neutral 0.0. `vor` additionally
    tolerates the -inf `draft_sim.build_pool` fills an unranked player with,
    which is a sentinel for its candidate SELECTION and would otherwise reach
    the softmax as an infinite score.

    Returns `(dropoff_at_pos, vor, durability, proj_change, last_of_tier,
    slots_left_at_pos)`.
    """
    if groups is None:
        groups = position_groups(positions)

    dropoff = _next_best_gap(points, groups) / GAMES
    vor = _finite_or_neutral(np.asarray(vor, dtype=float) / GAMES)
    durability = _finite_or_neutral(
        (np.asarray(durability, dtype=float) - DURABILITY_MIDPOINT)
        / DURABILITY_SCALE)
    proj_change = _finite_or_neutral(np.asarray(proj_change, dtype=float))

    # How many starting slots at this candidate's position this team still
    # has to fill, FLEX included. A generalization of `need`, which is binary
    # and so cannot tell "needs two more receivers" from "needs one" -- and
    # which ignores the FLEX slots entirely, so a team with both its RB slots
    # full reads as having no use for a running back at all.
    #
    # A FLEX slot is only counted as open once the surplus at the flex
    # positions has been subtracted: a team holding three RBs against two RB
    # slots has already spent one of them. Counted across the flex positions
    # together, because the slot does not belong to any one of them.
    #
    # Indexed by the CANDIDATE's own position, which is what makes this a
    # candidate-specific column rather than a fact about the pick that would
    # cancel out of the softmax.
    flex_used = sum(max(0.0, float(roster.get(pos, 0)) - starters.get(pos, 0))
                    for pos in FLEX_POSITIONS)
    flex_left = max(0.0, float(flex_slots) - flex_used)
    slots_left = _by_group(groups, [
        max(0.0, starters.get(pos, 0) - float(roster.get(pos, 0)))
        + (flex_left if pos in FLEX_POSITIONS else 0.0)
        for pos in groups.labels])

    return (dropoff, vor, durability, proj_change, _last_of_tier(tier),
            slots_left)


def feature_matrix(obs: PickObservation, settings) -> np.ndarray:
    pool = obs.pool
    n = len(pool)
    ranks = pool["market_rank"].to_numpy(dtype=float)
    positions = pool["position"].to_numpy()
    teams = max(settings.teams, 1)

    reach, fall = _log_rank_features(ranks, obs.overall_pick)

    columns = [reach, fall]
    for pos in _POSITION_DUMMIES:
        columns.append((positions == pos).astype(float))

    # `overall_pick` is 1-indexed, so pick 8 in an 8-team league is round 1.
    round_no = (obs.overall_pick - 1) // teams + 1
    early = 1.0 if round_no <= EARLY_ROUNDS else 0.0
    columns.append((positions == "QB").astype(float) * early)
    columns.append((positions == "TE").astype(float) * early)

    starters = settings.starters
    need = np.array([1.0 if obs.roster.get(p, 0) < starters.get(p, 0) else 0.0
                     for p in positions])
    columns.append(need)

    recent = obs.recent[:RUN_WINDOW]
    run = np.array([recent.count(p) / RUN_WINDOW for p in positions])
    columns.append(run)

    # One grouping pass for every within-position column below (`age` plus
    # the four stat-profile ones) and for the roster-shape block.
    groups = position_groups(positions)

    columns.append(_centre_within_position(pool["age"].to_numpy(), positions,
                                           groups))
    columns.append(pool["no_track_record"].to_numpy().astype(float))
    hype = pool["hype"].to_numpy(dtype=float)
    columns.append(np.nan_to_num(hype, nan=0.0) / HYPE_SCALE)
    columns.append(pool["trend"].to_numpy(dtype=float))

    # --- `_ROSTER_SHAPE_FEATURES`: what this team already holds, crossed
    # with the candidate's own position so the column varies inside the
    # choice set. See that constant for why a column that does not vary is
    # not a weak feature but no feature at all, and `roster_shape_features`
    # for why the arithmetic lives there rather than here.
    #
    # `roster` is indexed BY the candidate's own position, so two candidates
    # at different positions read different values off the same roster --
    # that indexing IS the interaction.
    #
    # `rounds_since_pos` is 0.0 when this team has never taken the position,
    # and in this data that cannot mean anything else: every team picks
    # exactly once per round in a snake, so their previous pick at a position
    # is at least a full round back and the smallest real value is 1/rounds.
    # A format giving one team two picks in a round would collide the two
    # readings; none of the drafts fitted here does.
    columns.extend(roster_shape_features(
        positions, obs.roster, obs.last_pick_at_pos, round_no, teams,
        max(settings.rounds, 1), groups))

    # --- `_STAT_PROFILE_FEATURES`: what the candidate has actually done.
    # Centred within position for the same reason `age` is -- so a
    # coefficient reads as a preference between comparable players rather
    # than re-learning that running backs carry the ball and receivers do
    # not, which the position dummies already say. Unknown stays neutral:
    # `_centre_within_position` leaves a non-finite value at 0.0, which is
    # also what a pool whose attribute join found nothing produces (see
    # `_ATTRIBUTE_DEFAULTS`), so "no history" reads the same either way.
    for name in _STAT_PROFILE_FEATURES:
        columns.append(_centre_within_position(
            pool[name].to_numpy(dtype=float), positions, groups))

    # --- `_POOL_SIGNAL_FEATURES`: what the board computes and what is left on
    # it. Same arrangement as the roster-shape block above -- the arithmetic,
    # the scaling and the treatment of an unknown value all live in
    # `pool_signal_features` so that `draft_sim._live_features` reads them off
    # the same definition rather than a second one that agrees today.
    #
    # `_pool_signal` rather than `pool[name]`: this league's own history has
    # no preseason board for 2020-2025 to join, so all five inputs are absent
    # there and NaN -- read as neutral -- is the honest answer. See that
    # helper.
    columns.extend(pool_signal_features(
        positions, _pool_signal(pool, "proj_points"), _pool_signal(pool, "vor"),
        _pool_signal(pool, "durability"), _pool_signal(pool, "proj_change"),
        _pool_signal(pool, "tier"), obs.roster, starters, settings.flex_slots,
        groups))

    return np.column_stack(columns) if n else np.zeros((0, len(FEATURE_NAMES)))


def _shifted_exp(scores: np.ndarray) -> np.ndarray:
    """exp(scores - scores.max()), the max-subtracted exponentiation shared by
    softmax normalization and the log-sum-exp term. Both need it; computing it
    once and reusing it (rather than each recomputing it from scores) halves
    the exp() calls per observation on the hot path fit() iterates over."""
    return np.exp(scores - scores.max())


def _softmax(scores: np.ndarray) -> np.ndarray:
    exp = _shifted_exp(scores)
    return exp / exp.sum()


def log_likelihood(beta, X_list, chosen_list) -> float:
    total = 0.0
    for X, k in zip(X_list, chosen_list):
        scores = X @ beta
        exp = _shifted_exp(scores)
        log_sum_exp = scores.max() + np.log(exp.sum())
        total += scores[k] - log_sum_exp
    return float(total)


def neg_log_likelihood(beta, X_list, chosen_list, prior=None, lam=0.0,
                       offsets=None, weights=None):
    """Value and gradient of the ridge-penalized negative log-likelihood.

    Convex in beta, which is why L-BFGS-B finds the global optimum rather than
    a local one -- the reason this uses scipy instead of a hand-rolled loop.

    `offsets`, when given, is one per choice set and is added to that set's
    scores without being differentiated. It is how `fit_subset` holds most of
    a manager's coefficients at their pooled values while fitting a handful:
    the frozen columns contribute a fixed per-player score and drop out of the
    gradient entirely, which is a genuinely smaller optimization problem
    rather than the same one with some coefficients discouraged.

    `weights`, when given, is one non-negative number per choice set and scales
    that set's contribution to both the value and the gradient. It is the
    M-step of `scoring.mixture`'s EM: a latent-class fit maximizes the EXPECTED
    complete-data log-likelihood, which is the ordinary log-likelihood with
    each pick counted by the responsibility its seat's posterior puts on the
    class being fitted. A fractional count cannot be expressed by duplicating
    rows, so it has to enter the objective here.

    The penalty is deliberately NOT scaled by the weights. `lam` is an absolute
    number of pseudo-observations' worth of pull toward `prior`, so a class
    holding a tenth of the corpus is shrunk ten times harder toward the pooled
    fit than one holding all of it -- which is the behavior the mixture wants,
    since it is the small class that has too few picks to estimate 29
    coefficients from.
    """
    value = 0.0
    grad = np.zeros_like(beta, dtype=float)
    for X, k, off, w in zip(X_list, chosen_list,
                            repeat(None) if offsets is None else offsets,
                            repeat(1.0) if weights is None else weights):
        scores = X @ beta
        if off is not None:
            scores = scores + off
        exp = _shifted_exp(scores)
        exp_sum = exp.sum()
        probs = exp / exp_sum
        log_sum_exp = scores.max() + np.log(exp_sum)
        value -= w * (scores[k] - log_sum_exp)
        grad += w * (probs @ X - X[k])
    if prior is not None and lam:
        diff = beta - prior
        value += lam * float(diff @ diff)
        grad += 2.0 * lam * diff
    return value, grad


def fit(X_list, chosen_list, prior=None, lam: float = 0.0,
        offsets=None, weights=None, start=None) -> np.ndarray:
    """The penalized conditional logit fit. Convex, so `start` cannot change
    the answer -- only how long L-BFGS-B takes to reach it.

    `start` exists to separate WHERE THE SEARCH BEGINS from WHAT THE PENALTY
    PULLS TOWARD, which `prior` had been doing both of. `scoring.mixture`'s EM
    refits every class on every iteration, always shrinking toward the same
    pooled vector but always resuming from that class's previous coefficients;
    restarting each of those fits from `prior` costs a full cold solve per
    class per iteration (~3s here) instead of a few L-BFGS steps (~0.1s), which
    is the difference between a K sweep that runs in minutes and one that runs
    overnight. Defaulting to `prior` (then zeros) keeps every existing caller
    on exactly the path it was on.
    """
    n_features = X_list[0].shape[1] if X_list else len(FEATURE_NAMES)
    if start is not None:
        start = np.asarray(start, dtype=float).copy()
    elif prior is None:
        start = np.zeros(n_features)
    else:
        start = np.asarray(prior, dtype=float).copy()
    if not X_list:
        return start
    result = minimize(neg_log_likelihood, start,
                      args=(X_list, chosen_list, prior, lam, offsets, weights),
                      jac=True, method="L-BFGS-B")
    if not result.success:
        warnings.warn(
            f"draft_model.fit: L-BFGS-B did not converge ({result.message}); "
            f"returning its result anyway from {len(X_list)} choice sets",
            RuntimeWarning,
        )
    return result.x


def fit_subset(X_list, chosen_list, pooled, keep, lam: float = 0.0) -> np.ndarray:
    """Fit only the `keep` coefficients; hold every other one at pooled.

    Fifteen coefficients against ~87 picks is under six observations per
    parameter, which is not a fit so much as a memorization of one league's
    six drafts. Two or three might work where fifteen cannot, so this exists
    to ask: a personal model over a handful of columns that plausibly describe
    a drafting personality, with the rest of the manager's behavior left to
    the pooled fit.

    Returns a FULL-length beta -- pooled everywhere, personal on `keep` --
    rather than the short vector it optimizes. That is what lets the result go
    straight into `log_likelihood`, `_heldout_gain` and the simulator with no
    special case anywhere downstream: a reduced personal model is just a beta
    that happens to agree with pooled on most of its entries.
    """
    pooled = np.asarray(pooled, dtype=float)
    keep = list(keep)
    rest = [i for i in range(len(pooled)) if i not in set(keep)]
    Z_list = [X[:, keep] for X in X_list]
    # The frozen columns' contribution: fixed per player, so it shifts every
    # score in a choice set without depending on what is being fitted.
    offsets = [X[:, rest] @ pooled[rest] for X in X_list]
    gamma = fit(Z_list, chosen_list, prior=pooled[keep], lam=lam, offsets=offsets)
    beta = pooled.copy()
    beta[keep] = gamma
    return beta


def prepare(observations, settings):
    X_list = [feature_matrix(o, settings) for o in observations]
    chosen = [o.chosen for o in observations]
    managers = [o.manager for o in observations]
    seasons = [o.season for o in observations]
    return X_list, chosen, managers, seasons


# Ridge strengths `select_lambda` cross-validates over. The top of the grid
# matters as much as the bottom, and for a different reason: the penalty pulls
# a personal fit toward the pooled prior, so at a large enough lambda the two
# coincide, `_heldout_gain` goes to exactly zero, and "this manager has no
# transferable signal of their own" is expressible. A grid that stops short of
# that cannot say it -- the best it can do is report the least-bad lambda it
# was allowed, and the shortfall it prints is a fact about the grid rather than
# about the manager.
#
# This grid used to stop at 100. On this league's six seasons four of the eight
# managers selected that maximum at every fold, which is cross-validation
# saying "more shrinkage, please" into a wall; extending the grid moved all
# four sharply toward zero (e.g. -0.0254 -> -0.0007). None of them crossed into
# positive, so the truncation was hiding an artifact rather than a signal --
# but a reported penalty that shrinks 36x when you lengthen a list is not a
# measurement, and the fix is to let the search finish.
LAMBDA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
MIN_PICKS_FOR_PERSONAL = 20

# Decay rate for the ADP baseline in backtest(): the baseline orders the pool
# by `market_rank` -- the same reference the model reads -- and a softmax over
# -TEMPERATURE * (0-based position in that order) gives a real probability
# distribution over "who does the market think goes next", unlike a uniform
# distribution, which assigns the market's #1 player and its #200th the same
# probability and so would be beaten by nearly anything. TEMPERATURE=1.0 means
# each step down the market's board is ~e times less likely than the one
# before it: sharp enough to be a meaningful baseline (most snake-draft picks
# land within a few spots of the top of the board), not so sharp that it
# degenerates into "always predict the market's next player" and stops being a
# distribution worth comparing log-loss against.
ADP_BASELINE_TEMPERATURE = 1.0

# Plain-language templates for the coefficients worth surfacing. Keyed by
# feature name; each maps (deviation from pooled) -> a phrase.
_PHRASES = {
    "reach": ("reaches for players the market ranks later",
              "avoids reaches, drafts in market order"),
    "fall": ("chases players who slide", "ignores players who slide"),
    "pos_RB": ("leans RB", "fades RB"),
    "pos_WR": ("leans WR", "fades WR"),
    "pos_TE": ("leans TE", "fades TE"),
    "pos_K": ("takes kickers early", "leaves kickers late"),
    "pos_DST": ("takes defenses early", "leaves defenses late"),
    "qb_early": ("early-QB guy", "waits on QB"),
    "te_early": ("early-TE guy", "waits on TE"),
    "need": ("fills starting slots first", "ignores roster needs"),
    "run": ("chases positional runs", "fades positional runs"),
    # Task 4 columns. `describe` takes the top-3 |diff| features and drops
    # any name missing here without looking further down the list, so a
    # feature added to FEATURE_NAMES and not to this table can report a
    # manager with a real, strong deviation back as "drafts close to league
    # average". All 15 of FEATURE_NAMES are named here; keep it that way
    # when adding one. Sign matters: `age` is centred
    # (positive = older than the position average), `hype` is
    # prod_rank - market_rank (positive = market ranks him ahead of his own
    # production -- a leap of faith), so a positive coefficient on either
    # means "leans toward more of that", not "leans toward the word that
    # comes first alphabetically".
    "age": ("leans veteran", "leans youth"),
    "no_track_record": ("bets on unproven players", "avoids unproven players"),
    "hype": ("chases hype over production", "fades hype, trusts production"),
    "trend": ("targets players trending up", "sticks with steady production"),
    # Task 3 columns, named here on the same rule: `describe` drops a
    # top-3 feature it has no phrase for WITHOUT looking further down the
    # list, so an unnamed feature turns a real deviation into "drafts close
    # to league average". Every entry of FEATURE_NAMES is named here; keep
    # it that way.
    #
    # Signs, since three of these are easy to read backwards. `held_at_pos` is
    # how many of that position the team already holds, so positive means
    # "wants more of what he has". `first_at_pos` is 1 when he has NONE, so
    # positive means "goes to an empty position first".
    # `rounds_since_pos` is rounds since he last took that position, so
    # positive means the wait itself pulls him back to it.
    "held_at_pos": ("doubles down on positions he already has",
                  "spreads picks across positions"),
    "first_at_pos": ("goes to an empty position first",
                     "keeps stacking positions he has started"),
    "rounds_since_pos": ("comes back to positions he has left alone",
                "drafts a position in bursts"),
    "first_at_pos_round": ("leaves empty positions until late",
                           "fills every position early"),
    "usage": ("targets high-volume players", "takes low-volume specialists"),
    "efficiency": ("targets efficient scorers", "takes volume over efficiency"),
    "played_share": ("targets players who stay on the field",
                     "tolerates injury risk"),
    # peak_gap is best career PPG minus last season's PPG, so positive means
    # the player has fallen furthest from his own peak.
    "peak_gap": ("bets on bounce-backs from a career peak",
                 "takes players at their current level"),
    # The `_POOL_SIGNAL_FEATURES` columns, on the same rule again: `describe`
    # drops a top-3 feature it has no phrase for without looking further down
    # the list, so an unnamed one turns a real deviation into "drafts close to
    # league average". Every entry of FEATURE_NAMES is named in this table.
    #
    # Signs. `dropoff_at_pos` is how far this candidate is above the next man
    # at his position, so positive means the cliff itself pulls him -- he
    # takes the last good one rather than the best one left. `durability` is
    # the board's percentile centred on the median, so positive is "prefers
    # the ones who stay on the field". `proj_change` is projection minus
    # recent actual, so positive means backing the forecast over the form.
    # `slots_left_at_pos` is how many starting slots at his position are still
    # open, so positive is a stronger form of `need`.
    "dropoff_at_pos": ("takes the last player before a positional cliff",
                       "ignores positional cliffs"),
    "vor": ("drafts on value over replacement", "ignores value over "
            "replacement"),
    "durability": ("targets players who stay healthy", "ignores injury "
                   "history"),
    "proj_change": ("backs the projection over recent form",
                    "trusts recent form over the projection"),
    "last_of_tier": ("takes the last player in a tier",
                     "lets a tier run out"),
    "slots_left_at_pos": ("fills the positions he has most slots open at",
                          "drafts regardless of how many slots are open"),
}


def _fit_maybe_subset(X_list, chosen_list, prior, lam, keep):
    """`fit` against the full feature set, or `fit_subset` against `keep`.

    One place, so every leave-one-season-out loop below reads the same whether
    it is measuring a 15-feature personal fit or a 2-feature one -- and so a
    reduced model is measured by exactly the machinery that judged the full
    one, which is the only way the two numbers are comparable.
    """
    if keep is None:
        return fit(X_list, chosen_list, prior=prior, lam=lam)
    return fit_subset(X_list, chosen_list, prior, keep, lam=lam)


def select_lambda(X_list, chosen_list, seasons, prior, grid=None,
                  keep=None) -> float:
    """Leave-one-season-out cross-validation over the ridge strength.

    Seasons, not random folds: picks inside one draft are not independent of
    each other, so a random split would leak the same draft across train and
    test and pick a lambda that is too loose.

    `keep` restricts the fit to that subset of feature indices (see
    `fit_subset`); `prior` is the full-length pooled vector either way.
    """
    grid = grid or LAMBDA_GRID
    unique = sorted(set(seasons))
    if len(unique) < 2:
        return grid[-1]                      # one season: shrink hard
    best, best_ll = grid[-1], -np.inf
    for lam in grid:
        total = 0.0
        for holdout in unique:
            train = [i for i, s in enumerate(seasons) if s != holdout]
            test = [i for i, s in enumerate(seasons) if s == holdout]
            if not train or not test:
                continue
            beta = _fit_maybe_subset([X_list[i] for i in train],
                                     [chosen_list[i] for i in train],
                                     prior, lam, keep)
            total += log_likelihood(beta, [X_list[i] for i in test],
                                    [chosen_list[i] for i in test])
        if total > best_ll:
            best, best_ll = lam, total
    return best


# The market-following prior for a league with no draft history of its own.
#
# A brand-new user's league has zero picks on record, so `fit_all` has nothing
# to fit and used to return {} -- no opponents, no simulation, the tool simply
# did not work for anyone but the one league baked into the database. This is
# the coefficient vector to fall back on instead. Every mock draft is this
# case too, permanently: eight strangers, no history, so this vector is not a
# fallback there but the whole model.
#
# THE NUMBERS THEMSELVES LIVE IN `scoring/mock_prior.py`, WHICH IS GENERATED.
# What stays here is what the vector is FOR, which does not change when the
# numbers do; the provenance of a particular set of numbers moves with them.
# `pipeline/fit_prior.py` (`make fit-prior`) rewrites that module, and only
# when a refit beats the incumbent on held-out top-1 -- the same rule stated
# above `_NEW_FEATURES`, enforced there by a person reading a table and here
# by a program that will not write the file otherwise. Editing a coefficient
# in that file by hand launders a guess as a measurement; refit instead.
#
# As a league accrues its own history, `fit_all` shrinks each manager off this
# prior via `select_lambda`, so the market default is the anchor that a real
# fit moves away from rather than a value that has to be unlearned.
COLD_START_PRIOR = mock_prior.PRIOR

# Two assertions, because they catch two different failures and the cheap one
# cannot see the expensive one. Length catches a feature added to
# FEATURE_NAMES without regenerating the prior. Order catches the silent
# version of the same thing: a vector fitted with the columns in one order
# and applied with them in another still has the right length and is wrong on
# every pick. `mock_prior.FEATURES` is written by the generator from
# FEATURE_NAMES, so agreeing is the normal case and disagreeing means the
# generated file is stale.
assert len(COLD_START_PRIOR) == len(FEATURE_NAMES), (
    "COLD_START_PRIOR must have one weight per feature; a feature was added to "
    "FEATURE_NAMES without extending the prior -- run `make fit-prior`")
assert list(mock_prior.FEATURES) == FEATURE_NAMES, (
    "scoring/mock_prior.py was fitted against a different feature order than "
    "FEATURE_NAMES holds now; regenerate it with `make fit-prior`")


def cold_start_fits() -> dict:
    """The fits to use when a league has no history to fit from.

    One entry, `__pooled__`, the market-following prior. There are no
    per-manager fits because there is nothing to distinguish the managers by
    yet -- every opponent drafts to the same market default until the league's
    own picks start telling them apart.
    """
    return {"__pooled__": COLD_START_PRIOR.copy()}


def fit_all(conn, settings=None, on_manager=None) -> dict:
    """Every manager's pick model, plus the pooled one they shrink toward.

    `on_manager(done, total, seasons)`, if given, is called once with done=0
    as soon as the manager set is known and again after each manager is
    fitted -- `seasons` being the sorted list of seasons the history spans.
    It exists for one caller: the connect screen, which otherwise shows a
    spinner for the 27.5-30.7s this function takes against the owner's own
    league (measured on data/nfl.duckdb: build_observations 9.9s, the pooled
    fit 3.3s, then eight per-manager fits at 1.5-3.1s each). That is 79-88%
    of a connect, and the per-manager loop is the only place in the whole
    connect where a REAL fraction exists to report -- everything else is one
    indivisible step, and a percentage invented for it would be a lie.

    Called before the pooled fit rather than after, so the first tick lands
    as soon as the manager set is known (about 10s in, once
    build_observations and prepare are done) instead of after the pooled fit
    adds another three seconds of silence. Never called at all on the
    cold-start path, which is how the caller tells "no history" apart from
    "no managers fitted yet".
    """
    settings = settings or league_mod.load(conn)
    observations = build_observations(conn)
    if not observations:
        # No history: fall back to the market prior rather than returning
        # nothing, which left the simulator with no opponents at all.
        return cold_start_fits()
    X_list, chosen, managers, seasons = prepare(observations, settings)
    names = sorted(set(managers))
    span = sorted(set(seasons))
    if on_manager is not None:
        on_manager(0, len(names), span)
    pooled = fit(X_list, chosen)
    fits = {"__pooled__": pooled}
    for done, manager in enumerate(names, start=1):
        idx = [i for i, m in enumerate(managers) if m == manager]
        Xm = [X_list[i] for i in idx]
        cm = [chosen[i] for i in idx]
        sm = [seasons[i] for i in idx]
        lam = select_lambda(Xm, cm, sm, prior=pooled)
        fits[manager] = fit(Xm, cm, prior=pooled, lam=lam)
        if on_manager is not None:
            on_manager(done, len(names), span)
    return fits


class PooledFits:
    """The league-average fit, refitted for every set of seasons it is not
    allowed to have seen.

    A baseline that has seen the picks it is scored on is not a baseline.
    `_heldout_gain` compares a personal fit trained on five seasons against
    pooled; when pooled was the all-seasons fit, it had already read the
    holdout draft. That is not a rounding error -- on this league the
    all-seasons pooled beats its own leave-that-season-out version by 0.028 to
    0.079 nats per pick, mean 0.043, which is larger than seven of the eight
    personal gains it was being used to judge. Every manager was being charged
    for a head start handed to their opponent.

    `backtest` has always refitted pooled inside each fold; this is the same
    discipline for the per-manager numbers, which are the ones on the cards.

    Two depths are needed, which is why this is a cache rather than a dict.
    The outer fold wants pooled without its holdout season. The inner
    selection inside `_nested_subset_gain` then holds out a second season, and
    its "stay pooled" candidate has to be scored without THAT season too --
    otherwise pooled is judged in-sample while every subset is judged out of
    it, and the procedure declines more often than the evidence warrants.
    Six single exclusions plus thirty pairs, fitted once and reused across
    every manager and candidate.

    Fitted over the whole league, all managers, because "the league-average
    model that has not seen this draft" is a statement about the league.
    """

    def __init__(self, X_list, chosen_list, seasons):
        self._X = X_list
        self._chosen = chosen_list
        self._seasons = seasons
        self._cache = {}

    def excluding(self, *seasons) -> np.ndarray:
        """Pooled coefficients fitted on every pick EXCEPT those seasons."""
        key = frozenset(seasons)
        if key not in self._cache:
            idx = [i for i, s in enumerate(self._seasons) if s not in key]
            self._cache[key] = fit([self._X[i] for i in idx],
                                   [self._chosen[i] for i in idx])
        return self._cache[key]

    @property
    def all_seasons(self) -> np.ndarray:
        """The fit that ships: every season, no exclusions."""
        return self.excluding()


def _fold_pooled(pooled, pooled_fits, *exclude):
    """Pooled coefficients for one fold, blind to `exclude`.

    Falling back to the caller's `pooled` when no `PooledFits` is given keeps
    synthetic and single-fold callers working, but it is the leaky comparison
    `PooledFits` documents -- both production callers (`write_profiles`,
    `reduced_model_report`) pass one.
    """
    if pooled_fits is None:
        return pooled
    return pooled_fits.excluding(*exclude)


def _heldout_gain(X_list, chosen, seasons, pooled, keep=None,
                  pooled_fits=None) -> float:
    """Per-pick log-likelihood advantage of a personal fit over pooled.

    Positive means the manager's own coefficients predict held-out picks
    better than the league-wide ones. Negative means they do not, and the
    simulator should use pooled for that manager.

    `keep` measures a reduced personal model -- those feature indices fitted
    personally, everything else held at pooled (see `fit_subset`). Same folds,
    same lambda search, same yardstick as the full fit, which is the point:
    "would two coefficients have worked where fifteen didn't" is only a real
    question if both are asked the same way.

    `pooled_fits` supplies a baseline that has not seen the season being
    scored, and is used as the fold's ridge prior as well so both sides of the
    comparison know the same amount. See `PooledFits` for what omitting it
    costs.
    """
    unique = sorted(set(seasons))
    if len(unique) < 2:
        return -np.inf
    personal_ll = pooled_ll = 0.0
    n = 0
    for holdout in unique:
        train = [i for i, s in enumerate(seasons) if s != holdout]
        test = [i for i, s in enumerate(seasons) if s == holdout]
        if not train or not test:
            continue
        fold_pooled = _fold_pooled(pooled, pooled_fits, holdout)
        Xtr = [X_list[i] for i in train]
        ctr = [chosen[i] for i in train]
        lam = select_lambda(Xtr, ctr, [seasons[i] for i in train],
                            prior=fold_pooled, keep=keep)
        beta = _fit_maybe_subset(Xtr, ctr, fold_pooled, lam, keep)
        Xt = [X_list[i] for i in test]
        ct = [chosen[i] for i in test]
        personal_ll += log_likelihood(beta, Xt, ct)
        pooled_ll += log_likelihood(fold_pooled, Xt, ct)
        n += len(test)
    return (personal_ll - pooled_ll) / n if n else -np.inf


# What the summary line says for a manager whose own coefficients don't beat
# the pooled ones on held-out seasons. It used to read "league average, not
# enough signal", which is true of the FORECAST and false of the manager: six
# real drafts is plenty of signal about what someone does, it just isn't
# enough to fit fifteen coefficients that transfer to a season they haven't
# drafted yet. The card carries measured tendencies right above this line --
# what they open with, how far ahead of the board they take players, when they
# get to a QB -- so the honest sentence points at those rather than declaring
# the manager unknowable.
POOLED_SUMMARY = ("forecast uses league-average coefficients; "
                  "the measured history is this manager's own")


def describe(beta, pooled, top: int = 3) -> str:
    diff = np.asarray(beta) - np.asarray(pooled)
    order = np.argsort(-np.abs(diff))
    phrases = []
    for i in order[:top]:
        name = FEATURE_NAMES[i]
        if name not in _PHRASES or abs(diff[i]) < 0.05:
            continue
        high, low = _PHRASES[name]
        phrases.append(high if diff[i] > 0 else low)
    return ", ".join(phrases) if phrases else "drafts close to league average"


def write_profiles(conn, settings=None) -> pd.DataFrame:
    settings = settings or league_mod.load(conn)
    observations = build_observations(conn)
    if not observations:
        empty = pd.DataFrame(columns=[
            "manager", "feature", "value", "pooled_value", "n_picks",
            "heldout_gain", "uses_personal", "summary"])
        write_table(conn, "manager_profiles", empty)
        return empty

    X_list, chosen, managers, seasons = prepare(observations, settings)
    # `pooled` (all seasons) is what ships as the fallback model; the
    # leave-one-season-out fits inside `pooled_fits` are only ever used to
    # SCORE, so the coefficients written below are unchanged by this.
    pooled_fits = PooledFits(X_list, chosen, seasons)
    pooled = pooled_fits.all_seasons
    rows = []
    for manager in sorted(set(managers)):
        idx = [i for i, m in enumerate(managers) if m == manager]
        Xm, cm = [X_list[i] for i in idx], [chosen[i] for i in idx]
        sm = [seasons[i] for i in idx]
        lam = select_lambda(Xm, cm, sm, prior=pooled)
        beta = fit(Xm, cm, prior=pooled, lam=lam)
        gain = _heldout_gain(Xm, cm, sm, pooled, pooled_fits=pooled_fits)
        uses_personal = bool(len(idx) >= MIN_PICKS_FOR_PERSONAL and gain > 0)
        effective = beta if uses_personal else pooled
        summary = describe(effective, pooled) if uses_personal else POOLED_SUMMARY
        for i, name in enumerate(FEATURE_NAMES):
            rows.append({"manager": manager, "feature": name,
                         "value": float(beta[i]), "pooled_value": float(pooled[i]),
                         "n_picks": len(idx),
                         "heldout_gain": float(gain) if np.isfinite(gain) else None,
                         "uses_personal": uses_personal, "summary": summary})
    profiles = pd.DataFrame(rows)
    write_table(conn, "manager_profiles", profiles)
    return profiles


_ROUND_BUCKET_ORDER = ("early", "mid", "late")


def backtest(conn, settings=None, features=None, observations=None) -> dict:
    """Leave-one-season-out validation across every season of history.

    Holding out only the newest season (the old behavior) answered one
    question with a sample six times smaller than what six seasons of
    history can support -- 112 evaluations instead of ~712 -- which is the
    difference between a number that moves on noise and one worth acting
    on. Rotating every season through as the holdout, accumulating hits and
    log-loss across all of them, is what makes the result trustworthy.

    `features`, when given, restricts both the fit and the score to that
    subset of `FEATURE_NAMES`: the design matrix handed to `fit` is sliced
    down to those columns before any fitting happens, so a dropped feature
    has no coefficient at all and genuinely cannot influence a prediction --
    not just omitted from the report. This is what makes `ablation` below a
    real measurement.

    `observations`, when given, is used as-is instead of calling
    `build_observations(conn)`. `ablation` calls this once per feature; on
    six seasons of history, deriving the same observations fresh for each
    call would replay the whole draft history five extra times for no
    reason.
    """
    settings = settings or league_mod.load(conn)
    if observations is None:
        observations = build_observations(conn)
    if not observations:
        return {"seasons": [], "top1": 0.0, "top5": 0.0,
                "logloss": float("inf"), "adp_top1": 0.0,
                "adp_logloss": float("inf"), "beats_adp": False, "by_round": []}

    X_list, chosen, managers, seasons = prepare(observations, settings)
    if features is not None:
        keep = [i for i, name in enumerate(FEATURE_NAMES) if name in features]
        X_list = [X[:, keep] for X in X_list]

    unique_seasons = sorted(set(seasons))
    teams = max(settings.teams, 1)

    hits1 = hits5 = adp_hits1 = n_total = 0
    ll = adp_ll = 0.0
    round_stats = {b: [0, 0, 0] for b in _ROUND_BUCKET_ORDER}   # top1, top5, n

    for holdout in unique_seasons:
        train = [i for i, s in enumerate(seasons) if s != holdout]
        test = [i for i, s in enumerate(seasons) if s == holdout]
        if not train or not test:
            # Fewer than two distinct seasons: this fold has no other side to
            # train or test against, so it is skipped rather than falling
            # back to the old single-holdout behavior of scoring a
            # zero-initialized, effectively-uniform model against it. If
            # every fold is skipped this way (genuinely one season total),
            # n_total stays 0 and the guard below reports top1/top5 = 0.0,
            # logloss = inf -- not that old uniform-prediction score.
            continue
        pooled = fit([X_list[i] for i in train], [chosen[i] for i in train])

        fits = {}
        for manager in set(managers):
            idx = [i for i in train if managers[i] == manager]
            if len(idx) < MIN_PICKS_FOR_PERSONAL:
                fits[manager] = pooled
                continue
            lam = select_lambda([X_list[i] for i in idx], [chosen[i] for i in idx],
                                [seasons[i] for i in idx], prior=pooled)
            fits[manager] = fit([X_list[i] for i in idx], [chosen[i] for i in idx],
                                prior=pooled, lam=lam)

        for i in test:
            X, k = X_list[i], chosen[i]
            probs = _softmax(X @ fits.get(managers[i], pooled))
            order = np.argsort(-probs)
            hit1 = int(order[0] == k)
            hit5 = int(k in order[:5])
            hits1 += hit1
            hits5 += hit5
            ll += np.log(max(probs[k], 1e-12))
            # ADP baseline, scored on `market_rank` -- the same board the
            # model reads. NOT pool position: build_observations sorts the
            # pool by the FFC `adp_rank`, while `market_rank` has been the
            # ESPN cheat-sheet ordering since _enrich_pool switched
            # references, and on this league's history the two disagree
            # about who is top-of-pool in 550 of 696 observations. Scoring
            # the baseline on a board the model never sees would confound a
            # better model with a worse yardstick; same reference both
            # sides. A uniform distribution over the pool is not a real
            # baseline either -- it would assign the market's #1 player and
            # its #200th the same probability, so "beats the market" would
            # be true almost by construction. Still immune to `features`:
            # masking drops columns from X, never rows, and this reads the
            # pool rather than X at all.
            mr = observations[i].pool["market_rank"].to_numpy(dtype=float)
            rankpos = np.argsort(np.argsort(mr))   # 0-based place on that board
            adp_hits1 += int(rankpos[k] == 0)
            adp_probs = _softmax(-ADP_BASELINE_TEMPERATURE * rankpos)
            adp_ll += np.log(max(adp_probs[k], 1e-12))
            n_total += 1

            stats = round_stats[_round_bucket(observations[i].overall_pick, teams)]
            stats[0] += hit1
            stats[1] += hit5
            stats[2] += 1

    if n_total == 0:
        # Every season was a degenerate fold (no train or no test partner --
        # only reachable with fewer than two distinct seasons). Reporting
        # 0.0/0.0 for hits1/n_total would read as a perfect log-loss of 0.0,
        # which is a worse lie than admitting nothing was evaluated.
        return {"seasons": unique_seasons, "top1": 0.0, "top5": 0.0,
                "logloss": float("inf"), "adp_top1": 0.0,
                "adp_logloss": float("inf"), "beats_adp": False, "by_round": []}

    by_round = [{"round_bucket": b, "top1": s[0] / s[2], "top5": s[1] / s[2], "n": s[2]}
                for b, s in round_stats.items() if s[2] > 0]
    report = {"seasons": unique_seasons, "top1": hits1 / n_total,
              "top5": hits5 / n_total, "logloss": -ll / n_total,
              "adp_top1": adp_hits1 / n_total, "adp_logloss": -adp_ll / n_total,
              "by_round": by_round}
    report["beats_adp"] = bool(report["logloss"] < report["adp_logloss"])
    return report


def ablation(conn, settings=None, candidates=None) -> pd.DataFrame:
    """Each new feature's contribution, measured rather than argued.

    At roughly 105 picks per manager a feature that does not pay for itself
    is worse than absent: it fits noise and drags every other coefficient
    with it. This table is what decides which of them ship.

    Builds the observations once and threads them into every `backtest`
    call below (see that function's docstring) rather than the six
    independent history replays a literal one-`backtest`-call-per-row
    reading would cost.

    `candidates` is which features to drop one at a time, defaulting to
    `_NEW_FEATURES`. The default is deliberately NOT `FEATURE_NAMES` and not
    `_NEW_FEATURES + UNMEASURED_FEATURES`: this runs inside every
    `make fit-managers`, each row costs a full leave-one-season-out backtest
    with a lambda search per manager, and tripling that on every fit to
    re-answer a question this league's 696 picks cannot settle is the wrong
    trade. `UNMEASURED_FEATURES` is measured where the evidence is:
    `pipeline/fit_prior.py` passes it to its own corpus-shaped equivalent of
    this function against 26 mock drafts, which is four times the sample.
    Passing it HERE measures those eight against this league's own history,
    which is a different and still-unanswered question.
    """
    candidates = list(_NEW_FEATURES if candidates is None else candidates)
    observations = build_observations(conn)
    full = backtest(conn, settings, observations=observations)
    rows = [{"dropped": "none", "top1": full["top1"], "top5": full["top5"],
             "delta_top1": 0.0, "delta_top5": 0.0}]
    for feature in candidates:
        keep = [f for f in FEATURE_NAMES if f != feature]
        cut = backtest(conn, settings, features=keep, observations=observations)
        # Both deltas, always. Reporting delta_top1 and leaving top5 as a raw
        # column invites reading the second one only when it agrees with the
        # first -- which is how `no_track_record` got defended on top-5 while
        # `age`, which top-5 argues against, kept its place unexamined.
        rows.append({"dropped": feature, "top1": cut["top1"],
                     "top5": cut["top5"],
                     "delta_top1": full["top1"] - cut["top1"],
                     "delta_top5": full["top5"] - cut["top5"]})
    return pd.DataFrame(rows)


# Candidate reduced personal models. Every column named here plausibly
# describes a drafting *personality* rather than a fact about the board: does
# this person stick to market order or jump it (`reach`, `fall`), what do they
# favour (the position dummies), do they chase positional runs (`run`).
# Everything not named stays at its pooled value.
#
# The list is short on purpose. Each extra candidate is another chance for one
# of them to look good on six seasons of noise, and `_nested_subset_gain`
# below exists precisely because "the best of N subsets, chosen by looking at
# the held-out seasons" is not a measurement.
PERSONAL_SUBSETS = {
    "reach": ("reach",),
    "reach+fall": ("reach", "fall"),
    "run": ("run",),
    "reach+run": ("reach", "run"),
    "pos_skill": ("pos_RB", "pos_WR", "pos_TE"),
    "pos": tuple(f"pos_{p}" for p in _POSITION_DUMMIES),
    "reach+pos_skill": ("reach", "pos_RB", "pos_WR", "pos_TE"),
}


def _subset_indices(names) -> tuple:
    return tuple(FEATURE_NAMES.index(name) for name in names)


def _nested_subset_gain(X_list, chosen, seasons, pooled,
                        pooled_fits=None) -> tuple:
    """Held-out gain of the whole selection *procedure*, not of one subset.

    Picking each manager's best subset by comparing held-out gains and then
    reporting that best gain is the oldest mistake in model selection: with
    seven candidates and eight managers, several will clear zero on noise
    alone. This instead chooses the subset inside each training fold -- an
    inner leave-one-season-out over the five training seasons only -- and then
    scores that choice on the season the choice never saw.

    "Stay pooled" is one of the candidates, so the procedure is allowed to
    decline, and a manager it always declines for scores exactly 0.0 rather
    than being forced into a personal fit it did not want. Returns
    (gain, [chosen candidate per fold]).

    `pooled_fits` is threaded through to both the inner selection and the
    outer score, so no baseline anywhere has seen the season it is judged on
    -- see `PooledFits`.
    """
    unique = sorted(set(seasons))
    if len(unique) < 2:
        return -np.inf, []
    personal_ll = pooled_ll = 0.0
    n = 0
    picked = []
    for holdout in unique:
        train = [i for i, s in enumerate(seasons) if s != holdout]
        test = [i for i, s in enumerate(seasons) if s == holdout]
        if not train or not test:
            continue
        fold_pooled = _fold_pooled(pooled, pooled_fits, holdout)
        inner_seasons = [seasons[i] for i in train]
        Xtr = [X_list[i] for i in train]
        ctr = [chosen[i] for i in train]
        # Inner selection: the pooled fit's own inner-holdout score is the bar
        # every subset has to clear, and it is scored out of sample like they
        # are -- `_inner_ll` drops the inner season from pooled as well as from
        # the subset fits.
        best_name, best_ll = None, _inner_ll(Xtr, ctr, inner_seasons,
                                             fold_pooled, None,
                                             pooled_fits, holdout)
        for name, features in PERSONAL_SUBSETS.items():
            score = _inner_ll(Xtr, ctr, inner_seasons, fold_pooled,
                              _subset_indices(features), pooled_fits, holdout)
            if score > best_ll:
                best_name, best_ll = name, score
        picked.append(best_name or "pooled")

        Xt = [X_list[i] for i in test]
        ct = [chosen[i] for i in test]
        if best_name is None:
            personal_ll += log_likelihood(fold_pooled, Xt, ct)
        else:
            keep = _subset_indices(PERSONAL_SUBSETS[best_name])
            lam = select_lambda(Xtr, ctr, inner_seasons, prior=fold_pooled,
                                keep=keep)
            beta = fit_subset(Xtr, ctr, fold_pooled, keep, lam=lam)
            personal_ll += log_likelihood(beta, Xt, ct)
        pooled_ll += log_likelihood(fold_pooled, Xt, ct)
        n += len(test)
    return ((personal_ll - pooled_ll) / n if n else -np.inf), picked


def _inner_ll(X_list, chosen, seasons, pooled, keep,
              pooled_fits=None, outer_holdout=None) -> float:
    """Total held-out log-likelihood of one candidate across an inner
    leave-one-season-out over `seasons`.

    `keep=None` scores pooled, which needs no fitting -- but it still needs the
    *right* pooled. Scored by a fit that saw the inner fold, the "stay pooled"
    candidate is judged in sample while every subset is judged out of it, and
    the selection declines far more often than the evidence warrants. That
    asymmetry is why this takes `outer_holdout`: the fit used here must be
    blind to both the season the outer fold is scoring and the season this
    inner fold is scoring.
    """
    total = 0.0
    unique = sorted(set(seasons))
    for holdout in unique:
        train = [i for i, s in enumerate(seasons) if s != holdout]
        test = [i for i, s in enumerate(seasons) if s == holdout]
        if not train or not test:
            continue
        exclude = [holdout] if outer_holdout is None else [outer_holdout, holdout]
        inner_pooled = _fold_pooled(pooled, pooled_fits, *exclude)
        Xt = [X_list[i] for i in test]
        ct = [chosen[i] for i in test]
        if keep is None:
            total += log_likelihood(inner_pooled, Xt, ct)
            continue
        Xtr = [X_list[i] for i in train]
        ctr = [chosen[i] for i in train]
        lam = select_lambda(Xtr, ctr, [seasons[i] for i in train],
                            prior=inner_pooled, keep=keep)
        total += log_likelihood(
            fit_subset(Xtr, ctr, inner_pooled, keep, lam=lam), Xt, ct)
    return total


def reduced_model_report(conn, settings=None) -> pd.DataFrame:
    """Does a SMALLER personal model generalize where the full one doesn't?

    One row per (manager, candidate): `heldout_gain` measured by exactly the
    machinery `write_profiles` uses, so the numbers sit on the same scale as
    the `full` row and as `manager_profiles.heldout_gain`. Two rows are not
    subsets:

    - `full`   -- all 15 features, what ships today.
    - `nested` -- the honest answer to "let each manager have whichever small
      model suits them", with the choice made inside each training fold and
      "stay pooled" among the options. This is the row to read. A per-subset
      row that clears zero and a `nested` row that doesn't means the subset
      was chosen with hindsight.

    `n_free` is how many coefficients that row fits personally: 15 on `full`,
    and -1 on `nested`, where the count is a different number every fold.

    Not wired into `make fit-managers`: it refits every candidate through a
    nested cross-validation and takes minutes, against a normal run's seconds.
    Run it when the history grows a season -- `make fit-managers REDUCED=1`.
    """
    settings = settings or league_mod.load(conn)
    observations = build_observations(conn)
    if not observations:
        return pd.DataFrame(columns=["manager", "subset", "n_free",
                                     "heldout_gain", "picked"])
    X_list, chosen, managers, seasons = prepare(observations, settings)
    pooled_fits = PooledFits(X_list, chosen, seasons)
    pooled = pooled_fits.all_seasons
    rows = []
    for manager in sorted(set(managers)):
        idx = [i for i, m in enumerate(managers) if m == manager]
        Xm, cm = [X_list[i] for i in idx], [chosen[i] for i in idx]
        sm = [seasons[i] for i in idx]
        rows.append({"manager": manager, "subset": "full",
                     "n_free": len(FEATURE_NAMES),
                     "heldout_gain": _heldout_gain(Xm, cm, sm, pooled,
                                                   pooled_fits=pooled_fits),
                     "picked": ""})
        for name, features in PERSONAL_SUBSETS.items():
            gain = _heldout_gain(Xm, cm, sm, pooled,
                                 keep=_subset_indices(features),
                                 pooled_fits=pooled_fits)
            rows.append({"manager": manager, "subset": name,
                         "n_free": len(features), "heldout_gain": gain,
                         "picked": ""})
        gain, picked = _nested_subset_gain(Xm, cm, sm, pooled,
                                           pooled_fits=pooled_fits)
        rows.append({"manager": manager, "subset": "nested", "n_free": -1,
                     "heldout_gain": gain, "picked": ",".join(picked)})
    return pd.DataFrame(rows)


def _round_bucket(overall_pick: int, teams: int) -> str:
    round_no = (overall_pick - 1) // max(teams, 1) + 1
    if round_no <= EARLY_ROUNDS:
        return "early"
    return "mid" if round_no <= 8 else "late"


def positional_bias(conn, settings=None) -> pd.DataFrame:
    """How far ahead of the market this league takes each position.

    `mean_gap` is `market_rank - overall_pick`, so positive means the league
    drafts that position earlier than the market ranks it (a player ranked
    2 taken at pick 1 scores +1); negative means the league lets it slide
    past where the market has it. Pooled across managers on purpose: a
    league-wide habit forced through eight per-manager coefficients would
    spend scarce data re-learning the same thing eight times.
    """
    settings = settings or league_mod.load(conn)
    observations = build_observations(conn)
    rows = []
    for obs in observations:
        chosen = obs.pool.iloc[obs.chosen]
        rows.append({"position": chosen["position"],
                     "round_bucket": _round_bucket(obs.overall_pick, settings.teams),
                     "gap": float(chosen["market_rank"]) - obs.overall_pick})
    if not rows:
        return pd.DataFrame(columns=["position", "round_bucket", "mean_gap", "n"])
    df = pd.DataFrame(rows)
    out = df.groupby(["position", "round_bucket"], as_index=False).agg(
        mean_gap=("gap", "mean"), n=("gap", "size"))
    return out


# Positions whose *first* pick is worth reporting on its own. When a manager
# takes their first QB, TE, K or DST is a decision with a round number
# attached, and knowing it changes who you can still afford to wait on. RB and
# WR are left out because everybody takes one early -- "first RB in round 1" is
# a fact about the format, not about the manager.
FIRST_AT_POSITIONS = ("QB", "TE", "K", "DST")

# A per-position reach number needs enough picks behind it to be worth
# printing; below this it is one or two drafts talking.
MIN_PICKS_FOR_POSITION_REACH = 4

# Positions kept out of every reach number. A kicker's market rank sits at
# 129-245 while every roster needs exactly one, so kickers go at picks 82-128
# and score +38 to +126 against the board *by construction*. That is a fact
# about roster slots, not about a manager, and left in it swamps the average:
# MaxMandia reads +2.89 overall and -0.39 without kickers, jtague99 +3.06 and
# -0.65. The card was asserting "takes players ahead of the board" about two
# people who do the opposite on every pick that was actually a choice, and the
# per-position row printed "K +49..+67" on all eight cards -- eight cards
# making one league-wide claim about eight different people.
#
# `positional_bias` deliberately keeps kickers, because it IS the league-wide
# claim and that +58 is the interesting part of it. This table is the
# per-manager one, and there the same number is noise with a sign.
REACH_EXCLUDED_POSITIONS = ("K", "DST")

# Long-format `metric` values `manager_tendencies` emits, in the order the card
# reads them. Exported so the API and its tests name them from one place.
TENDENCY_METRICS = ("first_pick", "reach_overall", "reach_bucket",
                    "reach_position", "first_at_position")


def manager_tendencies(conn, settings=None) -> pd.DataFrame:
    """What each manager has actually done, counted rather than fitted.

    Every number here is a descriptive statistic over real picks. None of it
    comes from the model, none of it depends on a coefficient generalizing,
    and all of it stays true whether or not a manager earns a personal fit --
    which, on this league's history, is exactly why it is worth computing:
    the pooled fallback says nothing specific about anybody, and this does.

    Long/tidy, one row per (manager, metric, key), because the metrics carry
    different shapes and a wide table would be mostly nulls:

    | metric            | key      | value                              | n                        |
    |-------------------|----------|------------------------------------|--------------------------|
    | first_pick        | position | drafts opened with that position   | drafts on record         |
    | reach_overall     | None     | mean `market_rank - overall_pick`  | picks behind the mean    |
    | reach_bucket      | bucket   | same, within early/mid/late        | picks behind the mean    |
    | reach_position    | position | same, within that position         | picks behind the mean    |
    | first_at_position | position | mean round of their first such pick| drafts it was averaged over |

    Sign on the reach metrics matches `positional_bias`: positive means taken
    earlier than the market ranked them (a player ranked 20 taken at pick 10
    scores +10), negative means let slide.

    Every reach metric excludes `REACH_EXCLUDED_POSITIONS` -- see that constant
    for why a kicker's +58 is a fact about roster slots rather than about a
    person, and what leaving them in did to two managers' headline number.

    Two different picks tables feed this on purpose. The position facts
    (`first_pick`, `first_at_position`) count `draft_picks` directly, so they
    see every pick the manager made. The reach facts read
    `build_observations`, which drops picks with no ADP row that season --
    unavoidably, since a reach is undefined without a market rank to reach
    past. `n` on each row is the count actually behind that number, so the
    two never silently claim the same denominator.
    """
    settings = settings or league_mod.load(conn)
    cols = ["manager", "metric", "key", "value", "n"]
    picks = read_table(conn, "draft_picks")
    teams = read_table(conn, "draft_teams")
    if picks.empty or teams.empty:
        return pd.DataFrame(columns=cols)
    merged = picks.merge(teams[["season", "team_id", "manager"]],
                         on=["season", "team_id"], how="inner")
    if merged.empty:
        return pd.DataFrame(columns=cols)
    drafts = teams.groupby("manager")["season"].nunique()

    rows = []
    # The first pick of each of their drafts. Sorting by overall_pick and
    # keeping the first row per (manager, season) survives a manager holding
    # two team_ids in one season -- the same guard /api/managers/history
    # applies to its round-1 list.
    firsts = (merged.sort_values("overall_pick")
              .drop_duplicates(["manager", "season"], keep="first")
              .dropna(subset=["position"]))
    for (manager, position), grp in firsts.groupby(["manager", "position"]):
        rows.append({"manager": manager, "metric": "first_pick",
                     "key": position, "value": float(len(grp)),
                     "n": int(drafts.get(manager, 0))})

    positioned = merged.dropna(subset=["position"])
    at_pos = positioned[positioned["position"].isin(FIRST_AT_POSITIONS)]
    firsts_at = (at_pos.sort_values("overall_pick")
                 .drop_duplicates(["manager", "season", "position"], keep="first"))
    for (manager, position), grp in firsts_at.groupby(["manager", "position"]):
        # `round` is a nullable Int64 from the ESPN import: .mean() on an
        # all-null group returns pd.NA, and float(pd.NA) raises -- which would
        # abort write_tendencies after write_profiles had already committed.
        # A partially-null group is worse than a crash: it averages the rounds
        # it has while `n` counts every season, rendering "K R15.0, 6 of 6
        # drafts" out of one real value. Count what was actually averaged.
        rounds = grp["round"].dropna()
        if rounds.empty:
            continue
        rows.append({"manager": manager, "metric": "first_at_position",
                     "key": position, "value": float(rounds.mean()),
                     "n": int(len(rounds))})

    gaps = pd.DataFrame([
        {"manager": o.manager,
         "position": o.pool.iloc[o.chosen]["position"],
         "bucket": _round_bucket(o.overall_pick, settings.teams),
         "gap": float(o.pool.iloc[o.chosen]["market_rank"]) - o.overall_pick}
        for o in build_observations(conn)])
    if not gaps.empty:
        gaps = gaps[~gaps["position"].isin(REACH_EXCLUDED_POSITIONS)]
    if not gaps.empty:
        for manager, grp in gaps.groupby("manager"):
            rows.append({"manager": manager, "metric": "reach_overall",
                         "key": None, "value": float(grp["gap"].mean()),
                         "n": int(len(grp))})
            for bucket in _ROUND_BUCKET_ORDER:
                bucket_grp = grp[grp["bucket"] == bucket]
                if bucket_grp.empty:
                    continue
                rows.append({"manager": manager, "metric": "reach_bucket",
                             "key": bucket, "value": float(bucket_grp["gap"].mean()),
                             "n": int(len(bucket_grp))})
            for position, pos_grp in grp.groupby("position"):
                if len(pos_grp) < MIN_PICKS_FOR_POSITION_REACH:
                    continue
                rows.append({"manager": manager, "metric": "reach_position",
                             "key": position, "value": float(pos_grp["gap"].mean()),
                             "n": int(len(pos_grp))})

    out = pd.DataFrame(rows, columns=cols)
    # `key` is null on the reach_overall rows; keep the column object-typed so
    # DuckDB stores it as VARCHAR rather than inferring a type from a frame
    # that happens to contain only nulls.
    out["key"] = out["key"].astype(object)
    return out


def write_tendencies(conn, settings=None) -> pd.DataFrame:
    """Persist `manager_tendencies` as the table of the same name.

    Computed at fit time, not per request: it replays every season's draft
    through `build_observations` to get a market rank for each pick, which is
    far too much work to repeat on every page load of the forecast tab.
    """
    tendencies = manager_tendencies(conn, settings)
    write_table(conn, "manager_tendencies", tendencies)
    return tendencies


def write_backtest(conn, settings=None) -> dict:
    """Run the backtest and persist it as a one-row `model_backtest` table.

    Spec Part 3: "If the model does not beat ADP-only, that is the finding,
    and the board should not present simulator output as authoritative."
    Printing it to stdout from `make fit-managers` cannot reach the board, so
    nothing stopped the board presenting the simulator as authoritative
    regardless. `/api/model` serves this row and the rail warns on it.

    `seasons` is persisted as a JSON-encoded string (`json.dumps`), not as
    DuckDB's native LIST column. Every other field in this row is a scalar
    that `/api/model` reads with a plain `row.get(...)` and a small
    `_..._or_none` conversion; a JSON string round-trips through that same
    shape (decode with `json.loads` on the way out) without `/api/model`
    needing a separate code path for one column's storage type. The
    returned `report` itself is untouched -- `seasons` stays a real list for
    `fit_managers`, which prints it directly.
    """
    report = backtest(conn, settings)
    row = dict(report)
    row["seasons"] = json.dumps(report["seasons"])
    write_table(conn, "model_backtest", pd.DataFrame([row]))
    return report
