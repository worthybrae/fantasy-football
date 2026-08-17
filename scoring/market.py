"""Blend FFC / ESPN / FantasyPros into a market consensus rank per board player.

NOTE: scoring.board imports add_market (Task 3), so importing board at module
level here would be circular — _norm_name is imported inside the helpers.
"""
import numpy as np
import pandas as pd

# Consensus inputs. Every component is PPR-native: FFC's ppr ADP feed,
# ESPN's explicit PPR expert rank (NOT their mixed-format ADP), FP's ppr
# ECR, MFL's IS_PPR-filtered ADP, CBS's ppr top-200. espn_ppr_rank is also
# a standalone board column, so it is not dropped with the other inputs.
_RANK_COLS = ["ffc_rank", "espn_ppr_rank", "fp_rank", "mfl_rank", "cbs_rank"]
# `ffc_rank` is NOT dropped with the other consensus inputs: it is the single
# source `draft_sim.build_pool` ranks the simulator's pool on, because it is
# the only one with both a consistent six-season history (`historic_adp`,
# what `draft_model._enrich_pool` fits `reach`/`fall` against) and a
# current-season feed. The fit and the simulator have to read the same
# source or a coefficient learned on one means something else applied to the
# other. It stays a consensus input as well -- the two uses are independent.
_DROP_COLS = ["espn_rank", "fp_rank", "mfl_rank", "cbs_rank"]
# How many players ESPN must rank before "ESPN doesn't rank him" is allowed
# to mean "undraftable". This is a presence test on the feed: live it ranks
# ~223, a failed fetch or an unjoined crosswalk ranks none, and a fixture
# ranks one or two.
#
# Deliberately NOT a coverage *share*. That was tried and was self-defeating:
# the board carries 676 non-DST players against ESPN's 223, a share of 0.33,
# so a >= 0.5 gate switched the filter off on exactly the real board it was
# written for -- and the wider the gap, the more there is to drop, the less
# likely it would ever fire. The gap is the thing being removed, not evidence
# the feed is broken.
ESPN_MIN_RANKED = 100

def _norm(name):
    from scoring.board import _norm_name  # deferred: avoids circular import
    return _norm_name(name)

def select_format(df, fmt="ppr"):
    """A source table's rows for the league's scoring format.

    The contract with the ADP importer (which owns pipeline/, populating this
    exact shape):
      * A source table gains a text `format` column with values in
        {'ppr','half','std'} -- one row-set per format the source publishes.
      * A source may not carry the requested format (mfl_adp has no 'half'
        feed, for instance). When the requested format has NO rows, fall back
        to that source's 'ppr' rows rather than dropping the source from the
        consensus.
      * BACKWARD COMPATIBILITY: an old database -- and many existing test
        fixtures -- seed these tables with NO `format` column at all. With
        the column absent, every row is PPR: the frame is returned untouched,
        so a PPR league (or an unknown one) sees exactly today's rows.

    The `format` column is dropped on the way out, so a table filtered to
    'ppr' is byte-identical to one that never carried the column -- which is
    what keeps the PPR path unchanged. ESPN is never passed through here:
    espn_adp is PPR-only by contract and stays PPR everywhere.
    """
    if df is None or df.empty or "format" not in df.columns:
        return df
    rows = df[df["format"] == fmt]
    if rows.empty:  # source lacks this format -> fall back to its PPR rows
        rows = df[df["format"] == "ppr"]
    return rows.drop(columns=["format"]).reset_index(drop=True)

def _espn_ranks(board, espn, sleeper):
    """ESPN's ADP rank, PPR rank and player id, per board row.

    `espn_id` rides out on exactly the paths that already resolve an ESPN row
    to a board row -- the gsis crosswalk first, the name fallback second. It
    is not a third matching strategy, it is the same two, carrying one more
    column. That matters because it makes the id available on the board
    itself, so a live draft pick (which arrives as an ESPN player id and
    nothing else) becomes an exact dictionary lookup instead of a name match
    performed under a 30-second clock.
    """
    out = pd.Series(np.nan, index=board.index)
    ppr = pd.Series(np.nan, index=board.index)
    ids = pd.Series(np.nan, index=board.index)
    if espn is None or espn.empty:
        return out, ppr, ids
    e = espn.dropna(subset=["espn_adp"]).copy()
    e["espn_rank"] = e["espn_adp"].rank(method="first")
    # Dedupe by espn_id, keeping lowest rank (best)
    e = e.sort_values("espn_rank").drop_duplicates("espn_id", keep="first")
    if sleeper is not None and not sleeper.empty:
        xwalk = sleeper[["gsis_id", "espn_id"]].drop_duplicates("espn_id")
        e = e.merge(xwalk, on="espn_id", how="left")
    else:
        e["gsis_id"] = None
    # The crosswalk itself can carry a junk duplicate: two different espn_ids
    # mapped to the same gsis_id (seen in the live sleeper_ids table). Left
    # unhandled that makes by_id's index non-unique, and board["player_id"]
    # .map(by_id) below raises InvalidIndexError. Dedupe the non-null-gsis
    # rows by gsis_id, keeping the best (lowest) rank; rows with no gsis_id
    # at all must survive untouched -- they still feed the name-fallback
    # path further down.
    has_id = e[e["gsis_id"].notna()].sort_values("espn_rank").drop_duplicates("gsis_id", keep="first")
    no_id = e[e["gsis_id"].isna()]
    e = pd.concat([has_id, no_id], ignore_index=True)
    by_id = has_id.set_index("gsis_id")["espn_rank"]
    mapped = board["player_id"].map(by_id)
    by_id_ppr = has_id.set_index("gsis_id")["espn_ppr_rank"]
    mapped_ppr = board["player_id"].map(by_id_ppr)
    by_id_espn = has_id.set_index("gsis_id")["espn_id"]
    mapped_ids = board["player_id"].map(by_id_espn)
    # name+position fallback for espn rows without a crosswalk hit (never DST)
    rest = e[e["gsis_id"].isna() & (e["position"] != "DST")].copy()
    if not rest.empty:
        rest["norm"] = rest["espn_name"].map(_norm)
        # Dedupe by (norm, position), keeping lowest rank (best)
        rest = rest.sort_values("espn_rank").drop_duplicates(["norm", "position"], keep="first")
        by_name = rest.set_index(["norm", "position"])["espn_rank"]
        by_name_ppr = rest.set_index(["norm", "position"])["espn_ppr_rank"]
        by_name_espn = rest.set_index(["norm", "position"])["espn_id"]
        key = pd.MultiIndex.from_arrays([board["name"].map(_norm), board["position"]])
        fallback = pd.Series(by_name.reindex(key).to_numpy(), index=board.index)
        fallback_ppr = pd.Series(by_name_ppr.reindex(key).to_numpy(), index=board.index)
        fallback_ids = pd.Series(by_name_espn.reindex(key).to_numpy(), index=board.index)
        mapped = mapped.fillna(fallback)
        mapped_ppr = mapped_ppr.fillna(fallback_ppr)
        mapped_ids = mapped_ids.fillna(fallback_ids)
    return mapped, mapped_ppr, mapped_ids

def _name_ranks(board, df, name_col, rank_col):
    """Generic (name, position) rank joiner for sources without an id
    crosswalk (MFL, CBS). Same dedupe rule as the FP path: keep the best
    (lowest) rank per player. No DST handling -- both feeds are players-only.

    Two passes: exact norm match, then a space-squashed match for names
    whose hyphens the source flattened differently (CBS slugs turn
    "Amon-Ra" into "amon ra"; the board norm makes "amonra")."""
    out = pd.Series(np.nan, index=board.index)
    if df is None or df.empty:
        return out
    f = df.dropna(subset=[rank_col]).copy()
    f["norm"] = f[name_col].map(_norm)
    f = f.sort_values(rank_col).drop_duplicates(["norm", "position"], keep="first")
    board_norm = board["name"].map(_norm)

    by_name = f.set_index(["norm", "position"])[rank_col]
    key = pd.MultiIndex.from_arrays([board_norm, board["position"]])
    mapped = pd.Series(by_name.reindex(key).to_numpy(), index=board.index, dtype=float)

    squash = lambda s: s.str.replace(" ", "", regex=False)
    f["squashed"] = squash(f["norm"])
    f = f.sort_values(rank_col).drop_duplicates(["squashed", "position"], keep="first")
    by_squashed = f.set_index(["squashed", "position"])[rank_col]
    key2 = pd.MultiIndex.from_arrays([squash(board_norm), board["position"]])
    fallback = pd.Series(by_squashed.reindex(key2).to_numpy(), index=board.index, dtype=float)
    return mapped.fillna(fallback)


def _fp_ranks(board, fp):
    ranks = pd.Series(np.nan, index=board.index)
    tiers = pd.Series(np.nan, index=board.index)
    if fp is None or fp.empty:
        return ranks, tiers
    f = fp.dropna(subset=["rank_ecr"]).copy()
    players = f[f["position"] != "DST"].copy()
    players["norm"] = players["fp_name"].map(_norm)
    # Dedupe by (norm, position), keeping lowest rank (best)
    players = players.sort_values("rank_ecr").drop_duplicates(["norm", "position"], keep="first")
    by_name = players.set_index(["norm", "position"])
    key = pd.MultiIndex.from_arrays([board["name"].map(_norm), board["position"]])
    ranks = pd.Series(by_name["rank_ecr"].reindex(key).to_numpy(), index=board.index, dtype=float)
    tiers = pd.Series(by_name["fp_tier"].reindex(key).to_numpy(), index=board.index, dtype=float)
    dst = f[f["position"] == "DST"].sort_values("rank_ecr").drop_duplicates("team", keep="first").set_index("team")
    is_dst = board["position"] == "DST"
    ranks.loc[is_dst] = board.loc[is_dst, "team"].map(dst["rank_ecr"]).astype(float)
    tiers.loc[is_dst] = board.loc[is_dst, "team"].map(dst["fp_tier"]).astype(float)
    return ranks, tiers

def add_market(board, espn, fp, sleeper, mfl=None, cbs=None, fmt="ppr"):
    out = board.copy()
    # Pick each opinion source's rows for the LEAGUE's scoring format before
    # it votes, so a half-PPR or standard league gets a format-appropriate
    # consensus. FFC is already handled upstream: build_board format-selects
    # the `adp` table before it becomes this board's `adp` column (and thus
    # `ffc_rank`). ESPN is exempt -- espn_adp is PPR-only by contract, and
    # espn_ppr_rank / espn_id must stay PPR wherever they are read. `fmt`
    # defaults to 'ppr', so callers that don't pass it (and PPR leagues) get
    # exactly today's consensus.
    fp = select_format(fp, fmt)
    mfl = select_format(mfl, fmt)
    cbs = select_format(cbs, fmt)
    out["ffc_rank"] = out["adp"].rank(method="first")
    out["espn_rank"], out["espn_ppr_rank"], out["espn_id"] = _espn_ranks(
        out, espn, sleeper)
    out["fp_rank"], out["fp_tier"] = _fp_ranks(out, fp)
    out["mfl_rank"] = _name_ranks(out, mfl, "mfl_name", "mfl_rank")
    out["cbs_rank"] = _name_ranks(out, cbs, "cbs_name", "cbs_rank")
    # Above this, ESPN's PPR rank is not an opinion about draft position --
    # it is ESPN saying the player is not a fantasy player at all. Their rank
    # runs across a 2565-player universe, and the real board ends at 519 with
    # a clean empty gap up to 978; every player above it in this feed also
    # has a null projection and a filler ADP of ~169.9. Tyreek Hill (retired)
    # reads 1899, Keenan Allen 1930, Najee Harris 2047.
    #
    # Dropped rather than kept, because a median cannot use it: two stale
    # sources still carrying Hill at 184 and 301 outvote the one source that
    # knows he retired, and he lands on the board at 301 -- a retired player
    # occupying a pick in the draft grid. Blanking it lets `espn_unranked`
    # below take him off the board entirely, which is the correct answer.
    ESPN_NOT_A_FANTASY_PLAYER = 600
    espn_out = out["espn_ppr_rank"] > ESPN_NOT_A_FANTASY_PLAYER
    out.loc[espn_out, "espn_ppr_rank"] = np.nan
    # ESPN ranks every fantasy-relevant player except defenses, which it never
    # ranks at all -- so "ESPN has no opinion" means undraftable for everyone
    # but a DST, where it means nothing.
    #
    # Only trusted when ESPN's feed is actually here. A missing, failed or
    # unjoined ESPN feed leaves every row unranked, and a filter that reads
    # that as "nobody is draftable" would empty the board -- turning one
    # source being down into a total outage, which is far worse than carrying
    # a few stale players. Below the threshold the flag is all-False and the
    # board keeps everyone.
    covered = out.loc[out["position"] != "DST", "espn_ppr_rank"].notna()
    trustworthy = int(covered.sum()) >= ESPN_MIN_RANKED
    out["espn_unranked"] = (out["espn_ppr_rank"].isna()
                            & (out["position"] != "DST")) & trustworthy

    ranks = out[_RANK_COLS].astype(float)
    # Median, not mean. These five sources are not equally reliable, and the
    # mean hands the worst of them a full vote. Measured against the median
    # of the other four across the top 100, mean absolute deviation runs
    # 10.7 (FantasyPros), 12.1 (FFC), 13.3 (CBS), 21.2 (ESPN) -- and 43.0
    # for MFL, whose p90 miss is 91 ranks and whose worst is 107. MFL's ADP
    # is drawn largely from best-ball and dynasty rooms drafted months
    # earlier, so it is not measuring the same event as the others.
    #
    # One source 30 ranks out moves a mean of five by 6 -- most of a round in
    # an 8-team league -- and it lands on exactly the players the sources
    # disagree about, which are the ones worth being right about. Chase Brown
    # sits at 9/12/16/21 across four sources and 43 on MFL; the mean puts him
    # 15th and the median 12th. Switching moved 40 of the top 100 by five or
    # more places.
    #
    # The median keeps every source as a vote and lets none of them drag the
    # answer, which is what a consensus is for. `market_spread` still reports
    # the full min-to-max range, so a disagreement this wide stays visible on
    # the board rather than being smoothed away here.
    out["market_rank"] = ranks.median(axis=1, skipna=True).round(1)
    n = ranks.notna().sum(axis=1)
    spread = ranks.max(axis=1) - ranks.min(axis=1)
    out["market_spread"] = spread.where(n >= 2)
    def _val(v):
        return None if pd.isna(v) else float(v)
    out["market_sources"] = [
        {"ffc": _val(r.ffc_rank), "espn": _val(r.espn_ppr_rank), "fp": _val(r.fp_rank),
         "mfl": _val(r.mfl_rank), "cbs": _val(r.cbs_rank),
         "fp_tier": None if pd.isna(r.fp_tier) else int(r.fp_tier)}
        for r in out.itertuples()]
    out["edge"] = out["market_rank"] - out["rank"]
    return out.drop(columns=["adp"] + _DROP_COLS + ["fp_tier"])
