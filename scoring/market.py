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

def _norm(name):
    from scoring.board import _norm_name  # deferred: avoids circular import
    return _norm_name(name)

def _espn_ranks(board, espn, sleeper):
    out = pd.Series(np.nan, index=board.index)
    ppr = pd.Series(np.nan, index=board.index)
    if espn is None or espn.empty:
        return out, ppr
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
    # name+position fallback for espn rows without a crosswalk hit (never DST)
    rest = e[e["gsis_id"].isna() & (e["position"] != "DST")].copy()
    if not rest.empty:
        rest["norm"] = rest["espn_name"].map(_norm)
        # Dedupe by (norm, position), keeping lowest rank (best)
        rest = rest.sort_values("espn_rank").drop_duplicates(["norm", "position"], keep="first")
        by_name = rest.set_index(["norm", "position"])["espn_rank"]
        by_name_ppr = rest.set_index(["norm", "position"])["espn_ppr_rank"]
        key = pd.MultiIndex.from_arrays([board["name"].map(_norm), board["position"]])
        fallback = pd.Series(by_name.reindex(key).to_numpy(), index=board.index)
        fallback_ppr = pd.Series(by_name_ppr.reindex(key).to_numpy(), index=board.index)
        mapped = mapped.fillna(fallback)
        mapped_ppr = mapped_ppr.fillna(fallback_ppr)
    return mapped, mapped_ppr

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

def add_market(board, espn, fp, sleeper, mfl=None, cbs=None):
    out = board.copy()
    out["ffc_rank"] = out["adp"].rank(method="first")
    out["espn_rank"], out["espn_ppr_rank"] = _espn_ranks(out, espn, sleeper)
    out["fp_rank"], out["fp_tier"] = _fp_ranks(out, fp)
    out["mfl_rank"] = _name_ranks(out, mfl, "mfl_name", "mfl_rank")
    out["cbs_rank"] = _name_ranks(out, cbs, "cbs_name", "cbs_rank")
    ranks = out[_RANK_COLS].astype(float)
    out["market_rank"] = ranks.mean(axis=1, skipna=True).round(1)
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
