"""Per-owner draft tendencies, weighted by how well each one is measured.

Built from `pipeline.draft_log`'s corpus rather than from any one league, and
rebuildable from it at any time -- nothing here is a source of truth, which is
the point: E006 threw away two traits and discovered a third only after the
fact, and a profile store that could not be recomputed would have been dead
weight.

WHY SHRINKAGE RATHER THAN A LIST OF TRAITS WE BELIEVE IN. E006 asked which
tendencies persist and found reach and roster shape do, while youth and
weekly-steadiness preferences could not be detected at eight managers over six
seasons -- not because they are absent, but because confirming an effect that
small needs about 220 owner-season pairs and there were 40. That is an
awkward result to encode as a yes/no. Shrinkage does not require the question
to be answered: each trait is pulled toward the population by exactly how
poorly it separates owners, so a real habit measured on many drafts is used
nearly as-is, and a trait that is mostly noise contributes almost nothing
because its personal and population values converge. Add drafts and the
weights move on their own.

The population baseline includes ANONYMOUS owners -- mock opponents nobody can
identify. They can never hold a personal profile, but they are most of what a
first-time owner is scored against, and excluding them would build the
baseline from the handful of people we happen to know.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Same decay E006 settled on: a manager's late picks are constrained by what
# is left and by the slots still open, so they say less about him than his
# early ones. Measured there at +0.52 against +0.41 for truncating at round 5.
ROUND_DECAY = 0.8
# Positions whose FINISHED count is the trait. Deliberately a count of the
# roster rather than a share of the early rounds: E006 measured tight end at
# +0.69 season to season this way and at nothing at all the other way.
SHAPE_POSITIONS = ("QB", "RB", "WR", "TE")
# Bounds on the shrinkage constant. A trait that looks perfectly separated on
# three drafts is not, and one that looks like pure noise still should not be
# discarded outright -- both are small samples making confident claims.
K_MIN, K_MAX = 0.25, 40.0

PROFILE_COLUMNS = ["owner_key", "trait", "personal", "population", "n_drafts",
                   "weight", "value"]


def _per_draft(picks: pd.DataFrame) -> pd.DataFrame:
    """One row per (owner, draft): the traits that draft expressed."""
    p = picks.copy()
    p["round"] = pd.to_numeric(p["round"], errors="coerce")
    p["adp_rank"] = pd.to_numeric(p["adp_rank"], errors="coerce")
    p["pick_no"] = pd.to_numeric(p["pick_no"], errors="coerce")
    # POSITIVE means he took the player earlier than the market had him.
    p["_reach"] = p["adp_rank"] - p["pick_no"]
    p["_w"] = ROUND_DECAY ** (p["round"].fillna(1) - 1)

    rows = []
    have = p.dropna(subset=["_reach"])
    if not have.empty:
        g = have.assign(_x=have["_reach"] * have["_w"]).groupby(
            ["owner_key", "draft_id", "is_anonymous"]).agg(
            x=("_x", "sum"), w=("_w", "sum")).reset_index()
        g = g[g["w"] > 0]
        rows.append(g.assign(trait="reach", value=g["x"] / g["w"])[
            ["owner_key", "draft_id", "is_anonymous", "trait", "value"]])
    for pos in SHAPE_POSITIONS:
        g = (p.assign(_hit=(p["position"] == pos).astype(float))
             .groupby(["owner_key", "draft_id", "is_anonymous"])["_hit"]
             .sum().rename("value").reset_index())
        rows.append(g.assign(trait=f"shape_{pos}")[
            ["owner_key", "draft_id", "is_anonymous", "trait", "value"]])
    if not rows:
        return pd.DataFrame(columns=["owner_key", "draft_id", "is_anonymous",
                                     "trait", "value"])
    return pd.concat(rows, ignore_index=True)


def _shrink_constant(per_draft: pd.DataFrame) -> float:
    """How many drafts it takes before a personal value outweighs the pooled one.

    K = within-owner variance over between-owner variance, so `weight =
    n / (n + K)` is the empirical-Bayes weight on a mean of n drafts.

    The between-owner term is CORRECTED before dividing. The spread of owner
    means is inflated by sampling noise -- each mean is itself an average of a
    few noisy drafts -- and using it raw would credit every trait with more
    genuine separation than it has, which is exactly the flattery this whole
    module exists to avoid.
    """
    counts = per_draft.groupby("owner_key")["value"].count()
    usable = counts[counts >= 2].index
    if len(usable) < 2:
        return K_MAX
    sub = per_draft[per_draft["owner_key"].isin(usable)]
    within = float(sub.groupby("owner_key")["value"].var(ddof=1).mean())
    means = sub.groupby("owner_key")["value"].mean()
    n_bar = float(counts[usable].mean())
    between = float(means.var(ddof=1)) - within / max(n_bar, 1.0)
    if not np.isfinite(within) or within <= 0:
        return K_MIN
    if not np.isfinite(between) or between <= 0:
        return K_MAX
    return float(np.clip(within / between, K_MIN, K_MAX))


def build(picks: pd.DataFrame) -> pd.DataFrame:
    """Shrunk trait values per owner, plus what they were shrunk toward."""
    if picks.empty:
        return pd.DataFrame(columns=PROFILE_COLUMNS)
    per_draft = _per_draft(picks)
    if per_draft.empty:
        return pd.DataFrame(columns=PROFILE_COLUMNS)

    out = []
    for trait, rows in per_draft.groupby("trait"):
        k = _shrink_constant(rows)
        # The baseline is the mean over OWNERS, not over drafts: otherwise
        # whoever has played the most mocks becomes the population.
        population = float(rows.groupby("owner_key")["value"].mean().mean())
        named = rows[~rows["is_anonymous"].astype(bool)]
        if named.empty:
            continue
        agg = named.groupby("owner_key")["value"].agg(["mean", "count"])
        weight = agg["count"] / (agg["count"] + k)
        out.append(pd.DataFrame({
            "owner_key": agg.index, "trait": trait,
            "personal": agg["mean"].to_numpy(),
            "population": population,
            "n_drafts": agg["count"].to_numpy(),
            "weight": weight.to_numpy(),
            "value": (weight * agg["mean"] + (1 - weight) * population).to_numpy(),
        }))
    if not out:
        return pd.DataFrame(columns=PROFILE_COLUMNS)
    return pd.concat(out, ignore_index=True)[PROFILE_COLUMNS].sort_values(
        ["owner_key", "trait"]).reset_index(drop=True)
