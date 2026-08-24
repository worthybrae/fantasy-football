"""The HYBRID cold-start opponent: flat champion early, nested mid/late.

The measured best predictor of a human opponent's pick (full corpus, held out
by draft: `docs/superpowers/findings/2026-08-24-hybrid-model.md`,
`pipeline/measure_hybrid.py`). The nested model (`nested_model.NestedModel`)
beats the flat champion overall but LOSES the early rounds -- there the best
player overall and the best at his position are the same consensus name, so
splitting the call into two factors only dilutes a near-deterministic pick. So
route by round: the flat cold-start prior (`draft_model.COLD_START_PRIOR`) in
the EARLY bucket, the nested model from mid on.

THE ROUTE IS THE EXACT ROUND BUCKET the measurement used and the rest of the
model already uses: `draft_model._round_bucket(overall_pick, teams) == "early"`
(rounds 1..EARLY_ROUNDS). It is not re-spelled here; a change to EARLY_ROUNDS
moves this router with it, and the by-round split in the findings file, in one
place.

ONE SERVING INTERFACE. `HybridModel.predict_serve` has the same signature as
`NestedModel.predict_serve`, so `draft_sim._nested_scores` calls it without
knowing which of the two it holds -- the only change the serving path needs to
carry the hybrid. Both branches read the SAME full live feature matrix the flat
model is served from (`draft_sim._live_features`, pinned equal to
`draft_model.feature_matrix`) and the same live position vector
(`nested_model.position_features_live`, pinned equal to
`position_model.features_for`), so the served hybrid vector equals its fit-side
vector -- flat softmax early, nested reconstruction mid/late -- to
floating-point tolerance, which `tests/test_draft_sim.py` pins.
"""
from __future__ import annotations

import numpy as np

from scoring import draft_model as dm
from scoring import nested_model as nm


class HybridModel:
    """Routes each opponent pick to the flat prior (early) or the nested model.

    Holds the flat cold-start coefficient vector and a fitted `NestedModel`.
    Stateless per call: the round is read off `overall_pick`/`teams`, so one
    instance serves every seat and every pick of a draft.
    """

    def __init__(self, flat_beta, nested: "nm.NestedModel"):
        self.flat_beta = np.asarray(flat_beta, dtype=float)
        assert self.flat_beta.shape == (len(dm.FEATURE_NAMES),), (
            "hybrid flat_beta must have one weight per draft_model feature")
        self.nested = nested

    def predict_serve(self, X, pos_x, cand_pos, overall_pick, teams) -> np.ndarray:
        """The per-candidate probability over the full available board.

        `X` is the full `draft_model.FEATURE_NAMES` feature matrix in board
        order. In the early bucket the flat prior scores it and a single
        max-shifted softmax over the whole board is the distribution -- exactly
        the flat vector's fit-side prediction (`feature_matrix @ COLD_START_PRIOR`,
        softmaxed). From mid on the nested factors take over, reading `pos_x`
        (the context vector) and `X` sliced to the within-position columns.

        The caps/must-fill masking is applied by the caller
        (`draft_sim._sample_from_scores`) on the log of this vector, the same
        for both branches -- setting a capped candidate to -inf and renormalizing
        is "drop it and renormalize the rest", the right thing for either a flat
        or a nested probability.
        """
        X = np.asarray(X, dtype=float)
        if dm._round_bucket(int(overall_pick), int(teams)) == "early":
            return nm._softmax(X @ self.flat_beta)
        return self.nested.predict_from(pos_x, X[:, nm.WITHIN_IDX], cand_pos)


def cold_start_hybrid() -> HybridModel:
    """The default cold-start opponent, built with NO corpus access.

    The flat half is `draft_model.COLD_START_PRIOR` (the shipped flat prior);
    the nested half is `nested_model.cold_start_nested()` (the fitted artifact
    in `scoring/nested_prior.py`). This is what a live/mock draft with no league
    history simulates its opponents with.
    """
    return HybridModel(dm.COLD_START_PRIOR.copy(), nm.cold_start_nested())
