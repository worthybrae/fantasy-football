"""Regenerate `scoring/nested_prior.py`, the committed cold-start nested prior.

WHY THIS EXISTS. `scoring/nested_prior.py` is a GENERATED module: the two
factors of a `nested_model.NestedModel` (the position classifier's
standardization and softmax weights, and the shared within-position
conditional-logit vector) serialized so `nested_model.cold_start_nested()` can
rebuild the cold-start nested opponent with NO corpus access at draft time.
That artifact ships inside the live hybrid opponent (`scoring/hybrid_model.py`).
It was fit once from a script that lived in scratch and was never committed, so
the live artifact could not be reproduced or audited from source. This is that
script, committed, so the artifact is regenerable going forward.

WHAT IT DOES, and why it matches the measurement exactly. It reads the
human-pick corpus the SAME way `pipeline.measure_nested` /
`pipeline.measure_hybrid` do -- `score_ladder.human_observations` (human picks
only, `autodrafted IS FALSE`, our seat excluded) turned into a
`nested_model.NestedDesign` by `nested_model.design_from_observations` -- fits
one `nested_model.NestedModel` on the WHOLE corpus (no leave-one-out; the
served artifact is the all-data fit, not a fold), and serializes its parameters
in the exact shape `cold_start_nested()` loads: POSITION_LAM, POSITION_FEATURES,
WITHIN_FEATURES, POSITION_MEAN/STD/W/B and WITHIN_BETA. The fit reuses the two
committed fitters unchanged (`position_model.MultinomialLogistic`,
`draft_model.fit`) through `NestedModel.fit`, so there is no third copy of
either softmax and this script cannot drift from what the model actually is.

THE COMMITTED ARTIFACT WAS FIT AT 76 DRAFTS. The live `scoring/nested_prior.py`
in the tree was fit on 5,557 known-human picks across 76 drafts and validated:
the hybrid it feeds passes the `tests/test_draft_sim.py` fit/serve parity gates
and measured at or above the flat baseline on the human-only ladder. The corpus
has grown since (the farm adds a draft roughly every fifteen minutes), so
running this script TODAY fits on more drafts and produces DIFFERENT numbers --
a refit, not a reproduction. That refit is not automatically the validated live
model. So this script writes to a path you name; regenerating the live artifact
is a deliberate act that MUST be re-validated before it is committed:

  1. `.venv/bin/pytest tests/test_draft_sim.py` -- the fit/serve parity gates,
     including `test_cold_start_hybrid_builds_from_artifact_and_serves` which
     loads the regenerated file through `cold_start_nested()`.
  2. `.venv/bin/python -m pipeline.measure_hybrid --league-db <league.duckdb>`
     -- confirm the regenerated hybrid still measures >= the flat baseline on
     held-out human picks. If it does not, DO NOT commit the refit: keep the
     validated artifact and treat the regression as the finding.

Do not overwrite `scoring/nested_prior.py` with an unvalidated refit.

Run (writes to scratch by default, NOT over the live artifact):
    .venv/bin/python -m pipeline.fit_nested_prior \\
        --league-db data/leagues/1251381776.duckdb \\
        --out /tmp/nested_prior_refit.py

`data/nfl.duckdb` is frequently locked by the API; pass a per-league file with
`--league-db` (any one carries the universal reference tables). Pass
`--out scoring/nested_prior.py` ONLY after the two checks above pass.
"""
from __future__ import annotations

import datetime as _dt
import sys

import numpy as np

import pipeline.fit_prior as fp
from pipeline import draft_log as dl
from pipeline import score_ladder as sl
from scoring import nested_model as nm
from scoring import position_model as pm


# ------------------------------------------------------------------ serialization
#
# The generated module is written to reproduce the shape `cold_start_nested()`
# reads and the layout the hand-checked live artifact already had: the two
# feature-order lists on one line each (they are read, not diffed value by
# value), and the numeric arrays six-per-line so a diff between two generations
# shows which coefficients moved. Numbers are full `repr` precision -- the
# artifact is the fit, and a rounded artifact would not reconstruct the fitted
# model.

_MODULE_TEMPLATE = '''"""GENERATED cold-start nested opponent prior. Do not edit by hand.

The nested analogue of `scoring/human_prior.py` / `scoring/mock_prior.py`: a
pre-fit `nested_model.NestedModel` serialized so the serving path can build the
cold-start nested opponent with NO corpus access at draft time.

P(player) = P(position | state) x P(player | position, board). Two factors:
  * the position factor is a `position_model.MultinomialLogistic` (lam={lam!r}),
    stored as its train-set standardization (mean/std) and its softmax weights
    (W, b) -- everything `predict_proba` needs.
  * the within factor is one shared `draft_model.fit` conditional-logit vector
    over same-position candidates, in `WITHIN_FEATURES` order.

Fitted on {n_picks} known-human picks (`autodrafted IS FALSE`, our seat
excluded) across {n_drafts} drafts of the cross-league corpus
({corpus}), on {date}. Regenerate with `python -m pipeline.fit_nested_prior`.
Loaded by `nested_model.cold_start_nested()`, which reconstructs the NestedModel
and checks the two feature-order lists below against the live definitions -- a
column reordered or renamed since this was written trips those asserts rather
than silently reading the wrong coefficient.
"""
import numpy as np

# The position factor's L2 strength, passed to MultinomialLogistic on rebuild.
POSITION_LAM = {lam!r}

# Feature-order guards. POSITION_FEATURES must equal position_model.FEATURE_NAMES
# and WITHIN_FEATURES must equal nested_model.WITHIN_POSITION_FEATURES at load
# time; the loader asserts both.
POSITION_FEATURES = {position_features!r}

WITHIN_FEATURES = {within_features!r}

# Standardization learned on the training rows (subtracted then divided
# before the softmax), MultinomialLogistic.mean_ / .std_.
POSITION_MEAN = {position_mean}

POSITION_STD = {position_std}

# Softmax weights over the six positions: W_ is (K, d), b_ is (K,).
POSITION_W = {position_w}

POSITION_B = {position_b}

# The within-position conditional-logit vector, in WITHIN_FEATURES order.
WITHIN_BETA = {within_beta}
'''


def _fmt_1d(arr, indent="    ", per_line=6) -> str:
    """`np.array([...])` with the values six to a line, full precision."""
    vals = [repr(float(v)) for v in np.asarray(arr, dtype=float).ravel()]
    rows = [indent + ", ".join(vals[i:i + per_line]) + ","
            for i in range(0, len(vals), per_line)]
    return "np.array([\n" + "\n".join(rows) + "\n])"


def _fmt_2d(arr, indent="    ", inner="        ", per_line=6) -> str:
    """`np.array([[...], ...])` -- one bracketed block per row, six per line."""
    arr = np.asarray(arr, dtype=float)
    blocks = []
    for row in arr:
        vals = [repr(float(v)) for v in row]
        lines = [inner + ", ".join(vals[i:i + per_line]) + ","
                 for i in range(0, len(vals), per_line)]
        blocks.append(indent + "[\n" + "\n".join(lines) + "\n" + indent + "],")
    return "np.array([\n" + "\n".join(blocks) + "\n])"


def render_module(model: "nm.NestedModel", n_picks: int, n_drafts: int,
                  corpus: str, date: str) -> str:
    """The text of `scoring/nested_prior.py` for a fitted `NestedModel`."""
    clf = model.position_
    return _MODULE_TEMPLATE.format(
        lam=float(model.position_lam),
        n_picks=n_picks,
        n_drafts=n_drafts,
        corpus=corpus,
        date=date,
        position_features=list(pm.FEATURE_NAMES),
        within_features=list(nm.WITHIN_POSITION_FEATURES),
        position_mean=_fmt_1d(clf.mean_),
        position_std=_fmt_1d(clf.std_),
        position_w=_fmt_2d(clf.W_),
        position_b=_fmt_1d(clf.b_),
        within_beta=_fmt_1d(model.within_),
    )


# ------------------------------------------------------------------------- the fit

def fit_nested_prior(corpus, league, draft_ids) -> "tuple[nm.NestedModel, int, int]":
    """Fit one `NestedModel` on the whole human-pick corpus.

    Returns the fitted model, the human-pick count and the draft count -- the
    two numbers the generated module's provenance prose records.
    """
    obs = sl.human_observations(corpus, league, draft_ids)
    design = nm.design_from_observations(obs)
    model = nm.NestedModel().fit(design)          # idx=None -> the whole corpus
    n_drafts = len(set(design.groups))
    return model, len(design), n_drafts


def _now() -> str:
    return _dt.date.today().isoformat()


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    league_db = fp._option(argv, "--league-db")
    out = fp._option(argv, "--out", "/tmp/nested_prior_refit.py")

    corpus = fp.open_corpus()
    league = fp.open_league(league_db)
    ids = fp.snapshot_draft_ids(corpus)
    print(f"corpus drafts {len(ids)}", flush=True)

    model, n_picks, n_drafts = fit_nested_prior(corpus, league, ids)
    corpus.close()
    league.close()

    text = render_module(model, n_picks, n_drafts, dl.CORPUS_PATH, _now())
    with open(out, "w") as fh:
        fh.write(text)
    print(f"fit on {n_picks} human picks / {n_drafts} drafts; wrote {out}",
          flush=True)
    if out.endswith("scoring/nested_prior.py"):
        print("WARNING: you wrote the LIVE artifact. Re-validate before "
              "committing: pytest tests/test_draft_sim.py and "
              "python -m pipeline.measure_hybrid.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
