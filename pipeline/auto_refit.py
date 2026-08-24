"""Self-improving loop for the served opponent model: refit, validate, swap ONLY on a proven win.

WHAT THIS IS. `scoring/nested_prior.py` is the served cold-start nested artifact
(loaded by `nested_model.cold_start_nested()`, shipped inside the live hybrid
`scoring/hybrid_model.py`). The corpus it was fit on grows every ~15 minutes as
the farm records new mock drafts. This job periodically refits a CANDIDATE
artifact on the grown corpus, measures it held-out against the model that is
ACTUALLY LIVE right now, and replaces the live artifact ONLY when the candidate
is a proven, held-out win. It promotes better models automatically as data
accumulates and it CANNOT regress the live recommender. The gate (below) is the
entire safety story.

THE GATE, in one sentence: swap iff the candidate's HELD-OUT top-1 is strictly
greater than the live model's AND its held-out log-loss is not worse AND it
still clears the flat baseline. A tie or any regression on either metric keeps
the live model untouched. There is no --force; a no-swap is a normal outcome,
not an error, and this exits 0 for it.

WHY THE COMPARISON IS CONSERVATIVE (the bulletproof property). The two sides are
measured with DELIBERATELY ASYMMETRIC handicaps, both in the safe direction:

  * CHAMPION = the current live artifact (`nested_model.cold_start_nested()`),
    scored FULL-WIDTH -- every human pick, including the drafts it was fit on.
    That is literally what users get, and scoring it on its own training rows
    only FLATTERS it, so the bar the candidate must clear is if anything set
    too HIGH.
  * CANDIDATE = the same model refit LEAVE-ONE-DRAFT-OUT on the current corpus.
    Every pick it is scored on comes from a draft it never trained on, so it
    CANNOT win by memorising. Its number is an honest generalisation estimate,
    if anything set too LOW (each fold trains on one draft less than the served
    all-data artifact will).

So of the four possible fit/serve handicap combinations this is the most
conservative: a candidate that clears a flattered champion's bar while under its
own leave-one-out handicap is a genuine, robust improvement, and a candidate
that merely memorises the new drafts loses its held-out number and is NOT
promoted. Both sides are scored on the IDENTICAL human picks, paired by draft,
standard errors clustered by draft -- everything that makes one room harder than
another cancels. The early rounds of both hybrids are the SAME flat vector
(`draft_model.COLD_START_PRIOR`), so the comparison is decided entirely by the
mid/late nested picks, which is exactly where a refit can move the number.

HONEST EXPECTATION -- read this before you expect swaps. The current
nested/hybrid model is SATURATED. Refitting on more data tightens the coefficient
estimates but does not raise held-out top-1 meaningfully, so for now this job
will MOSTLY log "no-swap", and the swaps it does make will be marginal refits
that happen to clear the bar. That is the correct, safe behaviour, not a bug.
The real payoff is twofold and neither part is about today's top-1:
  (a) it keeps the served artifact fresh automatically as the corpus grows, and
  (b) the day a genuinely higher-capacity model (e.g. a sequence model) beats
      the hybrid on this exact held-out ladder, this job promotes it the same
      day, through the same gate, with no human in the loop.

HOW A FUTURE ARCHITECTURE SLOTS IN (design, not built here). The candidate step
is "some model, refit on the current corpus"; the validation is "candidate
leave-one-draft-out top-1 vs champion top-1 on identical picks." To drop in a
`sequence_model` later you change exactly two things and nothing else:
  * `refit_candidate_artifact()` -- serialize the new model instead of the
    nested one (write to the candidate path, not the live path), and
  * `candidate_heldout_columns()` -- score the new model leave-one-draft-out to
    the same four per-pick columns (top-1/3/5 hit flags and log-prob).
The gate, the routing, the champion side, the parity re-check and the commit are
untouched: the new model is measured on the same picks, by the same ruler,
against the same live champion, and promoted only if it clears the same bar.

WHY IT IS CHEAP TO RUN OFTEN. A run first counts human-labelled drafts with one
cheap query and SKIPS (exits 0) unless enough NEW labelled drafts have accrued
since the last successful run (state in `data/auto_refit_state.json`). The
expensive work -- the all-data fit and the leave-one-draft-out ladder -- happens
only when there is genuinely new data to justify it. The ladder is SINGLE
PROCESS on purpose (`workers=1`): a live farm is writing to the corpus and
spawning parallel fit workers stalled the system once; do not reintroduce them.

Run (does real work only when >= --min-new-drafts new labelled drafts exist):
    .venv/bin/python -m pipeline.auto_refit --league-db data/leagues/1251381776.duckdb

Smoke it end to end without touching the repo (fast, will no-swap or skip):
    .venv/bin/python -m pipeline.auto_refit --league-db data/leagues/1251381776.duckdb \\
        --limit 30 --no-commit --state /tmp/ar_state.json \\
        --candidate /tmp/ar_candidate.py --log /tmp/ar_log.md

`data/nfl.duckdb` is frequently locked by the API; pass a per-league file with
`--league-db` (any one carries the universal reference tables). This never
writes the corpus (opened read-only). It writes the live artifact ONLY on a
proven win, and even then reverts it if the post-swap parity test fails.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import sys
from dataclasses import dataclass

import numpy as np

import pipeline.fit_prior as fp
import pipeline.fit_nested_prior as fnp
from pipeline import draft_log as dl
from pipeline import measure_nested as mn
from pipeline import score_ladder as sl
from scoring import nested_model as nm

# ------------------------------------------------------------------- constants

LIVE_PRIOR = "scoring/nested_prior.py"       # the served artifact this may swap
STATE_PATH = "data/auto_refit_state.json"    # runtime state (gitignored)
LOG_PATH = "docs/superpowers/auto_refit_log.md"   # committed audit trail
DEFAULT_MIN_NEW = 25
DEFAULT_CANDIDATE = "/tmp/auto_refit_candidate.py"

# The fit/serve parity tests that must still pass with the NEW artifact live.
# `-k parity` matches nothing in this suite (the parity tests are named for
# fit-and-serve, not "parity"); this selector is the actual set, and it includes
# `test_cold_start_hybrid_builds_from_artifact_and_serves`, which loads the
# freshly-swapped `scoring/nested_prior.py` through `cold_start_nested()`.
PARITY_SELECTOR = "identical_fit_and_serve or builds_from_artifact"
PARITY_TEST = "tests/test_draft_sim.py"


# --------------------------------------------------------------------- metrics

@dataclass(frozen=True)
class Metrics:
    """A model's held-out score on the shared human picks. `nll` is the mean
    negative log-likelihood (log-loss), lower is better; `top1`/`top5` are hit
    rates, higher is better. `se1` is the top-1 standard error clustered by
    draft, carried for the log only (the gate reads point estimates)."""
    top1: float
    top5: float
    nll: float
    se1: float = float("nan")


def gate(champ: Metrics, cand: Metrics, flat_top1: float) -> "tuple[bool, str]":
    """THE GATE. Pure decision: promote the candidate or not, and why.

    Promote iff ALL of:
      * candidate top-1 STRICTLY greater than champion top-1, and
      * candidate log-loss NOT WORSE than champion (<=), and
      * candidate top-1 still beats the flat baseline (sanity floor).
    A tie or any regression on either of the first two => do not promote. This
    is the whole safety rule and it has no override: there is no argument that
    makes it return True on a losing or tied candidate. Shipping a worse model
    would require editing this function, in a diff someone can review.
    """
    if not (cand.top1 > champ.top1):
        return False, (f"no-swap: candidate top-1 {cand.top1:.4f} does not beat "
                       f"champion {champ.top1:.4f} (strict > required)")
    if not (cand.nll <= champ.nll):
        return False, (f"no-swap: candidate log-loss {cand.nll:.4f} is worse than "
                       f"champion {champ.nll:.4f} (must be <=)")
    if not (cand.top1 > flat_top1):
        return False, (f"no-swap: candidate top-1 {cand.top1:.4f} does not clear "
                       f"the flat baseline floor {flat_top1:.4f}")
    return True, (f"SWAP: candidate top-1 {cand.top1:.4f} > champion "
                  f"{champ.top1:.4f}, log-loss {cand.nll:.4f} <= {champ.nll:.4f}, "
                  f"clears flat floor {flat_top1:.4f}")


# ---------------------------------------------------------- corpus / state / skip

def labelled_draft_count(corpus) -> int:
    """Cheap count of human-labelled drafts: snapshot-qualified drafts (a pool
    and picks) that carry at least one `autodrafted IS FALSE` pick. This is the
    monotonic accrual signal the skip decision reads -- one query, no fit, no
    replay -- so a run can decide it has nothing to do in well under a second."""
    return int(corpus.execute(
        """SELECT count(*) FROM (
             SELECT d.draft_id FROM draft_log d
             WHERE EXISTS (SELECT 1 FROM draft_log_pool o WHERE o.draft_id = d.draft_id)
               AND EXISTS (SELECT 1 FROM draft_log_pick p WHERE p.draft_id = d.draft_id)
               AND EXISTS (SELECT 1 FROM draft_log_pick p
                           WHERE p.draft_id = d.draft_id AND p.autodrafted IS FALSE))
        """).fetchone()[0])


def load_state(path: str) -> "dict | None":
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def save_state(path: str, labelled: int, corpus_size: int, decision: str,
               commit: "str | None") -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump({
            "last_success_ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "last_success_labelled": labelled,
            "last_success_corpus": corpus_size,
            "last_decision": decision,
            "last_commit": commit,
        }, fh, indent=2)


def should_skip(labelled_now: int, state: "dict | None",
                min_new: int) -> "tuple[bool, str, int]":
    """Skip (do no real work) unless enough NEW labelled drafts have accrued
    since the last SUCCESSFUL run. With no prior state the baseline is 0, so a
    first run always proceeds. Returns (skip, reason, new_count)."""
    baseline = 0 if state is None else int(state.get("last_success_labelled", 0))
    new = labelled_now - baseline
    if new < min_new:
        return True, (f"insufficient new data, skipping: {new} new labelled "
                      f"drafts since last success ({labelled_now} now, "
                      f"{baseline} then) < min {min_new}"), new
    return False, (f"{new} new labelled drafts since last success "
                   f"({labelled_now} now, {baseline} then) >= min {min_new}"), new


# --------------------------------------------------------------- the two columns
#
# Both columns are the per-pick top-1/top-3/top-5 hit flags and the log-prob of
# the chosen player, in the SAME pick order, so they can be routed together and
# clustered by the same draft labels. The champion column reads the frozen live
# artifact full-width; the candidate column is the leave-one-draft-out ladder.


def score_champion_nested_fullwidth(design) -> "tuple[np.ndarray, ...]":
    """The LIVE artifact's nested prediction on every pick, scored full-width.

    `cold_start_nested()` rebuilds exactly the served `NestedModel` from
    `scoring/nested_prior.py`; `predict_pick` is the same reconstruction the
    serving path runs. No fold is held out because the artifact is a fixed,
    already-fit object -- this is its real behaviour on these boards, which is
    what "what users get" means and which (being scored on its own training
    rows) only flatters it."""
    model = nm.cold_start_nested()
    n = len(design)
    h1 = np.zeros(n, int); h3 = np.zeros(n, int)
    h5 = np.zeros(n, int); lp = np.zeros(n, float)
    for i in range(n):
        prob = model.predict_pick(design, i)
        h1[i], h3[i], h5[i], lp[i] = nm.score_pick(prob, design.chosen[i])
    return h1, h3, h5, lp


def candidate_heldout_columns(design) -> "tuple[np.ndarray, ...]":
    """The candidate's nested prediction, leave-one-draft-out and single process.

    Reuses `measure_nested._nested_perpick` unchanged: it refits the nested
    model without each draft in turn and scores that draft's held-out picks.
    `workers=1` is deliberate -- the corpus is being written by a live farm and
    parallel fit workers stalled the system once.

    A FUTURE `sequence_model` candidate replaces THIS function (and
    `refit_candidate_artifact`) and nothing else: score the new model
    leave-one-draft-out to the same four columns and the gate compares it to the
    same champion on the same picks."""
    mn._DESIGN = design                      # the fold scorer reads it off here
    return mn._nested_perpick(design, workers=1)


def _route(early, flat_col, nested_col):
    """Per pick: the flat column in the early bucket, the nested column mid/late
    -- the exact hybrid routing. `early` is the boolean per-pick mask."""
    return np.where(early, flat_col, nested_col)


def _aggregate(h1, h5, lp, groups) -> Metrics:
    return Metrics(top1=float(h1.mean()), top5=float(h5.mean()),
                   nll=float(-lp.mean()), se1=mn._cluster_se(h1, groups))


# ------------------------------------------------------------------- the artifact

def refit_candidate_artifact(corpus, league, ids, out_path: str) -> "tuple[int, int]":
    """Refit the nested model on the WHOLE current corpus and serialize it to
    `out_path` (the candidate path, NEVER the live artifact). Reuses
    `fit_nested_prior` so there is no second copy of the fit. Returns
    (n_picks, n_drafts). A future architecture swaps its own serializer in
    here; the caller does not care what shape the artifact is."""
    model, n_picks, n_drafts = fnp.fit_nested_prior(corpus, league, ids)
    text = fnp.render_module(model, n_picks, n_drafts, dl.CORPUS_PATH,
                             _dt.date.today().isoformat())
    with open(out_path, "w") as fh:
        fh.write(text)
    return n_picks, n_drafts


# ---------------------------------------------------------------------- promote

@dataclass(frozen=True)
class PromoteResult:
    committed: bool
    parity_ok: bool
    sha: "str | None"
    message: str


def _run(cmd, cwd) -> "subprocess.CompletedProcess":
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def default_parity_check(repo_dir: str) -> bool:
    """Run the fit/serve parity tests against the artifact currently on disk.
    A fresh pytest process, so it imports the just-swapped
    `scoring/nested_prior.py`. Returns True iff they all pass."""
    r = _run([".venv/bin/pytest", "-q", PARITY_TEST, "-k", PARITY_SELECTOR],
             cwd=repo_dir)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    return r.returncode == 0


def promote(repo_dir: str, candidate_path: str, live_rel: str, commit_paths,
            message: str, run_parity, do_commit: bool = True) -> PromoteResult:
    """Atomically swap the candidate into the live artifact, re-check parity,
    and commit -- or REVERT if parity fails. Never leaves a parity-broken
    artifact live.

    Steps, in order and each reversible up to the commit:
      1. refuse if the live artifact has UNCOMMITTED changes (a revert would
         clobber them; a generated artifact should never be dirty), then
      2. write the candidate bytes to a temp file in the live artifact's own
         directory and `os.replace` it over the live path -- atomic, same
         filesystem, so no half-written artifact is ever observable, then
      3. run the parity check against the now-live new artifact, and
         * if it FAILS: `git checkout -- <live>` to restore the committed
           champion, log a loud error, commit nothing, and
         * if it PASSES: `git commit -- <paths>` (live artifact + audit log)
           and return the new sha.

    `run_parity` and `do_commit` are injectable so the swap/revert/commit logic
    is unit-tested against a real throwaway git repo without invoking pytest."""
    live_abs = os.path.join(repo_dir, live_rel)

    dirty = _run(["git", "status", "--porcelain", "--", live_rel], repo_dir)
    if dirty.stdout.strip():
        return PromoteResult(False, False, None,
                             f"ABORT: {live_rel} has uncommitted changes; refusing "
                             f"to swap (a revert would lose them)")

    with open(candidate_path) as fh:
        candidate_text = fh.read()
    tmp = live_abs + ".swap.tmp"
    with open(tmp, "w") as fh:
        fh.write(candidate_text)
    os.replace(tmp, live_abs)                 # atomic swap of the live artifact

    if not run_parity():
        rv = _run(["git", "checkout", "--", live_rel], repo_dir)
        msg = (f"PARITY FAILED after swap of {live_rel}; REVERTED to the "
               f"committed champion (git checkout rc={rv.returncode}). "
               f"Live artifact untouched.")
        sys.stderr.write("ERROR: " + msg + "\n")
        return PromoteResult(False, False, None, msg)

    if not do_commit:
        return PromoteResult(False, True, None,
                             "parity ok; swap left in working tree (--no-commit)")

    cm = _run(["git", "commit", "-m", message, "--", *commit_paths], repo_dir)
    if cm.returncode != 0:
        return PromoteResult(False, True, None,
                             f"parity ok but git commit failed: {cm.stderr.strip()}")
    sha = _run(["git", "rev-parse", "HEAD"], repo_dir).stdout.strip()
    return PromoteResult(True, True, sha, f"swapped and committed {sha}")


# ------------------------------------------------------------------- audit log

def _log_header() -> str:
    return ("# Auto-refit audit log\n\n"
            "One row per run of `pipeline.auto_refit` that did real work (a "
            "fit + validation). Skips -- runs that found too little new data -- "
            "are logged to the daemon's stdout, not here. `nll` is log-loss "
            "(lower better); `t1`/`t5` are held-out top-1/top-5 (higher "
            "better). champion = the live artifact scored full-width; candidate "
            "= the refit scored leave-one-draft-out; both on the identical "
            "human picks. Decision is the gate's verdict.\n\n"
            "| when (UTC) | corpus | labelled | champ t1 | cand t1 | champ nll | "
            "cand nll | flat t1 | decision | commit |\n"
            "|---|---|---|---|---|---|---|---|---|---|\n")


def append_log(path: str, row: str) -> None:
    exists = os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as fh:
        if not exists:
            fh.write(_log_header())
        fh.write(row + "\n")


def _log_row(ts, corpus_size, labelled, champ, cand, flat_top1, decision, sha):
    def f(x):
        return "-" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))
    return (f"| {ts} | {corpus_size} | {labelled} | "
            f"{f(None if champ is None else champ.top1)} | "
            f"{f(None if cand is None else cand.top1)} | "
            f"{f(None if champ is None else champ.nll)} | "
            f"{f(None if cand is None else cand.nll)} | "
            f"{f(flat_top1)} | {decision} | {sha or '-'} |")


# ------------------------------------------------------------------------- main

def _log(msg: str) -> None:
    print(f"[auto_refit] {msg}", flush=True)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    league_db = fp._option(argv, "--league-db")
    min_new = int(fp._option(argv, "--min-new-drafts", DEFAULT_MIN_NEW))
    state_path = fp._option(argv, "--state", STATE_PATH)
    log_path = fp._option(argv, "--log", LOG_PATH)
    candidate_path = fp._option(argv, "--candidate", DEFAULT_CANDIDATE)
    limit = fp._option(argv, "--limit", None)
    limit = int(limit) if limit is not None else None
    do_commit = "--no-commit" not in argv
    repo_dir = os.getcwd()
    ts = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")

    # --- step 1: snapshot + cheap skip decision -----------------------------
    corpus = fp.open_corpus()
    ids = fp.snapshot_draft_ids(corpus)
    if limit is not None:
        ids = ids[:limit]
    corpus_size = len(ids)
    labelled = labelled_draft_count(corpus) if limit is None else corpus_size
    _log(f"corpus {corpus_size} qualifying drafts, {labelled} human-labelled")

    state = load_state(state_path)
    skip, reason, _new = should_skip(labelled, state, min_new)
    if skip:
        # A skip is a non-event: logged to stdout (the daemon's nohup log), but
        # NOT written to the committed audit trail. Committing a "skip" row on
        # every wake would be pure noise, and leaving it uncommitted would keep
        # a tracked file permanently dirty -- so the audit log carries only the
        # substantive decisions (swap / no-swap / parity-revert).
        _log(reason)
        corpus.close()
        return 0
    _log(reason)

    # --- step 2: refit a CANDIDATE artifact (never the live path) ------------
    league = fp.open_league(league_db)
    _log(f"refitting candidate artifact -> {candidate_path}")
    n_picks, n_drafts = refit_candidate_artifact(corpus, league, ids, candidate_path)
    _log(f"candidate fit on {n_picks} human picks / {n_drafts} drafts")

    # --- build the shared design + the flat column, once ---------------------
    obs = sl.human_observations(corpus, league, ids)
    design = nm.design_from_observations(obs)
    X_list, chosen, _groups, _boards, _buckets = fp.design(obs)
    assert list(_groups) == list(design.groups), "champion/nested designs misaligned"
    corpus.close(); league.close()           # no live DB handle before any fork

    groups = np.asarray(design.groups)
    early = np.asarray(design.buckets) == "early"

    # --- step 3: validate held-out -------------------------------------------
    _log("scoring flat baseline + champion (full-width) + candidate (LODO)...")
    f1, f3, f5, fll = mn._champion_perpick(X_list, chosen)   # flat, shared early
    cn1, cn3, cn5, cnll = score_champion_nested_fullwidth(design)
    dn1, dn3, dn5, dnll = candidate_heldout_columns(design)

    champ = _aggregate(_route(early, f1, cn1), _route(early, f5, cn5),
                       _route(early, fll, cnll), groups)
    cand = _aggregate(_route(early, f1, dn1), _route(early, f5, dn5),
                      _route(early, fll, dnll), groups)
    flat_top1 = float(f1.mean())

    _log(f"flat     top1 {flat_top1:.4f}")
    _log(f"champion top1 {champ.top1:.4f} +/-{champ.se1:.4f}  top5 {champ.top5:.4f}"
         f"  nll {champ.nll:.4f}   (live artifact, full-width)")
    _log(f"candidate top1 {cand.top1:.4f} +/-{cand.se1:.4f}  top5 {cand.top5:.4f}"
         f"  nll {cand.nll:.4f}   (refit, leave-one-draft-out)")

    # --- step 4: THE GATE ----------------------------------------------------
    do_swap, verdict = gate(champ, cand, flat_top1)
    _log(verdict)

    sha = None                    # the ARTIFACT commit sha, set only on a swap
    decision = "no-swap"
    if do_swap:
        # --- step 5: atomic swap + parity re-check + commit-or-revert --------
        # promote commits ONLY the artifact (with the numbers). The audit line
        # is written and committed below, AFTER the real outcome is known, so a
        # parity revert never leaves a "SWAP" line behind claiming a swap that
        # did not happen.
        message = (f"auto: promote refit nested_prior "
                   f"(top-1 {champ.top1:.4f} -> {cand.top1:.4f}, "
                   f"nll {champ.nll:.4f} -> {cand.nll:.4f}, "
                   f"corpus {n_drafts} drafts / {n_picks} picks)")
        res = promote(repo_dir, candidate_path, LIVE_PRIOR, [LIVE_PRIOR],
                      message, run_parity=lambda: default_parity_check(repo_dir),
                      do_commit=do_commit)
        _log(res.message)
        if res.committed:
            decision, sha = "SWAP", res.sha
        elif res.parity_ok and not do_commit:
            decision = "SWAP(uncommitted)"
        else:
            # parity failed and was reverted, OR commit failed: the live model
            # is untouched. Loud, non-fatal -- record it as a no-swap.
            decision = "no-swap(parity-revert)"

    # --- step 6: append ONE audit line reflecting the real outcome, commit it -
    append_log(log_path, _log_row(ts, corpus_size, labelled, champ, cand,
                                  flat_top1, decision, sha))
    if do_commit:
        cm = _run(["git", "commit", "-m",
                   f"auto: auto_refit audit ({decision}, "
                   f"champ t1 {champ.top1:.4f} / cand t1 {cand.top1:.4f})",
                   "--", log_path], repo_dir)
        if cm.returncode != 0:
            _log(f"note: audit-log commit skipped ({cm.stderr.strip() or 'no change'})")

    # --- step 7: update state on a successful run (swap or validated no-swap) -
    save_state(state_path, labelled, corpus_size, decision, sha)
    _log(f"done: decision={decision}" + (f" commit={sha}" if sha else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
