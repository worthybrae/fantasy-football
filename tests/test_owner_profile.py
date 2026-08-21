"""Owner profiles: trait values weighted by how well they are measured.

The shrinkage is the whole design. E006 could not answer "is this tendency
real" for half the traits it tested -- confirming an effect that small needs
about 220 owner-season pairs and there were 40 -- so the profile does not ask
the question. It weights each trait by how much it separates owners, which
needs no verdict and improves on its own as drafts accumulate.
"""
import numpy as np
import pandas as pd
import pytest

from scoring.owner_profile import K_MAX, build


def _picks(owner, draft_id, positions, anon=False, reach=0.0):
    n = len(positions)
    return pd.DataFrame({
        "draft_id": draft_id, "owner_key": owner, "is_anonymous": anon,
        "pick_no": range(1, n + 1), "round": range(1, n + 1),
        "position": positions,
        # adp_rank set so that (adp_rank - pick_no) is exactly `reach`.
        "adp_rank": [i + reach for i in range(1, n + 1)],
        "player_id": [f"{owner}{draft_id}{i}" for i in range(n)],
    })


def _corpus(frames):
    return pd.concat(frames, ignore_index=True)


def test_an_owner_with_less_history_is_pulled_further_toward_the_population():
    """One draft is not a habit. The same personal value must count for less
    when it rests on less."""
    veteran = [_picks("vet", f"d{i}", ["TE", "TE", "RB"]) for i in range(6)]
    rookie = [_picks("new", "dx", ["TE", "TE", "RB"])]
    # Others, so the population sits well away from two tight ends.
    others = [_picks(f"o{j}", f"e{i}", ["TE", "RB", "RB"])
              for j in range(4) for i in range(6)]
    prof = build(_corpus(veteran + rookie + others))

    te = prof[prof["trait"] == "shape_TE"].set_index("owner_key")
    assert te.loc["vet", "weight"] > te.loc["new", "weight"]
    # Both want 2; the newcomer's estimate lands nearer the population's 1.
    assert te.loc["vet", "value"] > te.loc["new", "value"]


def test_a_trait_that_does_not_separate_owners_is_shrunk_harder():
    """Two traits, same number of drafts. One where owners genuinely differ,
    one where they all do the same thing and only wobble. The second must end
    up trusting the population more."""
    frames = []
    rng = np.random.default_rng(0)
    for owner, te in (("a", 3), ("b", 1), ("c", 3), ("d", 1)):
        for i in range(6):
            # TE count differs sharply BETWEEN owners and not within.
            # QB count is the same for everyone and wobbles by one.
            qb = 1 + int(rng.integers(0, 2))
            frames.append(_picks(owner, f"{owner}{i}",
                                 ["TE"] * te + ["QB"] * qb + ["RB"]))
    prof = build(_corpus(frames))
    w = prof.groupby("trait")["weight"].mean()

    assert w["shape_TE"] > w["shape_QB"], \
        "the trait owners actually differ on should be trusted more"


def test_the_population_is_an_average_over_owners_not_over_drafts():
    """Otherwise whoever has run the most mock drafts silently becomes the
    baseline everyone else is measured against."""
    heavy = [_picks("heavy", f"h{i}", ["TE", "TE", "TE", "RB"]) for i in range(30)]
    light = [_picks(f"l{j}", f"l{j}", ["RB", "RB"]) for j in range(3)]
    prof = build(_corpus(heavy + light))

    pop = prof[prof["trait"] == "shape_TE"]["population"].iloc[0]
    # Owner-averaged: (3 + 0 + 0 + 0)/4 = 0.75. Draft-averaged would be ~2.7.
    assert pop == pytest.approx(0.75, abs=0.01)


def test_an_anonymous_owner_feeds_the_baseline_but_gets_no_profile():
    """Mock opponents are most of what a first-time owner is compared to.
    Dropping them would build the population out of the few people we happen
    to be able to name."""
    named = [_picks("me", f"m{i}", ["TE", "RB"]) for i in range(3)]
    strangers = [_picks(f"anon:x:{j}", f"m{i}", ["QB", "QB"], anon=True)
                 for j in range(4) for i in range(3)]
    prof = build(_corpus(named + strangers))

    assert set(prof["owner_key"]) == {"me"}, "an anonymous seat got a profile"
    qb = prof[prof["trait"] == "shape_QB"]["population"].iloc[0]
    assert qb > 0, "the anonymous picks never reached the baseline"


def test_a_single_owner_cannot_be_measured_against_himself():
    """With nobody to compare to there is no between-owner variance, so the
    trait is maximally shrunk rather than maximally trusted -- the degenerate
    case must fail safe."""
    frames = [_picks("only", f"d{i}", ["TE", "RB"]) for i in range(6)]
    prof = build(_corpus(frames))

    assert (prof["weight"] <= 6 / (6 + K_MAX) + 1e-9).all()


def test_reach_is_positive_when_a_player_is_taken_before_his_rank():
    """Sign convention, pinned: a manager who takes players ahead of the
    market reads POSITIVE, so the trait can be read without a lookup."""
    frames = [_picks("eager", f"d{i}", ["RB", "WR"], reach=10.0) for i in range(6)]
    frames += [_picks(f"o{j}", f"e{i}", ["RB", "WR"], reach=0.0)
               for j in range(3) for i in range(6)]
    prof = build(_corpus(frames))

    eager = prof[(prof["owner_key"] == "eager") & (prof["trait"] == "reach")]
    assert eager["personal"].iloc[0] == pytest.approx(10.0)


def test_owners_who_do_not_actually_differ_are_not_credited_with_a_habit():
    """The flattery this module exists to avoid.

    Here nobody has a tight-end preference at all -- every owner draws the
    same way, and the only reason their averages differ is that six drafts is
    a small sample. The spread of owner MEANS is nonetheless positive, purely
    from that noise, so dividing by it raw reports genuine separation where
    there is none and hands each owner most of the weight on a habit he does
    not have. Subtracting the sampling term first is what keeps the weight
    low. Measured: 0.38 with the correction, 0.62 without.
    """
    rng = np.random.default_rng(4)
    frames = []
    for owner in range(6):
        for draft in range(6):
            te = int(rng.integers(0, 5))
            frames.append(pd.DataFrame({
                "draft_id": f"{owner}-{draft}", "owner_key": f"o{owner}",
                "is_anonymous": False, "pick_no": range(1, te + 2),
                "round": range(1, te + 2), "position": ["TE"] * te + ["RB"],
                "adp_rank": range(1, te + 2),
                "player_id": [f"{owner}{draft}{k}" for k in range(te + 1)]}))
    prof = build(_corpus(frames))

    assert prof[prof["trait"] == "shape_TE"]["weight"].mean() < 0.45
