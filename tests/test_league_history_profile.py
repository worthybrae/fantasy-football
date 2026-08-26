"""The seam between the walk and the profile.

`pipeline/league_activity.py` stores what ESPN said and rebuilds the
tables; `scoring/manager_profile.py` reads those tables back. Every other
test in this suite exercises one side or the other -- the walk against
fake payloads, the facts against tables written by hand -- so a table the
walk never fills, or a shape it fills differently than the fixtures
assume, passes both and still leaves a section of the page empty. This
runs one small league end to end and reads the profile off it.
"""
import json

import duckdb

from pipeline import league_activity as act
from scoring import manager_profile as mp
from tests.test_league_activity import SWID_A, _fetch_json, _served


def _walked(tmp_path):
    """One season walked out of the activity fixtures. Week 3 is the week
    the roster payload carries stats for, so it is the week the lineup
    facts can be measured on; weeks 1 and 2 answer 404 and store empty."""
    conn = duckdb.connect(str(tmp_path / "league.duckdb"))
    act.import_activity(conn, "53929318",
                        _fetch_json(_served(2024, weeks=(3,), final=3), []),
                        seasons=[2024], current_season=2025)
    return conn


def test_the_walk_fills_every_section_the_profile_reads(tmp_path):
    conn = _walked(tmp_path)

    # The settings the optimal lineup is measured against: without them the
    # whole Lineups section reads n=0 however many weeks were stored.
    assert mp._slot_counts(conn, 2024)

    lineups = mp.lineups(conn, SWID_A)
    assert lineups["n"] == 1
    week = lineups["weeks"][0]
    # A's only scoring starter in week 3 is the WR (18.1); the D/ST has a
    # projection and no actual, so it is no part of either number.
    assert (week["season"], week["week"]) == (2024, 3)
    assert week["started"] == 18.1 and week["optimal"] == 18.1 and week["left"] == 0.0

    # One waiver claim in the transaction fixture, kept as ESPN keeps a
    # processed one: the outcome row alone, naming a claim row that is gone.
    waivers = mp.waivers(conn, SWID_A)
    assert waivers["claims"] == 1 and waivers["n"] == 1
    assert waivers["won"] == 1 and waivers["drops"] == 1
    assert "bids" not in waivers          # this season did not run on FAAB

    json.dumps(mp.profile(conn, SWID_A))
    json.dumps(mp.league_overview(conn))
