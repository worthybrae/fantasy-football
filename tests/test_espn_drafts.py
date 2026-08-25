import json
from datetime import datetime, timezone

import pytest

from pipeline.espn_drafts import (SessionExpired, cookies_for, draft_token_url,
                                  league_draft, league_entries, league_ids,
                                  mint_draft_token, upcoming_drafts)


def _fetcher(routes):
    """A fake `fetch(url, cookies, headers)` answering by url substring.

    Records what it was called with, because two of the properties worth
    pinning here are about the REQUEST -- the cookies that went out and the
    header ESPN refuses the call without -- not about the response.
    """
    calls = []

    def fetch(url, cookies, headers=None):
        calls.append({"url": url, "cookies": cookies, "headers": headers})
        for key, answer in routes.items():
            if key in url:
                return answer
        return 404, ""
    fetch.calls = calls
    return fetch


def _entry(league_id, *, team_id=4, name=None, season=2026, date_ms=1_800_000_000_000,
           status=1, complete=False, size=12, game=1):
    """One fantasy entry, shaped like a real fan profile's.

    Every field here was read off an actual account (see
    `pipeline/espn_drafts.league_entries`): the team id is `entryId` on the
    entry, and the league's name, size, draft date and draft state hang off
    the group beside it.
    """
    return {"typeId": 9, "metaData": {"entry": {
        "gameId": game, "gameAbbrev": "ffl", "seasonId": season,
        "entryId": team_id,
        "entryMetadata": {"teamName": f"Team {team_id}",
                          "draftComplete": complete},
        "groups": [{"groupId": league_id, "groupSize": size,
                    "groupName": name or f"League {league_id}",
                    "draftDate": date_ms, "draftStatus": status,
                    "draftTypeName": "Snake"}],
    }}}


def _profile(*entries):
    return json.dumps({"id": "{ABC}", "preferences": list(entries)})


def _settings(name="Dynasty", size=12, date_ms=1_800_000_000_000, drafted=False):
    return json.dumps({"settings": {
        "name": name, "size": size,
        "draftSettings": {"type": "SNAKE", "date": date_ms, "drafted": drafted},
    }})


# -- the token -----------------------------------------------------------------

def test_mint_sends_the_cookie_and_the_header_espn_demands():
    """The whole reason this can run server-side is that `draftSecurity` wants
    a cookie, not a browser. And it 403s without `x-fantasy-source`, which is
    the kind of thing that is invisible until it is a production outage."""
    fetch = _fetcher({"draftSecurity": (200, " 8675309 \n")})
    token = mint_draft_token("123", "4", 2026,
                             cookies_for("{ABC}", "s2value"), fetch=fetch)
    assert token == "8675309"
    sent = fetch.calls[0]
    assert sent["cookies"] == {"SWID": "{ABC}", "espn_s2": "s2value"}
    assert sent["headers"]["x-fantasy-source"] == "kona"
    assert sent["url"] == draft_token_url("123", "4", 2026)


@pytest.mark.parametrize("status", [401, 403])
def test_mint_raises_session_expired_so_the_caller_deletes_rather_than_retries(status):
    """A credential ESPN has just rejected is not going to start working, and
    the user cannot be told. The type is the signal to drop the row."""
    fetch = _fetcher({"draftSecurity": (status, "")})
    with pytest.raises(SessionExpired):
        mint_draft_token("123", "4", 2026, cookies_for("{A}", "s2"), fetch=fetch)


def test_mint_refuses_a_body_that_is_not_a_token():
    """An HTML error page is a 200 with words in it. Left unchecked it would
    be handed to the socket as a token and fail there, pointing at the wrong
    thing entirely."""
    fetch = _fetcher({"draftSecurity": (200, "<html>Sign in</html>")})
    with pytest.raises(RuntimeError, match="did not return a draft token"):
        mint_draft_token("123", "4", 2026, cookies_for("{A}", "s2"), fetch=fetch)


def test_mint_accepts_a_negative_token():
    """MEASURED, NOT ASSUMED. A real ESPN mock draft answered `-1872384467`,
    and the first version of the check here was `isdigit()` -- which rejected
    a working token as "the draft is not open yet" and would have made the
    join button fail on exactly the drafts it exists for."""
    fetch = _fetcher({"draftSecurity": (200, "-1872384467\n")})
    assert mint_draft_token("123", "4", 2026, cookies_for("{A}", "s2"),
                            fetch=fetch) == "-1872384467"


# -- the league list -----------------------------------------------------------

def test_league_entries_carry_the_team_id_nothing_else_can_supply():
    """The join button needs a TEAM, not just a league, and `entryId` is the
    only place this account's team in that league is named. Everything else on
    the row -- the league's name and size, the draft's date and state -- comes
    out of the same one call, which is why the list is not a dozen."""
    fetch = _fetcher({"fan.api": (200, _profile(_entry("111", team_id=6,
                                                       name="Home")))})
    rows = league_entries("{ABC}", cookies_for("{ABC}", "s2"), 2026, fetch=fetch)
    assert len(rows) == 1
    row = rows[0]
    assert row["league_id"] == "111" and row["team_id"] == "6"
    assert row["name"] == "Home" and row["teams"] == 12
    assert row["team_name"] == "Team 6" and row["draft_type"] == "Snake"
    assert row["draft_at"] == datetime.fromtimestamp(1_800_000_000, tz=timezone.utc)


def test_league_entries_ignore_other_games_and_other_seasons():
    """One profile carries every season this account has played and every
    sport it plays. A 2019 league is not a draft anybody is waiting for."""
    payload = _profile(
        _entry("999", game=2),                 # basketball
        _entry("888", season=2019),            # last decade
        _entry("111"),
    )
    fetch = _fetcher({"fan.api": (200, payload)})
    rows = league_entries("{A}", cookies_for("{A}", "s2"), 2026, fetch=fetch)
    assert [r["league_id"] for r in rows] == ["111"]


def test_league_ids_is_empty_rather_than_raising_on_a_shape_it_cannot_read():
    """ESPN owns this shape and has moved it before. A background scan must
    not take a page down over somebody else's JSON."""
    fetch = _fetcher({"fan.api": (200, json.dumps({"something": "else"}))})
    assert league_ids("{A}", cookies_for("{A}", "s2"), fetch=fetch) == []


def test_league_ids_still_raises_on_an_expired_session():
    """Distinct from the empty case above: "we could not read your leagues" is
    a shrug, and "your session is dead" is a row to delete."""
    fetch = _fetcher({"fan.api": (401, "")})
    with pytest.raises(SessionExpired):
        league_ids("{A}", cookies_for("{A}", "s2"), fetch=fetch)


# -- when they draft -----------------------------------------------------------

def test_league_draft_reads_the_settings_view_this_project_already_fetches():
    fetch = _fetcher({"leagues/111": (200, _settings(date_ms=1_800_000_000_000))})
    row = league_draft("111", 2026, cookies_for("{A}", "s2"), fetch=fetch)
    assert row["name"] == "Dynasty" and row["teams"] == 12
    assert row["draft_type"] == "SNAKE" and row["drafted"] is False
    assert row["draft_at"] == datetime.fromtimestamp(1_800_000_000, tz=timezone.utc)
    assert "view=mSettings" in fetch.calls[0]["url"]


def test_league_draft_has_no_date_rather_than_a_fake_one():
    """A league whose commissioner has not scheduled the draft carries a 0.
    Printed through the epoch that would read as 1 January 1970."""
    fetch = _fetcher({"leagues/111": (200, _settings(date_ms=0))})
    assert league_draft("111", 2026, cookies_for("{A}", "s2"),
                        fetch=fetch)["draft_at"] is None


# -- the list a user actually sees ---------------------------------------------

def test_upcoming_drops_finished_drafts_and_sorts_soonest_first():
    soon = 1_800_000_000_000
    later = soon + 86_400_000
    fetch = _fetcher({"fan.api": (200, _profile(
        _entry("111", name="Later", date_ms=later),
        _entry("222", name="Done", date_ms=soon, complete=True),
        _entry("333", name="Soon", date_ms=soon),
    ))})
    rows = upcoming_drafts("{A}", cookies_for("{A}", "s2"), 2026, fetch=fetch,
                           now=datetime(2020, 1, 1, tzinfo=timezone.utc))
    assert [r["name"] for r in rows] == ["Soon", "Later"]


def test_upcoming_drops_a_draft_espn_marks_finished_by_status_alone():
    """Two sources for one fact, and either is enough: a profile mid-migration
    has been seen to carry the group's status without the entry's flag."""
    fetch = _fetcher({"fan.api": (200, _profile(
        _entry("111", name="Done", status=2),
        _entry("222", name="Open"),
    ))})
    rows = upcoming_drafts("{A}", cookies_for("{A}", "s2"), 2026, fetch=fetch,
                           now=datetime(2020, 1, 1, tzinfo=timezone.utc))
    assert [r["name"] for r in rows] == ["Open"]


def test_upcoming_keeps_an_unscheduled_league_and_puts_it_last():
    """"In a league, no date set" is a true and useful thing to show a drafter
    in August -- it is the league he needs to go and check."""
    fetch = _fetcher({"fan.api": (200, _profile(
        _entry("111", name="No date", date_ms=None),
        _entry("222", name="Dated"),
    ))})
    rows = upcoming_drafts("{A}", cookies_for("{A}", "s2"), 2026, fetch=fetch,
                           now=datetime(2020, 1, 1, tzinfo=timezone.utc))
    assert [r["name"] for r in rows] == ["Dated", "No date"]


def test_a_room_already_running_is_kept_and_flagged_rather_than_hidden():
    """The draft with a clock running is the one a user most needs to reach.
    Both routes to the flag are pinned: ESPN's own in-progress status, and a
    scheduled time that has simply passed."""
    fetch = _fetcher({"fan.api": (200, _profile(
        _entry("111", name="ESPN says live", status=3),
        _entry("222", name="Hour has passed"),
    ))})
    rows = upcoming_drafts("{A}", cookies_for("{A}", "s2"), 2026, fetch=fetch,
                           now=datetime(2030, 1, 1, tzinfo=timezone.utc))
    assert all(r["live"] for r in rows)


def test_upcoming_propagates_an_expired_session():
    """Every league would fail for the same reason, and the caller has a row
    to delete. Swallowing this would show an empty draft list to a user whose
    real problem is that they are signed out."""
    fetch = _fetcher({"fan.api": (403, "")})
    with pytest.raises(SessionExpired):
        upcoming_drafts("{A}", cookies_for("{A}", "s2"), 2026, fetch=fetch)
