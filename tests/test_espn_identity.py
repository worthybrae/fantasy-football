"""Proving ownership of an ESPN account, with no ESPN and no network.

`pipeline/espn_identity.py` is the half of credential custody that stops a
stranger writing a row under somebody else's public SWID. It is one function
with a fetch hole in it, deliberately, so that every branch -- ESPN agreeing,
ESPN refusing, ESPN answering for the wrong person, ESPN not answering at all,
and ESPN turning out to authenticate nothing -- is testable without a
credential and without a socket.
"""
import json

import pytest

from pipeline import espn_identity as ident

SWID = "{7A1F9C34-BEEF-4D01-9A55-C0FFEE001122}"
OTHER = "{11112222-3333-4444-5555-666677778888}"
S2 = "AEBz" + "fixtureNotARealSession" * 3


def _fetch(authed=(200, None), anon=(401, "")):
    """A `fetch(url, cookies)` double that answers differently with and
    without cookies -- which is the distinction the whole module turns on.

    `authed`'s body defaults to a profile naming SWID, since that is the
    ordinary case; every test that cares passes its own.
    """
    calls = []

    def fetch(url, cookies):
        calls.append((url, cookies))
        status, body = authed if cookies else anon
        if body is None:
            body = json.dumps({"id": SWID, "dmaId": "506"})
        return status, body

    fetch.calls = calls
    return fetch


def test_a_session_espn_vouches_for_is_accepted():
    """The ordinary case: ESPN answers for this account with these cookies,
    and refuses the same request without them."""
    fetch = _fetch()
    assert ident.verify_account(SWID, S2, fetch=fetch) == SWID
    # The cookies really were sent, and both of them: `espn_s2` alone is not
    # what ESPN authenticates on.
    _url, cookies = fetch.calls[0]
    assert cookies == {"espn_s2": S2, "SWID": SWID}


def test_a_session_belonging_to_someone_else_is_refused():
    """The interesting failure. The cookie works -- it is a real, live ESPN
    session -- it just is not this account's. That is precisely the attacker
    in the takeover: their own session, somebody else's SWID."""
    fetch = _fetch(authed=(200, json.dumps({"id": OTHER})))
    with pytest.raises(ident.OwnershipUnproven):
        ident.verify_account(SWID, S2, fetch=fetch)


def test_a_session_espn_rejects_is_refused():
    fetch = _fetch(authed=(401, "{}"))
    with pytest.raises(ident.OwnershipUnproven) as caught:
        ident.verify_account(SWID, S2, fetch=fetch)
    assert "401" in str(caught.value)


def test_an_answer_naming_no_account_is_refused():
    """"No id" must never read as "verified"."""
    for body in ("{}", "[]", "not json at all", ""):
        fetch = _fetch(authed=(200, body))
        with pytest.raises(ident.OwnershipUnproven):
            ident.verify_account(SWID, S2, fetch=fetch)


def test_espn_being_unreachable_is_a_refusal_not_a_shrug():
    """Fails closed. This runs on a write path that stores a full account
    session for thirty days, so "we could not check, so we stored it" is not
    an available answer."""
    def fetch(url, cookies):
        raise OSError("connection reset")

    with pytest.raises(ident.OwnershipUnproven):
        ident.verify_account(SWID, S2, fetch=fetch)


def test_the_refusal_never_prints_the_account_or_the_session():
    """The url this fails on carries the SWID in its path, and an httpx error
    wraps the whole url in its message -- which is why the transport failure
    goes through `redact` before it becomes a refusal anyone can read."""
    class _Boom(Exception):
        def __init__(self):
            super().__init__(f"Client error for url '{ident.fan_url(SWID)}'")

    def fetch(url, cookies):
        raise _Boom()

    with pytest.raises(ident.OwnershipUnproven) as caught:
        ident.verify_account(SWID, S2, fetch=fetch)
    message = str(caught.value)
    assert SWID not in message
    assert "7A1F9C34" not in message
    assert S2 not in message


def test_an_endpoint_that_answers_without_a_session_proves_nothing():
    """THE CONTROL, and the reason there are two requests rather than one.

    "GET the profile with these cookies and confirm the id comes back" is
    worth something only if the same request FAILS without them -- the id in
    the response is otherwise just the id from the url, and an attacker's
    request would sail through. Rather than assume ESPN keeps that endpoint
    authenticated, this measures it on every verification and refuses if the
    assumption has stopped holding.
    """
    public = _fetch(anon=(200, json.dumps({"id": SWID})))
    with pytest.raises(ident.OwnershipUnproven) as caught:
        ident.verify_account(SWID, S2, fetch=public)
    assert "without a session" in str(caught.value)


def test_the_control_tolerates_an_anonymous_request_that_simply_fails():
    """Unreachable anonymously is evidence the endpoint is not open, which is
    what the control was looking for -- not a reason to refuse."""
    def fetch(url, cookies):
        if not cookies:
            raise OSError("refused")
        return 200, json.dumps({"id": SWID})

    assert ident.verify_account(SWID, S2, fetch=fetch) == SWID


def test_the_id_is_found_wherever_espn_chooses_to_put_it():
    """A read of somebody else's JSON. Strict that there IS an id, tolerant
    about where it sits."""
    for body in ({"id": SWID}, {"guid": SWID}, {"swid": SWID},
                 {"profile": {"id": SWID}}):
        fetch = _fetch(authed=(200, json.dumps(body)))
        assert ident.verify_account(SWID, S2, fetch=fetch) == SWID


def test_every_spelling_of_one_swid_is_one_account():
    """Five spellings reach this project from five places -- ESPN's cookie,
    its own urls, a pasted draft link, a percent-encoded query string, and
    whatever a user pasted with a trailing space."""
    body = SWID.strip("{}")
    for spelling in (SWID, body, body.lower(), f"%7B{body}%7D", f" {SWID} ",
                     "{" + body.lower() + "}"):
        assert ident.canonical_swid(spelling) == SWID


def test_a_swid_that_is_not_guid_shaped_is_left_alone():
    """ESPN changing its id format should degrade to "one spelling, exactly as
    sent", never to a mangled key that silently splits one account in two."""
    assert ident.canonical_swid("  some-other-id  ") == "some-other-id"
    assert ident.canonical_swid(None) == ""


def test_no_account_named_is_refused_before_anything_is_fetched():
    def explode(url, cookies):
        raise AssertionError("should not have been called")

    for swid, s2 in (("", S2), (None, S2), (SWID, ""), (SWID, None)):
        with pytest.raises(ident.OwnershipUnproven):
            ident.verify_account(swid, s2, fetch=explode)
