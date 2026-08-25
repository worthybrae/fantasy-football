"""The HTTP edge of credential custody, over a real TestClient.

`tests/test_credentials.py` proves the store's properties. These prove the
ones that only exist once a browser is involved: that the cookie carries the
flags it has to carry, that a request without it gets nothing no matter what
else it presents, and that a plaintext connection is refused before it can
carry an account session.

The account-takeover test is the one to read first. SWID is public -- it is in
the invite POST's query string and the draft socket's JOIN url -- so an
endpoint that answers to it is an endpoint that hands strangers' ESPN accounts
to anyone who has seen a draft link. `test_a_valid_swid_with_no_cookie_gets_
nothing_back` presents a real, stored user's real SWID with no cookie at all.

Entirely offline: no ESPN call, no socket, no real credential.
"""
import pytest
from fastapi import Body, FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from api.custody import establish_custody, register_custody_routes, require_secure
from pipeline import credentials as cred
from pipeline.espn_identity import OwnershipUnproven

FAKE_SWID = "{7A1F9C34-BEEF-4D01-9A55-C0FFEE001122}"
FAKE_S2 = "AEBz" + "QqcustodyFIXTUREnotarealsessionvalue" * 3 + "%2Fend"
OTHER_SWID = "{11112222-3333-4444-5555-666677778888}"
OTHER_S2 = "AEBz" + "OTHERcustodyFIXTUREvalue" * 3

# Which account each fixture session really belongs to -- the answer the real
# `verify_account` gets from ESPN. Keyed on the SESSION and never on the
# claimed swid, because that is the whole distinction: a session identifies its
# own owner, and a client's claim about it is not evidence.
_ACCOUNTS = {FAKE_S2: FAKE_SWID, OTHER_S2: OTHER_SWID}


def _verifier(swid, espn_s2):
    owner = _ACCOUNTS.get(str(espn_s2))
    if owner is None:
        raise OwnershipUnproven("ESPN did not accept that session")
    return owner


@pytest.fixture
def custody_env(tmp_path, monkeypatch):
    """A configured-from-the-environment store, the way production gets one.

    Deliberately goes through the environment rather than passing a store
    object into the routes: the env plumbing (`ESPN_CUSTODY_KEYS`,
    `ESPN_CUSTODY_DB_PATH`) is itself part of what has to work, and a
    deployment that reads its key from somewhere else is the one failure mode
    that would make every other test in this file pass while the real server
    stored nothing.
    """
    monkeypatch.setenv(cred.KEYS_ENV, f"1:{cred.generate_key()}")
    monkeypatch.setenv(cred.DB_PATH_ENV, str(tmp_path / "custody.duckdb"))
    monkeypatch.delenv(cred.ALLOW_PLAINTEXT_ENV, raising=False)
    monkeypatch.delenv(cred.TRUST_FORWARDED_PROTO_ENV, raising=False)
    # The ownership proof, stubbed at the name the store resolves when it is
    # CONSTRUCTED -- which is why this is set before `default_store()` below.
    # Nothing in this file touches the network, and a test that accidentally
    # did would hang rather than quietly verify against the real ESPN.
    monkeypatch.setattr(cred, "verify_account", _verifier)
    cred.reset_default_store()
    yield cred.default_store()
    cred.reset_default_store()
    cred.close_all()


def _app():
    """The custody routes, plus a stand-in for the endpoint that mints.

    `/_connect` is a harness, not a product endpoint: it is the two lines
    `/api/live/connect-token` runs (the same guard, the same
    `establish_custody`) with none of the 17-second session build in front of
    them. Written out here rather than importing the real handler because what
    is under test is the custody wiring, and the real endpoint's own copy of
    those two lines is covered separately below.
    """
    app = FastAPI()
    # The harness's own mint path joins the guarded set, so the middleware
    # inspects its body exactly as it does the real endpoint's.
    register_custody_routes(app, paths={"/_connect", "/api/live/connect-token"})

    @app.post("/_connect")
    def _connect(request: Request, response: Response, body: dict = Body(...)):
        if body.get("espn_s2") or request.cookies.get(cred.COOKIE_NAME):
            require_secure(request)
        establish_custody(request, response, body["swid"], body["espn_s2"])
        return {"ok": True}

    return app


def _client(secure=True):
    """TestClient over https by default, because that is what production is.

    The base url is what `request.url.scheme` reads, so this is the only lever
    these tests need to exercise the transport rule.
    """
    return TestClient(_app(),
                      base_url="https://testserver" if secure
                      else "http://testserver")


def _connected(client):
    client.post("/_connect", json={"swid": FAKE_SWID, "espn_s2": FAKE_S2})
    return client


# --- The account-takeover case -----------------------------------------------

def test_a_valid_swid_with_no_cookie_gets_nothing_back(custody_env):
    """THE ENDPOINT THIS DESIGN EXISTS TO NOT BE.

    A real user, really stored, whose SWID an attacker really has -- because
    SWIDs are public. Presented every way a query string can present it, with
    no cookie. The answer is "not connected", every time, and the response
    body carries nothing of theirs.
    """
    _connected(_client())                     # the victim connects, elsewhere
    assert custody_env.counts()["credentials"] == 1

    attacker = TestClient(_app(), base_url="https://testserver")
    for params in ({"swid": FAKE_SWID},
                   {"SWID": FAKE_SWID},
                   {"swid": FAKE_SWID.strip("{}")},
                   {"memberId": FAKE_SWID},
                   {"espn_s2": FAKE_S2}):
        resp = attacker.get("/api/espn/custody", params=params)
        assert resp.status_code == 200
        assert resp.json() == {"connected": False}
        assert FAKE_SWID not in resp.text and FAKE_S2 not in resp.text

    # And it cannot be used to log the victim out either.
    assert attacker.post("/api/espn/custody/disconnect",
                         params={"swid": FAKE_SWID}).status_code == 401
    assert attacker.post("/api/espn/custody/disconnect-everywhere",
                         params={"swid": FAKE_SWID}).status_code == 401
    assert custody_env.counts()["credentials"] == 1


# --- Cookies: missing, wrong, dead -------------------------------------------

def test_no_cookie_is_refused(custody_env):
    """The status probe answers 200 `connected: false` -- it is what the
    connect screen calls on load, and a 401 for every first-time visitor would
    be an error in every console. The two DELETIONS answer 401, because a
    request to delete something you have not proved you own is a refusal."""
    client = TestClient(_app(), base_url="https://testserver")
    assert client.get("/api/espn/custody").json() == {"connected": False}
    assert client.post("/api/espn/custody/disconnect").status_code == 401
    assert client.post(
        "/api/espn/custody/disconnect-everywhere").status_code == 401


def test_a_well_formed_but_wrong_cookie_is_refused(custody_env):
    """Right length, right alphabet, not ours. It must not be distinguishable
    from a cookie that never existed."""
    import secrets

    _connected(_client())
    forged = TestClient(_app(), base_url="https://testserver",
                        cookies={cred.COOKIE_NAME: secrets.token_urlsafe(32)})
    assert forged.get("/api/espn/custody").json() == {"connected": False}
    assert forged.post("/api/espn/custody/disconnect").status_code == 401
    assert custody_env.counts()["credentials"] == 1


def test_a_cookie_for_a_deleted_session_is_refused(custody_env):
    """The browser keeps its cookie after the row is gone -- from its own
    disconnect, from another browser's disconnect-everywhere, or from the
    reaper. Every one of those must land as "not connected"."""
    client = _connected(_client())
    assert client.get("/api/espn/custody").json()["connected"] is True

    stolen = dict(client.cookies)
    assert client.post("/api/espn/custody/disconnect").status_code == 200

    replayed = TestClient(_app(), base_url="https://testserver", cookies=stolen)
    assert replayed.get("/api/espn/custody").json() == {"connected": False}
    assert replayed.post("/api/espn/custody/disconnect").status_code == 401


# --- The cookie itself -------------------------------------------------------

def test_the_minted_cookie_is_httponly_secure_and_lax(custody_env):
    """Each flag is load-bearing.

    `HttpOnly`: the value is the password to a stored ESPN account session, so
    one XSS anywhere on the site would otherwise hand over the account rather
    than the page. `Secure`: it must never ride a plaintext connection.
    `SameSite=Lax`: it should not be attached to third-party subrequests.
    """
    client = _client()
    resp = client.post("/_connect",
                       json={"swid": FAKE_SWID, "espn_s2": FAKE_S2})
    assert resp.status_code == 200
    header = resp.headers["set-cookie"]
    assert header.startswith(f"{cred.COOKIE_NAME}=")
    lowered = header.lower()
    assert "httponly" in lowered
    assert "secure" in lowered
    assert "samesite=lax" in lowered
    assert "path=/" in lowered
    # And the value in the header is not in the database (see the store's own
    # test); what matters here is that it is not echoed into the body either.
    assert FAKE_S2 not in resp.text and FAKE_SWID not in resp.text


def test_the_response_never_echoes_the_credential(custody_env):
    """Not in the body, not in a header other than the cookie, not in an
    error. An echoed espn_s2 is one that reaches the browser's history, every
    intermediary's access log, and any error reporter the page installs."""
    client = _client()
    resp = client.post("/_connect",
                       json={"swid": FAKE_SWID, "espn_s2": FAKE_S2})
    blob = resp.text + "\n".join(f"{k}: {v}" for k, v in resp.headers.items()
                                 if k.lower() != "set-cookie")
    assert FAKE_S2 not in blob


# --- Transport ---------------------------------------------------------------

def test_plain_http_is_refused_unless_the_dev_opt_out_is_set(custody_env,
                                                             monkeypatch):
    """Fails closed. The dev opt-out is its own switch and nothing else
    reaches it -- see `pipeline/credentials.ALLOW_PLAINTEXT_ENV` for why that
    separation is the point rather than an inconvenience."""
    plain = _client(secure=False)
    resp = plain.post("/_connect",
                      json={"swid": FAKE_SWID, "espn_s2": FAKE_S2})
    assert resp.status_code == 400
    assert cred.ALLOW_PLAINTEXT_ENV in resp.json()["detail"]
    assert custody_env.counts()["credentials"] == 0, \
        "a refused request stored the credential anyway"

    monkeypatch.setenv(cred.ALLOW_PLAINTEXT_ENV, "1")
    allowed = _client(secure=False)
    ok = allowed.post("/_connect",
                      json={"swid": FAKE_SWID, "espn_s2": FAKE_S2})
    assert ok.status_code == 200
    assert custody_env.counts()["credentials"] == 1
    # ...and in that mode the cookie is NOT marked Secure, because a Secure
    # cookie on http is silently dropped by the browser -- the user would
    # appear never to have connected, on the one path that exists so local
    # development works at all. HttpOnly stays either way: it costs nothing
    # and there is no dev reason to let page script read the value.
    header = ok.headers["set-cookie"].lower()
    assert "secure" not in header
    assert "httponly" in header


def test_a_cookie_on_a_plaintext_connection_is_refused_too(custody_env,
                                                           monkeypatch):
    """The guard follows the credential, not the route. A request that
    presents the custody cookie is carrying a credential whether or not it is
    also sending one."""
    monkeypatch.setenv(cred.ALLOW_PLAINTEXT_ENV, "1")
    plain = _connected(_client(secure=False))
    cookies = dict(plain.cookies)
    monkeypatch.delenv(cred.ALLOW_PLAINTEXT_ENV)

    strict = TestClient(_app(), base_url="http://testserver", cookies=cookies)
    assert strict.get("/api/espn/custody").status_code == 400
    assert strict.post("/api/espn/custody/disconnect").status_code == 400


def test_a_request_carrying_nothing_is_not_gated(custody_env):
    """A visitor who has never connected anything, probing the status endpoint
    over plain http, is carrying no credential and gets a plain answer. The
    gate exists to protect a credential, so with none in the request there is
    nothing for it to do."""
    plain = TestClient(_app(), base_url="http://testserver")
    assert plain.get("/api/espn/custody").json() == {"connected": False}


# --- The two ways out --------------------------------------------------------

def test_disconnect_ends_this_browser_and_leaves_the_others(custody_env):
    """One browser's logout is not everyone's. The other browser is frequently
    the one with the draft open."""
    first = _connected(_client())
    second = _connected(_client())
    assert custody_env.counts() == {"credentials": 1, "sessions": 2}

    assert first.post("/api/espn/custody/disconnect").status_code == 200
    assert second.get("/api/espn/custody").json()["connected"] is True
    assert custody_env.counts() == {"credentials": 1, "sessions": 1}


def test_disconnect_everywhere_cascades_to_every_browser(custody_env):
    """The whole of what we can offer a user who wants out: our copy is gone.
    ESPN has no per-application revocation, so their session at ESPN is
    untouched -- which is exactly why the disclosure copy has to say so."""
    first = _connected(_client())
    second = _connected(_client())

    resp = first.post("/api/espn/custody/disconnect-everywhere")
    assert resp.status_code == 200
    assert resp.json()["everywhere"] is True
    assert second.get("/api/espn/custody").json() == {"connected": False}
    assert custody_env.counts() == {"credentials": 0, "sessions": 0}


def test_a_refused_disconnect_still_clears_the_dead_cookie(custody_env):
    """A cookie that names no session is worth nothing to us and is one more
    credential-shaped value the browser keeps sending. It goes, even on the
    401 -- which is why the handler builds its own response rather than
    raising."""
    import secrets

    forged = TestClient(_app(), base_url="https://testserver",
                        cookies={cred.COOKIE_NAME: secrets.token_urlsafe(32)})
    resp = forged.post("/api/espn/custody/disconnect")
    assert resp.status_code == 401
    assert cred.COOKIE_NAME in resp.headers.get("set-cookie", "")
    assert 'max-age=0' in resp.headers["set-cookie"].lower() or \
           '=""' in resp.headers["set-cookie"]


def test_the_status_probe_names_the_account_without_naming_it(custody_env):
    """"Connected as ...0022" is enough for a user with two ESPN accounts to
    tell which one this browser holds. The full SWID stays out of the response
    body, where it would reach every intermediary's access log."""
    client = _connected(_client())
    body = client.get("/api/espn/custody").json()
    assert body["connected"] is True
    assert body["account_hint"] == "1122"
    assert FAKE_SWID not in str(body)


# --- The real endpoint's own copy of the guard -------------------------------

def _live_app(tmp_path):
    """`/api/live/connect-token` mounted on an empty database.

    Empty is enough: the transport guard is the FIRST statement in the
    handler, before the progress row and before anything that would need a
    seeded board, so a refusal here is a refusal that happened before any work
    -- which is the property being asserted.
    """
    from api.live import register_live_routes
    from pipeline.db import get_conn

    path = str(tmp_path / "live.duckdb")
    app = FastAPI()
    register_live_routes(app, get_conn(path), path)
    # `create_app` mounts these on the same app as the live routes, and they
    # are what protects `/api/live/connect-token` -- so a harness without them
    # would be testing a configuration that does not ship.
    register_custody_routes(app)
    return app


def test_connect_token_refuses_plain_http_when_it_carries_a_session(
        custody_env, tmp_path):
    """The bookmarklet's own endpoint, over http, with an account session in
    the body. Refused before it does anything at all."""
    client = TestClient(_live_app(tmp_path), base_url="http://testserver")
    resp = client.post("/api/live/connect-token", json={
        "leagueId": "1", "teamId": "2", "swid": FAKE_SWID,
        "token": "1953383334", "season": "2026", "espn_s2": FAKE_S2})
    assert resp.status_code == 400
    assert cred.ALLOW_PLAINTEXT_ENV in resp.json()["detail"]
    assert custody_env.counts()["credentials"] == 0


def test_connect_token_without_an_account_session_is_left_alone(
        custody_env, tmp_path):
    """The pre-existing single-user path: a draft nonce, no espn_s2, no
    cookie, http://localhost. It must reach its own validation exactly as
    before -- a 422 for the missing token, not a 400 for the transport --
    because refusing it would break the owner's own machine to protect a
    credential that is not in the request.
    """
    client = TestClient(_live_app(tmp_path), base_url="http://testserver")
    resp = client.post("/api/live/connect-token", json={
        "leagueId": "1", "teamId": "2", "swid": FAKE_SWID,
        "token": "", "season": "2026"})
    assert resp.status_code == 422, resp.text
    assert "missing" in resp.json()["detail"]
    assert custody_env.counts() == {"credentials": 0, "sessions": 0}


# --- A malformed request must not become a disclosure ------------------------
#
# FastAPI validates the body BEFORE the handler runs, so a check written as the
# handler's first statement never sees a request that fails validation -- and a
# body that fails validation is still a body that was transmitted. FastAPI's
# own 422 then echoes the rejected input straight back. The two findings
# compound: omit one field over plain http and the account session went out in
# the clear AND came back in the response.

def test_a_malformed_body_never_echoes_the_session(custody_env, tmp_path):
    """Over TLS, where the transport guard has nothing to say, the 422 itself
    is the leak. Pydantic attaches the offending value to every error as
    `input`, and on a missing-field error `input` is the ENTIRE body -- so the
    espn_s2 comes back to the caller, into the browser's network tab, and into
    every proxy's access log between here and there."""
    client = TestClient(_live_app(tmp_path), base_url="https://testserver")
    for body in (
            # a missing field
            {"leagueId": "1", "teamId": "2", "swid": FAKE_SWID,
             "token": "t", "espn_s2": FAKE_S2},
            # a wrongly typed one
            {"leagueId": "1", "teamId": "2", "swid": FAKE_SWID, "token": 7,
             "season": [], "espn_s2": FAKE_S2},
            # a body that is not the shape at all
            {"espn_s2": FAKE_S2},
    ):
        resp = client.post("/api/live/connect-token", json=body)
        assert resp.status_code == 422, resp.text
        assert FAKE_S2 not in resp.text, "the 422 echoed the account session"
        assert "input" not in resp.text
        # Still useful: it says which field and why.
        assert "loc" in resp.text and "msg" in resp.text


def test_a_malformed_body_over_plain_http_is_refused_before_validation(
        custody_env, tmp_path):
    """The transport refusal has to happen at a layer validation cannot get in
    front of. Every one of these bodies fails validation, and every one of
    them carries the credential."""
    client = TestClient(_live_app(tmp_path), base_url="http://testserver")
    for body in ({"leagueId": "1", "espn_s2": FAKE_S2},
                 {"espn_s2": FAKE_S2},
                 {"espn_s2": {"nested": FAKE_S2}},
                 [{"espn_s2": FAKE_S2}]):
        resp = client.post("/api/live/connect-token", json=body)
        assert resp.status_code == 400, (body, resp.text)
        assert cred.ALLOW_PLAINTEXT_ENV in resp.json()["detail"]
        assert FAKE_S2 not in resp.text

    # Not even as raw bytes that are not JSON at all: the guard searches for
    # the field NAME, so nothing about the body's shape can smuggle it past.
    raw = client.post("/api/live/connect-token",
                      content=f'espn_s2={FAKE_S2}'.encode(),
                      headers={"content-type": "text/plain"})
    assert raw.status_code == 400
    assert FAKE_S2 not in raw.text
    assert custody_env.counts() == {"credentials": 0, "sessions": 0}


def test_the_guard_lets_an_ordinary_request_through_with_its_body_intact(
        custody_env, tmp_path):
    """The middleware reads the body to decide, so it has to hand the SAME
    body to the route afterwards. If it did not, every guarded request would
    hang or arrive empty -- which would look like a bug in the endpoint rather
    than in the guard."""
    client = TestClient(_live_app(tmp_path), base_url="http://testserver")
    resp = client.post("/api/live/connect-token", json={
        "leagueId": "1", "teamId": "2", "swid": FAKE_SWID,
        "token": "", "season": "2026"})
    # Reached the handler and hit its OWN validation, which means the body
    # survived the middleware.
    assert resp.status_code == 422
    assert "missing" in resp.json()["detail"]


# --- The takeover, at the HTTP layer -----------------------------------------

def test_a_stranger_cannot_take_over_an_account_through_the_api(custody_env):
    """The same attack as the store test, driven the way it would really
    arrive: a POST naming the victim's public SWID and carrying the attacker's
    own ESPN session.

    It fails for a different reason than it used to. The row is keyed on
    HMAC(espn_s2) now, so naming somebody else's account does not point the
    write at their row -- there is no name in the key to aim at. See
    `CredentialStore.connect` for why the previous defence (making ESPN name
    the owner) had to go: the endpoint it asked turned out to be public.
    """
    victim = _connected(_client())
    assert victim.get("/api/espn/custody").json()["account_hint"] == "1122"

    attacker = TestClient(_app(), base_url="https://testserver")
    resp = attacker.post("/_connect",
                         json={"swid": FAKE_SWID, "espn_s2": OTHER_S2})
    assert resp.status_code == 200          # they stored THEIR OWN session

    # The victim's browser is untouched and still resolves to their session.
    still = victim.get("/api/espn/custody").json()
    assert still["connected"] is True
    assert still["account_hint"] == "1122"
    # Two rows, and the attacker's cookie reaches only their own.
    assert custody_env.counts()["credentials"] == 2
    assert attacker.get("/api/espn/custody").json()["connected"] is True


def test_the_account_hint_is_what_was_claimed_not_a_verified_identity(custody_env):
    """A caller can mislabel THEIR OWN row, and that is all.

    Worth pinning because the hint reads like an identity and no longer is:
    nothing verifies the SWID, so a connect that names somebody else's account
    stores that name against its own session. The consequences stop there --
    the row is still keyed by the secret, still only reachable with the cookie
    minted for it, and the leagues it can actually read are ESPN's answer to
    the SESSION, not to the label.
    """
    attacker = TestClient(_app(), base_url="https://testserver")
    attacker.post("/_connect", json={"swid": FAKE_SWID, "espn_s2": OTHER_S2})
    # Their own row wears the victim's name...
    assert attacker.get("/api/espn/custody").json()["account_hint"] == "1122"
    # ...and the victim, who has never connected, is unaffected: one row.
    assert custody_env.counts()["credentials"] == 1


def test_half_a_session_is_refused_at_the_api(custody_env):
    """The refusal that is left. An empty secret would hash to a shared row id
    and an empty swid cannot be sent to ESPN, so neither is stored -- and no
    cookie is handed out for a credential that was never written."""
    client = TestClient(_app(), base_url="https://testserver")
    for body in ({"swid": FAKE_SWID, "espn_s2": ""}, {"swid": "", "espn_s2": OTHER_S2}):
        resp = client.post("/_connect", json=body)
        assert resp.status_code >= 400
        assert "set-cookie" not in resp.headers
    assert custody_env.counts() == {"credentials": 0, "sessions": 0}


# --- A failed request must not strand a credential ---------------------------

def test_a_request_that_fails_after_connecting_strands_nothing(custody_env):
    """THE UNREACHABLE ROW. `establish_custody` writes the credential and puts
    the cookie on the response object -- but the response is only sent if the
    handler returns. Anything raising afterwards leaves a live espn_s2 stored
    with NO browser holding the cookie, and every way out of the store needs
    that cookie. Nobody, including the person whose session it is, could
    delete it; it would survive the full thirty-day TTL.
    """
    from api.custody import abandon_custody

    app = FastAPI()
    register_custody_routes(app)

    @app.post("/_connect_then_fail")
    def _boom(request: Request, response: Response, body: dict = Body(...)):
        minted = establish_custody(request, response, body["swid"],
                                   body["espn_s2"])
        try:
            raise RuntimeError("the listener would not start")
        except BaseException:
            abandon_custody(minted)
            raise

    client = TestClient(app, base_url="https://testserver",
                        raise_server_exceptions=False)
    resp = client.post("/_connect_then_fail",
                       json={"swid": FAKE_SWID, "espn_s2": FAKE_S2})
    assert resp.status_code == 500
    assert custody_env.counts() == {"credentials": 0, "sessions": 0}, \
        "a failed request left a credential nobody can delete"


def test_the_cookie_still_reaches_a_handler_that_returns_a_response_object(
        custody_env):
    """FastAPI merges the injected `response` object's headers into whatever
    the handler RETURNS -- unless the handler returns a Response of its own,
    which is used as-is. That silently drops the cookie and produces exactly
    the stranded row above, so the connect path sets it on the returned object
    directly."""
    from api.custody import set_session_cookie

    app = FastAPI()
    register_custody_routes(app)

    @app.post("/_connect_returning_a_response")
    def _explicit(request: Request, response: Response, body: dict = Body(...)):
        minted = establish_custody(request, response, body["swid"],
                                   body["espn_s2"])
        out = JSONResponse(content={"connected": True})
        set_session_cookie(out, minted)
        return out

    client = TestClient(app, base_url="https://testserver")
    resp = client.post("/_connect_returning_a_response",
                       json={"swid": FAKE_SWID, "espn_s2": FAKE_S2})
    assert resp.status_code == 200
    assert cred.COOKIE_NAME in resp.headers.get("set-cookie", "")
    assert client.get("/api/espn/custody").json()["connected"] is True


# --- The store being unavailable is a 503, never a 500 -----------------------

def test_a_locked_store_is_a_clean_503(custody_env, monkeypatch):
    """DuckDB allows one writer per file, so under `uvicorn --workers 2` every
    worker but one raises IOException -- AFTER the credential has been read
    off the wire. Unhandled, that is a 500 with a traceback on a request
    carrying an ESPN session."""
    import duckdb

    def refuse(*a, **k):
        raise duckdb.IOException("Conflicting lock is held")

    monkeypatch.setattr(cred.duckdb, "connect", refuse)
    cred.close_all()
    client = TestClient(_app(), base_url="https://testserver",
                        raise_server_exceptions=False)
    resp = client.post("/_connect", json={"swid": FAKE_SWID,
                                          "espn_s2": FAKE_S2})
    assert resp.status_code == 503
    assert FAKE_S2 not in resp.text
