"""The HTTP edge of credential custody: the cookie, the guard, the two ways
out.

`pipeline/credentials.py` holds the crypto and the tables and knows nothing
about FastAPI. This module is the thin layer that decides, per request,
whether the connection is allowed to carry credential material at all, turns
the browser cookie into a stored session, and mounts the two endpoints a user
needs to end the arrangement.

WHAT IS DELIBERATELY NOT HERE. There is no endpoint that takes a SWID and
returns anything about the account it names. Not a "which account is
connected" lookup, not an existence check, not a 404-vs-200 difference. SWID
is public -- it rides in the invite POST's query string and in the draft
socket's JOIN url, so anyone who has seen a user's draft link has it -- and
any endpoint keyed on it is an account-takeover endpoint no matter how little
it appears to return. Authorization here is the `httpOnly` cookie and only the
cookie. `GET /api/espn/custody` ignores every query parameter it is given,
and `tests/test_custody_api.py` proves it by presenting a real user's real
SWID with no cookie and getting back a disconnected answer.

WHY THE TLS GUARD IS CONDITIONAL. It fires whenever a request carries or would
create custody material -- an `espn_s2` in the body, or the custody cookie in
the headers -- and not otherwise. Two reasons, and the second is the one that
matters. First, a request carrying neither has nothing to protect. Second,
`/api/live/connect-token` predates this subsystem and is still the single-user
local path (a draftSecurity nonce, no account session, no cookie); refusing
that over http://localhost would break the owner's own machine to protect a
credential that is not in the request. So the guard follows the credential
rather than the route.
"""
from __future__ import annotations

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse

from pipeline import credentials as cred


def _forwarded_proto(request: Request) -> str | None:
    return request.headers.get("x-forwarded-proto")


def require_secure(request: Request) -> None:
    """Refuse a plaintext request that is about to carry a credential.

    Translates the transport rule (which lives in `pipeline.credentials`, so
    it is testable without a server) into the one HTTP status a browser fetch
    can act on. 400 rather than 426 Upgrade Required: every caller here is a
    `fetch()` that checks `res.ok` and shows `detail`, and 426 buys nothing it
    would use while being a status some proxies rewrite.
    """
    try:
        cred.require_secure_transport(request.url.scheme,
                                      _forwarded_proto(request))
    except cred.InsecureTransport as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


def _store(store=None):
    return store if store is not None else cred.default_store()


def set_session_cookie(response: Response, minted: cred.MintedSession,
                       secure: bool | None = None) -> None:
    """Put the minted session in the browser, and only in the browser.

    `httponly` because the value is the password to a stored ESPN account
    session: no page script has any reason to read it, and one XSS on any page
    of this site would otherwise hand over every visitor's ESPN account rather
    than merely their view of it.

    `samesite="lax"` because the only cross-site navigation that needs to
    carry it is a top-level GET back from ESPN, and `strict` would break that
    while `none` would send it on every third-party subrequest.

    `secure` tracks the transport check rather than being hard-coded True: a
    `Secure` cookie is silently DROPPED by the browser on a plain-HTTP
    connection, so hard-coding it would turn the dev opt-out into a mode where
    connecting appears to succeed and the user is never logged in. Production
    never reaches that branch, because `require_secure` refused the request
    before this function was called.
    """
    if secure is None:
        # `is_secure_transport("http")` can only be True when the plaintext
        # opt-out is set, so this reads: Secure unless we are explicitly in
        # plaintext dev, where a Secure cookie would simply never be stored.
        secure = not cred.is_secure_transport("http")
    response.set_cookie(
        cred.COOKIE_NAME, minted.cookie, max_age=_max_age(minted),
        httponly=True, secure=secure, samesite="lax", path="/")


def _max_age(minted: cred.MintedSession) -> int:
    """Seconds until the stored session's own expiry.

    Derived from the row rather than from a constant so the cookie and the
    database agree: a cookie outliving its row means a browser that believes
    it is connected and is not, and a cookie dying first means a stored
    credential nobody can reach until the reaper takes it.

    Naive UTC on both sides, matching the TIMESTAMP columns the row came from
    (see `pipeline.credentials._utc` for why the whole subsystem is naive).
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return max(int((minted.expires_at - now).total_seconds()), 0)


def establish_custody(request: Request, response: Response, swid: str,
                      espn_s2: str, store=None) -> cred.MintedSession:
    """Take custody of an ESPN session and hand this browser its key.

    The one function that puts a credential into the store. Both entry points
    (`/api/live/connect-token` and anything added later) go through it, so the
    transport check, the encryption and the cookie flags cannot drift apart
    between callers.
    """
    require_secure(request)
    try:
        minted = _store(store).connect(swid, espn_s2)
    except cred.CustodyUnavailable as exc:
        # 503, not 500: the code is fine and the request was fine; the
        # deployment has no key. `str(exc)` is written to name the variable
        # and never the value (see _parse_keys).
        raise HTTPException(status_code=503, detail=str(exc)) from None
    set_session_cookie(response, minted)
    return minted


def custody_for(request: Request, store=None):
    """The stored ESPN session this browser is authorised for, or None.

    THE ONLY WAY the rest of the API is allowed to reach a stored credential.
    Takes the whole request rather than a cookie string so that no caller can
    pass a value it got from somewhere else -- a query parameter, a JSON body,
    a header it read off the draft URL -- and have it treated as authorization.
    """
    cookie = request.cookies.get(cred.COOKIE_NAME)
    if not cookie:
        return None
    # A cookie on the wire is a credential on the wire, so the guard applies
    # to reads exactly as it does to the connect that minted it.
    require_secure(request)
    try:
        return _store(store).resolve(cookie)
    except cred.CustodyUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None


def _account_hint(swid: str) -> str:
    """The last four characters of the SWID, for "connected as ...".

    Enough for a user with two ESPN accounts to tell which one this browser
    holds, and not enough to be the SWID. Returned only to a request that has
    already proved it holds the session, so this is the user reading their own
    identifier back -- but it is truncated anyway, because a full SWID in a
    response body is a full SWID in every intermediary's access log.
    """
    body = (swid or "").strip("{}")
    return body[-4:] if len(body) >= 4 else ""


def register_custody_routes(app, store=None):
    """Mount the disconnect controls and the status probe.

    Three endpoints and no more. Every one of them is authorised by the
    cookie; none of them accepts an identifier of any kind.
    """

    @app.get("/api/espn/custody")
    def custody_status(request: Request):
        """Whether THIS browser holds a stored ESPN session.

        Answers 200 either way rather than 401 for "no": this is the probe the
        connect screen calls on load, for visitors who have never connected
        anything, and a 401 on the ordinary case would be an error in every
        console and a retry in every client.

        It accepts no parameters. A SWID in the query string is ignored, which
        is the entire point -- see this module's docstring.
        """
        resolved = custody_for(request, store)
        if resolved is None:
            return {"connected": False}
        return {"connected": True,
                "account_hint": _account_hint(resolved.swid),
                "expires_at": resolved.expires_at.isoformat() + "Z"}

    def _cleared(removed: bool, body: dict):
        """The response both disconnects give, with the cookie cleared.

        Cleared even on the refusal path, and built as an explicit
        JSONResponse for exactly that reason: a `raise HTTPException` discards
        the injected `response` object's headers, so the browser would keep
        sending a dead credential-shaped value on every request forever. A
        cookie that names no session is worth nothing to us and is one more
        thing to leak, so it goes.
        """
        out = JSONResponse(
            status_code=200 if removed else 401,
            content=body if removed else {"detail": "not connected"})
        out.delete_cookie(cred.COOKIE_NAME, path="/")
        return out

    @app.post("/api/espn/custody/disconnect")
    def custody_disconnect(request: Request):
        """Log this browser out. The stored ESPN session survives.

        Does NOT require the credential to be readable -- only that the
        session row exists. A user whose credential cannot be decrypted (a
        retired key version) must still be able to clear their own browser,
        and making the logout depend on the thing that is broken would be
        exactly the wrong dependency.
        """
        cookie = request.cookies.get(cred.COOKIE_NAME)
        if cookie:
            require_secure(request)
        removed = bool(cookie) and _store(store).disconnect(cookie)
        return _cleared(removed, {"disconnected": True})

    @app.post("/api/espn/custody/disconnect-everywhere")
    def custody_disconnect_everywhere(request: Request):
        """Delete the stored ESPN session and log out every browser.

        This is the whole of what this product can offer a user who wants out,
        and it is worth being precise about what it does not do: ESPN has no
        per-application revocation, so deleting our copy does not invalidate
        the session at ESPN. A user who believes the credential has been
        exposed still has to log out of ESPN itself. The connect-screen copy
        says so; this endpoint only promises that WE no longer hold it.
        """
        resolved = custody_for(request, store)
        if resolved is None:
            return _cleared(False, {})
        _store(store).forget_credential(resolved.credential_id)
        return _cleared(True, {"disconnected": True, "everywhere": True})

    return app
