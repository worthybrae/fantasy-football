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

import duckdb
from fastapi import HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers

from pipeline import credentials as cred
from pipeline import redact
from pipeline.espn_identity import OwnershipUnproven

# Paths whose REQUEST BODY may carry an ESPN account session, and which the
# transport guard therefore has to look inside. Everything under
# `/api/espn/custody` is covered by the cookie check alone and needs no entry
# here; this list is for endpoints that take a credential as a field.
CREDENTIAL_BEARING_PATHS = frozenset({"/api/live/connect-token"})

# How much of such a body to buffer while deciding. These bodies are a few
# hundred bytes; the cap exists so that a deliberately enormous POST cannot be
# used to make the guard hold a request in memory. Past the cap the guard
# stops buffering and lets the rest stream through, having already seen far
# more than enough to find the field name.
_MAX_GUARDED_BODY = 64 * 1024


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


class CredentialTransportGuard:
    """Refuse a plaintext request carrying credentials, BEFORE the app sees it.

    WHY THIS IS NOT A CHECK INSIDE THE HANDLER, which is where it started.
    FastAPI validates the request body before it enters the handler function,
    so a check written as the handler's first statement never runs on a
    request whose body fails validation -- and a body that fails validation is
    still a body that was transmitted. Worse, FastAPI's own 422 echoes the
    rejected input straight back, so a connect with one field misspelled sent
    the account session over plaintext AND printed it into the response, the
    access log and the browser's network tab. Measured, on the real model:
    omit `season` and the 422 body contains the whole `espn_s2`.

    ASGI middleware runs before routing, before validation, and before
    anything can serialise a value. That is the only layer at which "plain
    HTTP fails closed" is a statement about every request rather than about
    the requests that happened to be well formed.

    Written as raw ASGI rather than `BaseHTTPMiddleware` because it has to
    read the body and then hand the SAME body to the application. Raw ASGI can
    buffer the messages and replay them; `BaseHTTPMiddleware` cannot without
    consuming the stream the route is about to read.

    The body is searched for the BYTES `espn_s2`, not parsed as JSON. That is
    the point: a malformed body, a wrongly typed field, an extra field, a body
    that is not JSON at all -- none of them parse, and all of them can still
    carry the credential. A byte search cannot be fooled by a shape, and its
    only failure mode is refusing a request that merely mentions the name,
    which costs nothing.
    """

    def __init__(self, app, paths=None, store=None):
        self.app = app
        self.paths = frozenset(paths or CREDENTIAL_BEARING_PATHS)

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)
        headers = Headers(scope=scope)
        # The cookie is in the headers, so this costs nothing and covers every
        # path: a request presenting the custody cookie is carrying the
        # password to a stored ESPN session whatever else it is doing.
        carries = cred.COOKIE_NAME in _cookie_names(headers.get("cookie", ""))

        replay = []
        if not carries and scope.get("path", "") in self.paths:
            buffered = b""
            while True:
                message = await receive()
                replay.append(message)
                if message.get("type") != "http.request":
                    break
                buffered += message.get("body", b"") or b""
                if not message.get("more_body", False):
                    break
                if len(buffered) >= _MAX_GUARDED_BODY:
                    break
            carries = b"espn_s2" in buffered.lower()

        if carries:
            try:
                cred.require_secure_transport(
                    scope.get("scheme", ""), headers.get("x-forwarded-proto"))
            except cred.InsecureTransport as exc:
                response = JSONResponse(status_code=400,
                                        content={"detail": str(exc)})
                return await response(scope, receive, send)

        if not replay:
            return await self.app(scope, receive, send)

        # Hand the application the messages already taken off the wire, then
        # get out of the way. Without this the route would await a body that
        # has already been consumed and hang.
        pending = list(replay)

        async def replaying_receive():
            if pending:
                return pending.pop(0)
            return await receive()

        return await self.app(scope, replaying_receive, send)


def _cookie_names(header: str):
    """Just the names in a Cookie header. Values are never parsed here -- the
    guard only needs to know whether ours is present, and a parser that
    touches the value is a parser that can log it."""
    return {part.split("=", 1)[0].strip()
            for part in (header or "").split(";") if part.strip()}


def safe_validation_error_handler(request: Request, exc):
    """FastAPI's 422, with the rejected input removed.

    THE DEFAULT HANDLER ECHOES THE BODY. Pydantic attaches the offending value
    to every error as `input`, and FastAPI serialises the lot, so one missing
    field on a connect returns the caller's whole `espn_s2` in the response --
    into the browser's network tab, into every proxy's access log, and into
    whatever error reporter the page installs. The field being validated does
    not have to be the credential; `input` on a missing-field error is the
    ENTIRE body.

    So the input never leaves this function. What goes back is what a client
    can actually act on -- which field, and what is wrong with it -- and even
    that is passed through the scrubber, because `loc` and `msg` are built
    from names this server does not control.

    The transport guard above stops such a request over plaintext; this stops
    it over TLS as well, where the value would still reach the log. Neither
    substitutes for the other.
    """
    safe = [{"type": error.get("type"),
             "loc": [redact.redact(part) for part in error.get("loc", ())],
             "msg": redact.redact(error.get("msg", ""))}
            for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": safe})


def install_credential_guards(app, paths=None):
    """Both pre-handler protections, mounted together so neither is forgotten.

    One call rather than two exported pieces: they cover the same failure from
    two sides (the request must not arrive in the clear; the rejection must
    not echo it back), and an app that installed one and not the other would
    look protected while leaking.
    """
    app.add_middleware(CredentialTransportGuard, paths=paths)
    app.add_exception_handler(RequestValidationError,
                              safe_validation_error_handler)
    return app


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
    transport check, the ownership proof, the encryption and the cookie flags
    cannot drift apart between callers.

    The caller's existing cookie is passed through, and it is what decides
    whether this connect may REPLACE a stored session or only attach a new
    browser to it (see `CredentialStore.connect`, lock 2). Read from the
    request rather than taken as an argument for the same reason `custody_for`
    is: a cookie the caller assembled from somewhere else is not authorization.
    """
    require_secure(request)
    try:
        minted = _store(store).connect(
            swid, espn_s2, cookie=request.cookies.get(cred.COOKIE_NAME))
    except OwnershipUnproven as exc:
        # 403, not 401: the caller is not being asked to authenticate to US,
        # they are being told the session they sent does not demonstrably
        # belong to the account they named. `str(exc)` is written to be
        # showable; it names no value and distinguishes no SWID.
        raise HTTPException(status_code=403, detail=str(exc)) from None
    except cred.CustodyUnavailable as exc:
        # 503, not 500: the code is fine and the request was fine; the
        # deployment has no key, or another process holds the store's write
        # lock. `str(exc)` is written to name the variable and never the
        # value (see _parse_keys).
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except duckdb.Error as exc:
        # Any other storage failure. Never a 500 with a traceback: this
        # request has an ESPN session in it, and an unhandled exception is the
        # one path where a framework decides for itself what to print.
        raise HTTPException(
            status_code=503,
            detail=f"the credential store is not usable "
                   f"({type(exc).__name__})") from None
    set_session_cookie(response, minted)
    return minted


def abandon_custody(minted, store=None) -> None:
    """Undo an `establish_custody` whose request then failed.

    THE STRANDED ROW. `establish_custody` writes the credential and puts the
    cookie on the response object, but the response is only sent if the
    handler returns. If anything after it raises, the row holds a live
    `espn_s2` and NO browser holds the cookie for it -- and every way out of
    the store (`disconnect`, `disconnect-everywhere`, `custody_for`) requires
    that cookie. Nobody, including the user whose session it is, can delete
    it. It survives the full thirty-day TTL with no remedy.

    So the failure path deletes it. Best effort and silent: this runs while an
    exception is already on its way up, and a second exception raised here
    would replace a diagnosable error with a confusing one. If it fails the
    reaper still gets the row eventually, which is the same guarantee the
    stranded case had -- just without pretending it is fine.
    """
    if minted is None:
        return
    try:
        _store(store).forget_credential(minted.credential_id)
    except Exception:      # noqa: BLE001 -- see above
        pass


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
    except duckdb.Error as exc:
        # Same reasoning as `establish_custody`: a second uvicorn worker
        # holding the file lock must be a clean 503, not an unhandled 500 on a
        # request that carries the custody cookie.
        raise HTTPException(
            status_code=503,
            detail=f"the credential store is not usable "
                   f"({type(exc).__name__})") from None


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


def register_custody_routes(app, store=None, paths=None):
    """Mount the disconnect controls and the status probe.

    Three endpoints and no more. Every one of them is authorised by the
    cookie; none of them accepts an identifier of any kind.

    Also installs the two pre-handler guards on the app it is given (see
    `install_credential_guards`). They are mounted here rather than left to
    the caller because they protect `/api/live/connect-token` as much as these
    routes, and an app that mounted the endpoints without them would be the
    dangerous configuration.
    """
    install_credential_guards(app, paths=paths)

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
