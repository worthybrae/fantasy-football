"""What each answer tells a cache it may do with it.

A CDN in front of this app is only as useful as the headers it is given.
Cloudflare's "respect origin" cache rule does exactly what the name says:
with no `Cache-Control` on a response it falls back to its own defaults,
which for an API answer is either nothing at all or -- with a
cache-everything rule in place -- the wrong thing entirely. So every
response this app sends says what it is, and the ones that say nothing are
made to say the safe thing.

THE DEFAULT IS PRIVATE, AND THAT IS THE WHOLE DESIGN.
`install_private_default` adds a middleware that stamps `private, no-store`
on any `/api/...` response that did not set a `Cache-Control` of its own. A new endpoint is therefore
uncacheable until somebody decides otherwise, which is the right way round:
forgetting to mark a public endpoint costs a cache miss, while forgetting to
mark a private one would put one visitor's answer in front of the next
visitor. The middleware never overwrites a header that is already there --
`/api/live/events` sets its own `no-cache` on the way out and keeps it.

`public` is the opt-in. It sets `max-age=0` for the browser and a real
`s-maxage` for the shared cache, because those are two different questions:
a reader's own browser should ask again (the page is polling anyway, and a
stale tab is worse than a conditional request), while the CDN can serve one
build of the answer to everyone who arrives in the window.

NOTHING ELSE ON THE END. `stale-while-revalidate` was here and came off: it
is not shared-cache-scoped -- a browser honours it too, which is the
opposite of the `max-age=0` above -- and the Cloudflare plan this is aimed
at ignores it anyway. Two directives that both mean what they say is worth
more than a third that means something different at each hop.

THE PUBLIC LIST IS A LIST OF COOKIE-FREE HANDLERS. `public` on a response
whose body depends on a cookie is the one mistake here that is not
recoverable by editing a header later: the CDN will have served one
reader's answer to the next. Every route that calls `public` must produce
the same bytes for a request with no cookies as for a request with any --
`test_a_public_landing_answer_does_not_vary_with_a_cookie` in
tests/test_api.py holds that for the two landing endpoints, which are the
ones with a live session close enough to reach.
"""
from __future__ import annotations

from starlette.datastructures import MutableHeaders
from starlette.middleware.gzip import GZipMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# The answer for anything with a reader's name on it, and the default for
# every `/api` route that does not choose. `no-store` rather than
# `no-cache`, so nothing is written down at any hop, not even to be
# revalidated later.
PRIVATE = "private, no-store"

# A stream. Not `private`, because these are anonymous and identical for
# everybody -- there is simply nothing here for a cache to hold, and a proxy
# that tried would buffer the stream instead of passing it through.
NO_STORE = "no-store"

# Vite fingerprints its output, so a given `/assets/index-CZ67MPId.js` is
# that exact build forever and a new build has a new name. `immutable` is
# the promise that lets a browser skip even the conditional request.
IMMUTABLE = "public, max-age=31536000, immutable"

# The document that boots the app, and it names the hashed assets above. It
# has to be revalidated on every load or a deploy would reach nobody who had
# already visited -- their browser would keep booting the old bundle by its
# old asset names, which are still cached and still served.
NO_CACHE = "no-cache"

# Below this a compressed body is not worth the CPU on either end, and for
# the smallest ones gzip's own framing makes the response bigger.
MIN_GZIP_BYTES = 1024

# The prefix the default applies to. Everything else this process serves --
# the SPA, its assets, the ADP pages, the sitemap -- sets its own header in
# the module that serves it.
API_PREFIX = "/api/"


def public(response, s_maxage: int, max_age: int = 0) -> None:
    """Let a shared cache hold this answer for `s_maxage` seconds.

    ONLY FOR A HANDLER THAT READS NO COOKIE. See the module docstring: this
    header is a promise that every reader may be served the same bytes.

    `max_age` is what the reader's own browser gets and defaults to zero:
    the pages that call this are polling on their own timers, and a browser
    cache that answered those polls locally would make the poll pointless.
    """
    response.headers["Cache-Control"] = (
        f"public, max-age={max_age}, s-maxage={s_maxage}")


def private(response) -> None:
    """Nobody but this reader, and nobody writes it down."""
    response.headers["Cache-Control"] = PRIVATE


class DefaultPrivate:
    """Stamp `private, no-store` on `/api` answers that named no policy.

    Raw ASGI rather than `@app.middleware("http")` on purpose. The
    `BaseHTTPMiddleware` that decorator builds wraps every response in its
    own task group and re-reads the body through a stream, which is a lot of
    machinery to put in front of two SSE endpoints that are meant to stay
    open for an hour. This only ever touches the header list on the way out.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope.get("path", "").startswith(API_PREFIX):
            await self.app(scope, receive, send)
            return

        async def stamped(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                # NEVER OVERWRITE. A handler that set its own header has
                # already answered this question, and the answer is more
                # informed than a default that only knows the path.
                if "cache-control" not in headers:
                    headers["Cache-Control"] = PRIVATE
            await send(message)

        await self.app(scope, receive, stamped)


def install(app) -> None:
    """Compression, for the whole app.

    Call this immediately after the `FastAPI()` is constructed. Starlette's
    own gzip already leaves `text/event-stream` alone (it is in
    `DEFAULT_EXCLUDED_CONTENT_TYPES`), so neither SSE endpoint is at risk of
    having its events held back inside a compressor's buffer.
    """
    app.add_middleware(GZipMiddleware, minimum_size=MIN_GZIP_BYTES)


def install_private_default(app) -> None:
    """Stamp the default on the way out, from as far out as there is.

    CALL THIS LAST -- the final line of `create_app`, after every router and
    after every other middleware. Starlette builds the stack so that the
    middleware added LAST sits OUTERMOST, and outermost is the only place
    this works from.

    The reason is a response that never reaches a route. `api/custody.py`
    adds `CredentialTransportGuard` while its routes are registered, and
    that guard answers a plaintext request to a credential endpoint with a
    400 of its own making -- routing never happens, no handler runs, and
    nothing sets a header. Registered inside that guard, this stamp would
    never see the 400. Registered outside it, every response the app emits
    passes through here on the way to the socket, whoever produced it.
    """
    app.add_middleware(DefaultPrivate)
