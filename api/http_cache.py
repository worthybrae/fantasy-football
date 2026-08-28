"""What each answer tells a cache it may do with it.

A CDN in front of this app is only as useful as the headers it is given.
Cloudflare's "respect origin" cache rule does exactly what the name says:
with no `Cache-Control` on a response it falls back to its own defaults,
which for an API answer is either nothing at all or -- with a
cache-everything rule in place -- the wrong thing entirely. So every
response this app sends says what it is, and the ones that say nothing are
made to say the safe thing.

THE DEFAULT IS PRIVATE, AND THAT IS THE WHOLE DESIGN. `install` adds a
middleware that stamps `private, no-store` on any `/api/...` response that
did not set a `Cache-Control` of its own. A new endpoint is therefore
uncacheable until somebody decides otherwise, which is the right way round:
forgetting to mark a public endpoint costs a cache miss, while forgetting to
mark a private one would put one visitor's answer in front of the next
visitor. The middleware never overwrites a header that is already there --
`/api/live/events` sets its own `no-cache` on the way out and keeps it.

`public` is the opt-in. It sets `max-age=0` for the browser and a real
`s-maxage` for the shared cache, because those are two different questions:
a reader's own browser should ask again (the page is polling anyway, and a
stale tab is worse than a conditional request), while the CDN can serve one
build of the answer to everyone who arrives in the window. The
`stale-while-revalidate` on the end is what keeps a burst off the origin
when the window closes: the CDN serves the expiring copy and refreshes
behind it, rather than sending every waiting reader through to us at once.
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


def public(response, s_maxage: int, max_age: int = 0,
           stale_while_revalidate: int | None = None) -> None:
    """Let a shared cache hold this answer for `s_maxage` seconds.

    `max_age` is what the reader's own browser gets and defaults to zero:
    the pages that call this are polling on their own timers, and a browser
    cache that answered those polls locally would make the poll pointless.

    `stale_while_revalidate` defaults to the same window as `s_maxage` --
    long enough that a popular answer is never fetched from the origin by a
    reader who is waiting, short enough that nobody is served something an
    entire second window out of date.
    """
    swr = s_maxage if stale_while_revalidate is None else stale_while_revalidate
    response.headers["Cache-Control"] = (
        f"public, max-age={max_age}, s-maxage={s_maxage}, "
        f"stale-while-revalidate={swr}")


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
    """Compression and the private default, for the whole app.

    Call this immediately after the `FastAPI()` is constructed and before
    any route is registered. Middleware added later sits further out, so
    gzip ends up outermost and compresses whatever the header stamp let
    through -- which is the order that also means a body is compressed once,
    at the edge of the app, rather than per router.

    Starlette's own gzip already leaves `text/event-stream` alone (it is in
    `DEFAULT_EXCLUDED_CONTENT_TYPES`), so neither SSE endpoint is at risk of
    having its events held back inside a compressor's buffer.
    """
    app.add_middleware(DefaultPrivate)
    app.add_middleware(GZipMiddleware, minimum_size=MIN_GZIP_BYTES)
