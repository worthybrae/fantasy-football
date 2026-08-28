"""HEAD, answered wherever GET is.

An uptime monitor asks `HEAD /`. Every one of them does -- it is the
cheapest probe there is, since the server sends the headers it would have
sent for a GET and none of the bytes. This app answered `405 Method Not
Allowed` to all of them, on the document the SPA is served from and on
every `/api` read, which reads to a monitor as an outage.

The cause is that FastAPI registers a route for exactly the methods it was
given, and `@app.get` means GET. Writing `methods=["GET", "HEAD"]` on every
decorator in the app would work and would then have to be remembered on
every endpoint added after this one, which is the kind of rule that lasts
about a month. So it is done once, from outside routing: the request goes
downstream as a GET, and the body is dropped on the way back out. The
headers are the GET's headers -- Cache-Control, ETag, Content-Type,
Content-Length -- which is what HEAD is defined to return and the whole
reason the probe is worth answering.

THE SCOPE IS COPIED, NOT EDITED. uvicorn holds the very dict this
middleware is handed, and it reads `scope["method"]` again AFTER the
response has started: seeing HEAD it sets the expected body length to zero
and writes no body. Rewrite the method in place and it instead waits for a
body of `Content-Length` bytes that this middleware has just removed.

THE FIRST BODY MESSAGE IS THE LAST. Two endpoints here are event streams
that stay open for as long as a draft lasts (`api/live.py`,
`api/demo.py`). A HEAD that faithfully mirrored every chunk as an empty
chunk would hold that stream open forever to send nothing, so the response
is closed at the first one and everything after it is dropped.
"""
from __future__ import annotations

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class HeadAsGet:
    """Route a HEAD as its GET, and return the GET's headers with no body.

    Raw ASGI rather than `@app.middleware("http")`, for the reason
    `api/http_cache.DefaultPrivate` gives: the `BaseHTTPMiddleware` that
    decorator builds re-reads every response through a task group and a
    stream, which is not something to put in front of an SSE endpoint that
    is meant to stay open for an hour.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") != "HEAD":
            await self.app(scope, receive, send)
            return

        done = False

        async def bodiless(message: Message) -> None:
            nonlocal done
            if message["type"] != "http.response.body":
                await send(message)
                return
            if done:
                return
            done = True
            await send({"type": "http.response.body", "body": b"",
                        "more_body": False})

        await self.app({**scope, "method": "GET"}, receive, bodiless)


def install(app) -> None:
    """Add the rewrite. Outside routing, so it covers every route there is."""
    app.add_middleware(HeadAsGet)
