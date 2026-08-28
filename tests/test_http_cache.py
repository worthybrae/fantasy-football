"""api.http_cache: what every answer tells a cache it may do with it.

WHY THIS FILE EXISTS. A CDN in front of this app reads one header and acts
on it, and the two ways of getting that header wrong are not symmetrical. A
public answer marked private costs a cache miss. A private answer left
unmarked, in front of a cache-everything rule, gets stored under a URL and
handed to the next visitor -- one reader's league, one reader's draft, one
reader's stored ESPN session, served to a stranger. So the property these
tests hold is not "the fast endpoints are fast": it is that an endpoint
which said nothing is treated as private, and that saying nothing is the
only thing a new endpoint can do by accident.

Nothing here imports `api.main` or `api.live`. The middleware's behaviour is
a property of a path prefix and a header, so a two-route app states it
exactly, and this file keeps passing while those modules are being worked
on.
"""
import json

import pytest
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from api import http_cache, lobby
from api.static import register_spa


# -- the unit ------------------------------------------------------------


def test_public_writes_a_browser_window_and_a_shared_one():
    """Two different questions with two different answers: the reader's own
    browser revalidates (`max-age=0`), the CDN holds a copy for everybody.

    And nothing else on the end. `stale-while-revalidate` was here and came
    off: it is not shared-cache-scoped, so a browser honours it too, which
    is the opposite of what the `max-age=0` beside it is for."""
    res = Response()
    http_cache.public(res, 15)
    assert res.headers["Cache-Control"] == "public, max-age=0, s-maxage=15"


def test_private_says_nobody_writes_it_down():
    res = Response()
    http_cache.private(res)
    assert res.headers["Cache-Control"] == "private, no-store"


# -- the default ---------------------------------------------------------


def _wire(app, build=None):
    """The two halves of the middleware in the order `create_app` uses them.

    Compression first, before any route; the private default LAST, after
    everything `build` registers. That order is the point of the split --
    see `install_private_default` -- so every test here goes through it
    rather than calling the two by hand in whatever order happens to work.
    """
    http_cache.install(app)
    if build is not None:
        build(app)
    http_cache.install_private_default(app)
    return TestClient(app)


def _app():
    """An app wearing the middleware, with the three shapes that matter:
    an endpoint that names no policy, one that names its own, and a path
    outside `/api`."""
    app = FastAPI()
    http_cache.install(app)

    # Named for the real one in api/live.py, which sets no header of its own
    # and carries a reader's league, seat and board. This is the case the
    # default exists for.
    @app.get("/api/live/state")
    def live_state():
        return {"active": True, "my_slot": 3}

    @app.get("/api/market/overview")
    def overview(response: Response):
        http_cache.public(response, 300)
        return {"drafts": 100}

    @app.get("/api/live/events")
    def events():
        # api/live.py sets exactly this on its stream. The middleware must
        # leave it alone.
        return Response(content="ok", headers={"Cache-Control": "no-cache"})

    @app.get("/health")
    def health():
        return {"ok": True}

    http_cache.install_private_default(app)
    return TestClient(app)


def test_an_api_answer_that_named_no_policy_is_private():
    """THE FAILURE THIS FILE EXISTS FOR. `/api/live/state` is one reader's
    draft. Nothing in api/live.py sets a header on it, so the only thing
    standing between that answer and a shared cache is this default."""
    res = _app().get("/api/live/state")
    assert res.headers["Cache-Control"] == "private, no-store"


def test_a_handler_that_named_its_own_policy_keeps_it():
    """The two SSE endpoints set their own header on the way out, and a
    default that overwrote it would replace a considered answer with one
    that knows only the path."""
    res = _app().get("/api/live/events")
    assert res.headers["Cache-Control"] == "no-cache"


def test_a_public_endpoint_is_not_overwritten_either():
    res = _app().get("/api/market/overview")
    assert "s-maxage=300" in res.headers["Cache-Control"]


def test_a_response_that_never_reached_a_route_is_stamped_too():
    """THE REASON THE DEFAULT GOES ON LAST. `api/custody.py` adds
    `CredentialTransportGuard` while its routes are registered, and that
    guard refuses a plaintext request to a credential endpoint with a 400 of
    its own -- before routing, so no handler runs and nothing sets a header.
    Registered before that guard, this stamp would never see the 400 and a
    cache-everything rule would be free to store it. Registered after it, as
    `create_app` does, every response the app emits passes through here.

    The guard below stands in for it: same shape, one file's worth of
    plumbing less."""

    class Refuse:
        """Short-circuits without routing, exactly as the custody guard does."""

        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope["type"] == "http" and scope["path"] == "/api/x":
                await Response(content="no", status_code=400)(scope, receive, send)
                return
            await self.app(scope, receive, send)

    def build(app):
        @app.get("/api/x")
        def never_runs():        # pragma: no cover -- the guard gets there first
            return {"ok": True}

        app.add_middleware(Refuse)

    res = _wire(FastAPI(), build).get("/api/x")

    assert res.status_code == 400
    assert res.headers["Cache-Control"] == "private, no-store"


def test_paths_outside_the_api_are_left_to_their_own_modules():
    """The SPA, its assets and the ADP pages all set headers where they are
    served. Stamping `private` on them from here would make every one of
    them uncacheable and undo the point of the exercise."""
    res = _app().get("/health")
    assert "cache-control" not in {k.lower() for k in res.headers}


# -- the surfaces --------------------------------------------------------


def _built(tmp_path):
    """A `web/dist` as Vite leaves it: an entry document and a hashed asset."""
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>app</title>")
    (dist / "assets" / "index-abc123.js").write_text("console.log(1)")
    (dist / "robots.txt").write_text("User-agent: *\n")
    return dist


def test_hashed_assets_are_immutable(tmp_path):
    """Vite fingerprints these names, so the bytes behind one cannot change
    and a browser holding it need never ask again."""
    client = _wire(FastAPI(), lambda app: register_spa(app, _built(tmp_path)))

    res = client.get("/assets/index-abc123.js")

    assert res.status_code == 200
    assert res.headers["Cache-Control"] == "public, max-age=31536000, immutable"


def test_the_document_is_always_revalidated(tmp_path):
    """The other half of `immutable`. This document names the hashed bundle,
    so a cached copy of it keeps booting the build it was written for --
    whose assets are all still there and still served. Cache it and a deploy
    reaches nobody who has visited before."""
    client = _wire(FastAPI(), lambda app: register_spa(app, _built(tmp_path)))

    assert client.get("/").headers["Cache-Control"] == "no-cache"
    # Same document, reached through the router fallback.
    assert client.get("/archive").headers["Cache-Control"] == "no-cache"


def test_files_copied_out_of_public_are_cacheable_but_not_forever(tmp_path):
    """`robots.txt`, the favicon and the demo video keep their own names
    across builds, so `immutable` would be a lie about them."""
    client = _wire(FastAPI(), lambda app: register_spa(app, _built(tmp_path)))

    res = client.get("/robots.txt")

    assert res.headers["Cache-Control"] == "public, max-age=3600"


def test_the_lobby_is_held_by_a_shared_cache_for_fifteen_seconds():
    """ESPN's open mock rooms, proxied, with nobody's name on the answer --
    the one request every visitor to the landing page makes."""
    lobby.clear_cache()
    try:
        lobby._lobby_summary(fetch=lambda season=None: [], now_ms=1_000.0)
        lobby._lobby_rooms(fetch=lambda season=None: [], now_ms=1_000.0)
        client = _wire(FastAPI(), lobby.register_lobby_routes)

        summary = client.get("/api/lobby")
        rooms = client.get("/api/lobby/rooms")

        assert summary.status_code == 200
        assert "s-maxage=15" in summary.headers["Cache-Control"]
        # The rooms directory is the same module-level cache read through a
        # second shape, with no cookie and no request touched either.
        assert rooms.status_code == 200
        assert "s-maxage=15" in rooms.headers["Cache-Control"]
    finally:
        lobby.clear_cache()


# -- compression ---------------------------------------------------------


def _gzip_app():
    app = FastAPI()
    http_cache.install(app)

    @app.get("/api/big")
    def big():
        # ~20 KB of JSON, which is the size of a real board page and
        # compresses to a fraction of it.
        return {"rows": [{"player_id": f"p{i}", "name": "A Long Player Name"}
                         for i in range(400)]}

    @app.get("/api/small")
    def small():
        return {"ok": True}

    @app.get("/api/stream")
    def stream():
        async def body():
            yield "data: one\n\n"
            yield "data: two\n\n"
        return StreamingResponse(body(), media_type="text/event-stream")

    http_cache.install_private_default(app)
    return TestClient(app)


def test_a_big_json_body_is_gzipped():
    res = _gzip_app().get("/api/big", headers={"Accept-Encoding": "gzip"})

    assert res.headers["Content-Encoding"] == "gzip"
    # httpx decompresses on the way in, so the payload is still readable.
    assert len(res.json()["rows"]) == 400
    assert len(json.dumps(res.json())) > 20_000


def test_a_reader_who_did_not_ask_for_gzip_does_not_get_it():
    res = _gzip_app().get("/api/big", headers={"Accept-Encoding": "identity"})

    assert "content-encoding" not in {k.lower() for k in res.headers}


def test_a_small_body_is_left_alone():
    """Under the threshold, gzip's own framing makes the response bigger and
    both ends pay for it."""
    res = _gzip_app().get("/api/small", headers={"Accept-Encoding": "gzip"})

    assert "content-encoding" not in {k.lower() for k in res.headers}


def test_a_stream_is_never_compressed():
    """THE ONE THAT WOULD NOT LOOK LIKE A BUG. A compressor holds bytes back
    until it has enough of them to be worth emitting, which for an event
    stream means a pick lands and the reader is told about it whenever the
    NEXT few arrive. The draft room would simply look slow, and nothing
    would be in any log. Starlette excludes `text/event-stream` by default;
    this holds that we are relying on it."""
    res = _gzip_app().get("/api/stream", headers={"Accept-Encoding": "gzip"})

    assert "content-encoding" not in {k.lower() for k in res.headers}
    assert res.text == "data: one\n\ndata: two\n\n"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
