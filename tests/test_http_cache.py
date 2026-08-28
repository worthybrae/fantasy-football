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
    browser revalidates (`max-age=0`), the CDN holds a copy for everybody."""
    res = Response()
    http_cache.public(res, 15)
    value = res.headers["Cache-Control"]
    assert "public" in value
    assert "max-age=0" in value
    assert "s-maxage=15" in value
    assert "stale-while-revalidate=15" in value


def test_private_says_nobody_writes_it_down():
    res = Response()
    http_cache.private(res)
    assert res.headers["Cache-Control"] == "private, no-store"


# -- the default ---------------------------------------------------------


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
    app = FastAPI()
    http_cache.install(app)
    register_spa(app, _built(tmp_path))

    res = TestClient(app).get("/assets/index-abc123.js")

    assert res.status_code == 200
    assert res.headers["Cache-Control"] == "public, max-age=31536000, immutable"


def test_the_document_is_always_revalidated(tmp_path):
    """The other half of `immutable`. This document names the hashed bundle,
    so a cached copy of it keeps booting the build it was written for --
    whose assets are all still there and still served. Cache it and a deploy
    reaches nobody who has visited before."""
    app = FastAPI()
    http_cache.install(app)
    register_spa(app, _built(tmp_path))
    client = TestClient(app)

    assert client.get("/").headers["Cache-Control"] == "no-cache"
    # Same document, reached through the router fallback.
    assert client.get("/archive").headers["Cache-Control"] == "no-cache"


def test_files_copied_out_of_public_are_cacheable_but_not_forever(tmp_path):
    """`robots.txt`, the favicon and the demo video keep their own names
    across builds, so `immutable` would be a lie about them."""
    app = FastAPI()
    http_cache.install(app)
    register_spa(app, _built(tmp_path))

    res = TestClient(app).get("/robots.txt")

    assert res.headers["Cache-Control"] == "public, max-age=3600"


def test_the_lobby_is_held_by_a_shared_cache_for_fifteen_seconds():
    """ESPN's open mock rooms, proxied, with nobody's name on the answer --
    the one request every visitor to the landing page makes."""
    lobby.clear_cache()
    try:
        lobby._lobby_summary(fetch=lambda season=None: [], now_ms=1_000.0)
        app = FastAPI()
        http_cache.install(app)
        lobby.register_lobby_routes(app)

        res = TestClient(app).get("/api/lobby")

        assert res.status_code == 200
        assert "s-maxage=15" in res.headers["Cache-Control"]
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
