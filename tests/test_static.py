"""Serving the built frontend off the same origin as the API.

WHY ONE ORIGIN AT ALL. `api/custody.py` sets the credential cookie
`samesite="lax"`, which means a browser will not send it on a request to a
different site than the one in the address bar. Put the SPA on its own
domain and every route that reads a stored ESPN session -- the draft list,
the join, the status probe the connect screen runs on load -- stops seeing
the cookie. The fix would be `SameSite=None` plus CORS with credentials,
which is more moving parts and a weaker cookie. Serving both from one
process is the cheaper and safer arrangement, so these tests hold the line
that makes it work.
"""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.static import register_spa


def _built(tmp_path):
    """A `web/dist` as Vite leaves it: an entry document and a hashed asset."""
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>app</title>")
    (dist / "assets" / "index-abc123.js").write_text("console.log(1)")
    return dist


def _app(dist):
    app = FastAPI()

    @app.get("/api/ping")
    def ping():
        return {"ok": True}

    register_spa(app, dist)
    return TestClient(app)


def test_the_api_still_wins():
    """THE FAILURE THIS FILE EXISTS FOR. A catch-all that answers every path
    with index.html will happily answer `/api/...` with it too, and the
    symptom is not a 500 -- it is every fetch in the app receiving 200 OK
    with HTML in it, which reads to the frontend as a corrupt payload rather
    than a routing mistake."""
    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as tmp:
        client = _app(_built(pathlib.Path(tmp)))
        assert client.get("/api/ping").json() == {"ok": True}


def test_an_unknown_api_path_is_a_404_not_the_app():
    """Same rule, one step further: a mistyped or retired endpoint has to
    fail as an endpoint. Handing it the SPA would turn a 404 a caller can
    act on into a 200 it cannot."""
    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as tmp:
        client = _app(_built(pathlib.Path(tmp)))
        assert client.get("/api/nothing-here").status_code == 404


def test_a_deep_link_gets_the_app():
    """The whole point of the fallback. `/archive` and `/draft` are routes in
    the browser's router, not files on disk, so a reload or a pasted link
    must still be answered with the document that boots the router."""
    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as tmp:
        client = _app(_built(pathlib.Path(tmp)))
        res = client.get("/archive")
        assert res.status_code == 200
        assert "<title>app</title>" in res.text


def test_hashed_assets_are_served_as_files():
    """Vite fingerprints its output, so these are the URLs the document
    asks for. Answering one with index.html is the same bug as above wearing
    a different hat: the browser gets HTML where it expected JavaScript."""
    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as tmp:
        client = _app(_built(pathlib.Path(tmp)))
        res = client.get("/assets/index-abc123.js")
        assert res.status_code == 200
        assert "console.log(1)" in res.text


def test_no_build_is_not_a_broken_server():
    """A checkout with no `web/dist` -- every test run, and any deployment
    where the frontend build is a separate step -- must still serve the API.
    The mount is skipped, not fatal."""
    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as tmp:
        app = FastAPI()

        @app.get("/api/ping")
        def ping():
            return {"ok": True}

        register_spa(app, pathlib.Path(tmp) / "nothing")
        client = TestClient(app)
        assert client.get("/api/ping").json() == {"ok": True}
        assert client.get("/archive").status_code == 404


def test_the_document_and_its_assets_disagree_on_purpose():
    """A YEAR AND NO SECONDS, IN THE SAME BUILD. Vite hashes the names under
    `/assets`, so those bytes can never change and a browser holding one
    need never ask again. The document that NAMES them keeps its own name
    across every build, so a cached copy of it goes on booting the build it
    was written for -- whose assets are all still there, still valid, still
    served. Cache the document and a deploy reaches nobody who has visited
    before; revalidate the assets and every load pays for round trips to be
    told nothing changed. The pair only works as a pair."""
    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as tmp:
        client = _app(_built(pathlib.Path(tmp)))
        asset = client.get("/assets/index-abc123.js")
        document = client.get("/archive")
        assert asset.headers["Cache-Control"] == "public, max-age=31536000, immutable"
        assert document.headers["Cache-Control"] == "no-cache"


def test_head_on_the_document_is_answered_not_refused():
    """An uptime monitor asks `HEAD /` -- it is the cheapest probe there is,
    and it was a 405 here. The route answers both methods in its own right,
    so this holds even without the app-wide rewrite in `api/head.py`."""
    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as tmp:
        client = _app(_built(pathlib.Path(tmp)))
        for path in ("/", "/archive", "/favicon.svg"):
            res = client.head(path)
            assert res.status_code == 200, path
            assert res.content == b"", path
        # And the headers are still the ones a GET would have carried --
        # a probe that cannot read the cache policy is worth less than one
        # that can.
        assert (client.head("/").headers["Cache-Control"]
                == client.get("/").headers["Cache-Control"])


def test_head_on_a_missing_api_path_is_still_a_404():
    """The catch-all's refusal to answer `/api/...` with the document is not
    weakened by answering a second method: a HEAD to a retired endpoint has
    to fail as an endpoint, exactly as the GET does."""
    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as tmp:
        client = _app(_built(pathlib.Path(tmp)))
        assert client.head("/api/nothing-here").status_code == 404
