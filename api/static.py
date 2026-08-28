"""Serving the built frontend from the API process.

ONE ORIGIN, BECAUSE THE COOKIE SAYS SO. `api/custody.py` sets the credential
cookie `samesite="lax"`. A browser does not send a Lax cookie on a
cross-site request, so a frontend on its own domain would silently lose the
stored ESPN session on every route that needs it -- the draft list, the
join, the status probe the connect screen runs on load. The alternative is
`SameSite=None` plus CORS with credentials: more parts, weaker cookie, and a
second thing to keep in step. Serving both halves from one process costs a
static mount and removes the whole class of problem.

In development none of this runs: Vite serves the frontend on :5173 and
proxies `/api` to :8000 (see web/vite.config.ts), so `web/dist` does not
exist and `register_spa` does nothing. It exists for the built image, where
there is no Vite and one process answers everything.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api import http_cache

# Where `npm run build` leaves its output, relative to the repository root.
# Overridable because the Docker image lays the tree out its own way and a
# path that only works from a checkout would make the image depend on being
# one.
DIST_ENV = "WEB_DIST_PATH"
DEFAULT_DIST = "web/dist"

# `favicon.svg`, `robots.txt`, the demo video and its poster: Vite copies
# these out of `public/` under their own names, so the next build reuses
# every one of those names and `immutable` would be a lie. An hour is long
# enough that a reader who scrolls the landing page twice does not refetch
# two megabytes of video, and short enough that a deploy is visible to a
# returning visitor within one.
PUBLIC_FILE_CACHE = "public, max-age=3600"


class _HashedAssets(StaticFiles):
    """`/assets`, served with the year-long promise its file names earn.

    Every name under here is content-hashed by Vite, so the bytes behind a
    given URL cannot change: a new build writes new names and rewrites the
    document that asks for them. That is exactly the case `immutable` is
    for -- a browser that has the file does not even send a conditional
    request, and the CDN in front of us holds one copy for everybody.
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = http_cache.IMMUTABLE
        return response


def dist_path() -> Path:
    return Path(os.environ.get(DIST_ENV) or DEFAULT_DIST)


def register_spa(app, dist: Path | None = None) -> bool:
    """Serve `dist` as the app's frontend. Returns whether it was mounted.

    REGISTER THIS LAST. Starlette matches routes in the order they were
    added, and the fallback below matches every path there is. Anything
    registered after it is unreachable.

    A missing build is not an error. Every test run and every dev server is
    a checkout with no `web/dist` in it, and a server that refused to start
    without a frontend would make the API depend on a step that has nothing
    to do with it.
    """
    root = Path(dist) if dist is not None else dist_path()
    index = root / "index.html"
    if not index.is_file():
        return False

    # The hashed output, served as files. Mounted at the same `/assets` the
    # document asks for -- Vite writes absolute paths into index.html, so
    # this prefix is not a choice.
    assets = root / "assets"
    if assets.is_dir():
        app.mount("/assets", _HashedAssets(directory=assets), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        """Any path the API did not claim: a file if there is one, the app
        if there is not.

        THE `/api` REFUSAL IS THE POINT. This route matches everything,
        including endpoints that do not exist, and answering those with
        index.html would turn a 404 a caller can act on into a 200 carrying
        HTML -- which reaches a `fetch` as a corrupt payload rather than as
        a routing mistake, and is a genuinely horrible afternoon. A real
        endpoint never reaches here at all: it was registered first and
        matched first.

        Files before the fallback, so `/favicon.ico`, `/robots.txt` and
        anything else Vite copies out of `public/` are served as themselves.
        `resolve()` and the containment check keep a crafted path (`../..`,
        or an absolute one) inside the build directory.
        """
        if path.startswith("api/"):
            raise HTTPException(status_code=404, detail="No such endpoint.")
        if path:
            candidate = (root / path).resolve()
            if candidate.is_file() and candidate.is_relative_to(root.resolve()):
                return FileResponse(
                    candidate,
                    headers={"Cache-Control": PUBLIC_FILE_CACHE})
        # `/archive`, `/draft`, `/mocks` -- routes in the browser's router,
        # not files here. A reload or a pasted link has to be answered with
        # the document that boots the router.
        # NEVER CACHED FURTHER THAN A REVALIDATION. This document names the
        # hashed bundle, and those names are cached for a year. A CDN or a
        # browser holding yesterday's copy of it would keep booting
        # yesterday's build -- whose assets are all still there, still
        # served, and still valid -- so a deploy would reach nobody who had
        # visited before.
        return FileResponse(index, headers={"Cache-Control": http_cache.NO_CACHE})

    return True
