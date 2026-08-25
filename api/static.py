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

# Where `npm run build` leaves its output, relative to the repository root.
# Overridable because the Docker image lays the tree out its own way and a
# path that only works from a checkout would make the image depend on being
# one.
DIST_ENV = "WEB_DIST_PATH"
DEFAULT_DIST = "web/dist"


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
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

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
                return FileResponse(candidate)
        # `/archive`, `/draft`, `/mocks` -- routes in the browser's router,
        # not files here. A reload or a pasted link has to be answered with
        # the document that boots the router.
        return FileResponse(index)

    return True
