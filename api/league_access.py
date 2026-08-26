"""Who may read a league's history: an account that holds the league.

One check, shared by every per-league route (the history routes here and
the report routes when they land), so "does this session own this league"
is answered one way. The league list is the same one `GET /api/espn/drafts`
serves, read through the same `league_entries` call and the same cache.
"""
from __future__ import annotations

from fastapi import HTTPException, Request

from api.drafts import CURRENT_SEASON, Session, _cached, session_for
from pipeline import espn_drafts as drafts


def is_mock(name: str | None) -> bool:
    return "mock" in (name or "").lower()


def owned_league(request: Request, store, league_id: str, fetch=None) -> Session:
    """The session behind this request, if it holds `league_id`.

    403 with no session or a league the account does not hold. 400 for a
    mock room: a mock has no history worth a page.
    """
    session = session_for(request, store)
    if session is None:
        raise HTTPException(status_code=403, detail="Sign in to read a league.")
    try:
        rows = _cached((session.swid, "entries", str(CURRENT_SEASON)),
                       lambda: drafts.league_entries(session.swid, session.cookies,
                                                     str(CURRENT_SEASON), fetch=fetch))
    except drafts.SessionExpired:
        raise HTTPException(status_code=403, detail="ESPN no longer accepts this session.")
    mine = next((r for r in rows if str(r.get("league_id")) == str(league_id)), None)
    if mine is None:
        raise HTTPException(status_code=403, detail="That league is not on this account.")
    if is_mock(mine.get("name")):
        raise HTTPException(status_code=400, detail="A mock room has no history.")
    return session
