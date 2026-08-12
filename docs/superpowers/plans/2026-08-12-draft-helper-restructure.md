# Draft Helper Restructure — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Make the site a draft helper for ESPN leagues. You land on a page that
asks for your ESPN draft URL, it syncs to that draft, and from then on you are
in the live room. The research tool that exists today moves to `/legacy`.

**Architecture:** One new endpoint takes a draft URL, resolves the league id,
builds the `DraftSession` and starts the socket listener on a thread. The
client polls `/api/live/state` as before. Routing is restructured so `/` is the
connect screen, `/draft` is the live room, and everything that exists today is
re-parented under `/legacy` unchanged.

**Tech Stack:** FastAPI, Playwright, React 19 + react-router-dom 7, Vite.

## Global Constraints

- **ESPN only.** Real leagues and mock drafts are the same code path — both
  carry `leagueId` in the URL and both resolve through `parse_league_id`.
- **The URL must be the draft room, not the waiting room.** A waiting-room URL
  resolves to the same league id but the socket does not carry picks until the
  draft starts; the connect screen says so rather than silently sitting idle.
- **You draft in our window.** `run_listener` opens a VISIBLE browser and the
  human drafts in it. One browser, one socket — the connection ESPN already
  trusts. Never open a second socket as the same team.
- **The seed stays pinned for the session's lifetime.**
- **Nothing under `/legacy` changes behaviour.** It is a re-parenting, not a
  rewrite. Existing components are imported as they are.
- Run tests with `.venv/bin/python -m pytest`; frontend with `npx tsc -b && npm run build` in `web/`.
- DuckDB is single-writer: no test may open `data/nfl.duckdb`.

---

## File Structure

| file | responsibility |
|---|---|
| `api/live.py` (modify) | add `POST /api/live/connect` and `POST /api/live/select`; own the listener thread |
| `web/src/App.tsx` (modify) | new route table |
| `web/src/pages/Connect.tsx` (create) | the landing screen |
| `web/src/pages/LiveDraft.tsx` (create) | the live room |
| `web/src/api.ts` (modify) | `connectDraft`, `selectPlayer`, `LiveState` |
| `tests/test_live_api.py` (modify) | connect + select endpoint tests |

---

### Task 1: Connect endpoint and listener thread

**Files:**
- Modify: `api/live.py`
- Test: `tests/test_live_api.py`

**Interfaces:**
- Consumes: `build_session`, `register_live_routes` state (existing in this
  file); `pipeline.espn_league.parse_league_id`;
  `pipeline.draft_listener.DraftListener`, `run_listener`;
  `pipeline.espn_live.apply_picks`.
- Produces: `POST /api/live/connect` accepting `{"url": str, "my_slot": int}`,
  returning `{"connected": bool, "league_id": str, "board_fingerprint": str}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_connect_rejects_a_url_with_no_league_id(tmp_path):
    from fastapi.testclient import TestClient
    from api.main import create_app
    c = TestClient(create_app(str(tmp_path / "t.duckdb")))
    r = c.post("/api/live/connect",
               json={"url": "https://fantasy.espn.com/football/mockdraftlobby",
                     "my_slot": 4})
    assert r.status_code == 422
    assert "league id" in r.json()["detail"].lower()


def test_connect_accepts_a_real_draft_url_and_a_mock_url(tmp_path, monkeypatch):
    """Both forms carry leagueId and resolve identically -- a mock draft is
    not a special case, which is what makes mocks usable as a rehearsal."""
    from api.live import _resolve_league_id
    assert _resolve_league_id(
        "https://fantasy.espn.com/football/draft?leagueId=53929318") == "53929318"
    assert _resolve_league_id(
        "https://fantasy.espn.com/football/draft?leagueId=539649131&seasonId=2026"
    ) == "539649131"
    assert _resolve_league_id("539649131") == "539649131"


def test_connect_warns_on_a_waiting_room_url(tmp_path):
    """A waiting-room URL resolves to the same league, but the socket carries
    no picks until the draft starts. Saying so beats sitting silently idle."""
    from api.live import _is_waiting_room
    assert _is_waiting_room("https://fantasy.espn.com/football/waitingroom?leagueId=1")
    assert not _is_waiting_room("https://fantasy.espn.com/football/draft?leagueId=1")
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_live_api.py -k "connect or waiting" -v`
Expected: FAIL with `ImportError: cannot import name '_resolve_league_id'`

- [ ] **Step 3: Implement**

Append to `api/live.py`:

```python
import re
import threading

from fastapi import HTTPException
from pydantic import BaseModel

from pipeline.draft_listener import DraftListener, run_listener
from pipeline.espn_league import STATE_PATH, parse_league_id
from pipeline.espn_live import apply_picks


class ConnectBody(BaseModel):
    url: str
    my_slot: int


def _resolve_league_id(url: str) -> str:
    """League id from anything ESPN shows you.

    A real league's draft URL, a mock's, or a bare id all carry the same
    thing. Treating a mock as an ordinary league is deliberate: it is what
    lets a mock draft rehearse the whole system without a special path
    through it that would then be the untested one on draft night.
    """
    try:
        return parse_league_id(url)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="No league id in that URL. Open your draft room and copy "
                   "the address bar -- it should contain leagueId=.")


def _is_waiting_room(url: str) -> bool:
    return "waitingroom" in (url or "").lower()
```

Then inside `register_live_routes`, add:

```python
    @app.post("/api/live/connect")
    def live_connect(body: ConnectBody):
        league_id = _resolve_league_id(body.url)
        cur = conn.cursor()
        try:
            session = build_session(cur, body.my_slot)
        finally:
            cur.close()
        listener = DraftListener(session.crosswalk)

        def pump():
            """Write every pick the socket reports, then recompute.

            `apply_picks` replaces the table wholesale, so re-folding the
            whole event stream on each change is correct rather than
            wasteful -- and it keeps `picks_from_events` the single
            definition of pick numbering.
            """
            def on_change():
                c2 = conn.cursor()
                try:
                    apply_picks(c2, listener.picks())
                    made = c2.execute("SELECT count(*) FROM drafted").fetchone()[0]
                finally:
                    c2.close()
                _recompute(session, made)

            run_listener(listener, body.url, STATE_PATH, on_change=on_change)

        with lock:
            state.update({"session": session, "listener": listener,
                          "candidates": [], "as_of_pick": None,
                          "unmapped": [], "last_poll_at": None})
            state["generation"] = state.get("generation", 0) + 1
        threading.Thread(target=pump, daemon=True).start()
        return {"connected": True, "league_id": league_id,
                "board_fingerprint": session.board_fingerprint,
                "waiting_room": _is_waiting_room(body.url)}
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_live_api.py -v`

- [ ] **Step 5: Full suite**

Run: `.venv/bin/python -m pytest tests -q` — 426 passed / 1 skipped before.

- [ ] **Step 6: Commit**

```bash
git add api/live.py tests/test_live_api.py
git commit -m "feat: connect to a draft by its ESPN URL

One endpoint takes the draft room address, resolves the league id, builds
the session and starts the socket listener on a thread. A mock draft is not
a special case -- both forms carry leagueId -- which is what lets a mock
rehearse the whole system rather than exercising a different path from the
one that runs on draft night."
```

---

### Task 2: Route restructure and the connect screen

**Files:**
- Modify: `web/src/App.tsx`, `web/src/api.ts`
- Create: `web/src/pages/Connect.tsx`

**Interfaces:**
- Consumes: `POST /api/live/connect` from Task 1.
- Produces: routes `/` (Connect), `/draft` (placeholder until Task 3),
  `/legacy`, `/legacy/players/:slug`, `/legacy/draft-board`.

- [ ] **Step 1: Add the API client**

Append to `web/src/api.ts`:

```typescript
export interface ConnectResult {
  connected: boolean
  league_id: string
  board_fingerprint: string
  waiting_room: boolean
}

export async function connectDraft(url: string, mySlot: number): Promise<ConnectResult> {
  const res = await fetch('/api/live/connect', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url, my_slot: mySlot }),
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail ?? 'Could not connect to that draft')
  }
  return res.json()
}
```

- [ ] **Step 2: Build the connect screen**

`web/src/pages/Connect.tsx`:

```tsx
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { connectDraft } from '../api'

export function Connect() {
  const [url, setUrl] = useState('')
  const [slot, setSlot] = useState(1)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const navigate = useNavigate()

  async function go(e: React.FormEvent) {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      await connectDraft(url.trim(), slot)
      navigate('/draft')
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="connect">
      <h1>Sync to your draft</h1>
      <p className="connect-lede">
        Open your ESPN draft room and paste its address. Works for real leagues
        and mock drafts alike.
      </p>
      <form onSubmit={go}>
        <label htmlFor="draft-url">Draft room URL</label>
        <input
          id="draft-url" value={url} autoFocus
          onChange={(e) => setUrl(e.target.value)}
          placeholder="https://fantasy.espn.com/football/draft?leagueId=…"
        />
        <label htmlFor="slot">Your draft slot</label>
        <input
          id="slot" type="number" min={1} max={20} value={slot}
          onChange={(e) => setSlot(Number(e.target.value))}
        />
        <button type="submit" disabled={busy || !url.trim()}>
          {busy ? 'Connecting…' : 'Sync to draft'}
        </button>
      </form>
      {error && <p className="connect-error">{error}</p>}
      {/* The waiting room resolves to the same league but its socket carries
          no picks until the draft starts, so it looks identical to a broken
          connection. Naming it up front is cheaper than debugging it live. */}
      <p className="connect-note">
        Use the draft room address, not the waiting room — the waiting room
        reports no picks until the draft actually begins.
      </p>
      <p className="connect-legacy">
        Looking for the research tool? It's at <a href="/legacy">/legacy</a>.
      </p>
    </main>
  )
}
```

- [ ] **Step 3: Restructure the routes**

In `web/src/App.tsx`, replace the `<Routes>` block with:

```tsx
      <Routes>
        <Route path="/" element={<Connect />} />
        <Route path="/draft" element={<LiveDraft />} />
        {/* The research tool, re-parented unchanged. Same components, same
            behaviour -- only the paths moved. */}
        <Route path="/legacy" element={<Board />} />
        <Route path="/legacy/players/:slug" element={<PlayerPage />} />
        <Route path="/legacy/draft-board" element={<DraftBoardPage />} />
      </Routes>
```

Add the imports, and until Task 3 lands use a placeholder:

```tsx
function LiveDraft() {
  return <main className="live-pending">Draft room — built in Task 3.</main>
}
```

Every internal link that points at `/players/…` or `/draft-board` must be
updated to its `/legacy` equivalent. Search the whole `web/src` tree for
those strings; a missed one is a dead link.

- [ ] **Step 4: Verify the build**

Run: `cd web && npx tsc -b && npm run build`
Expected: no type errors, build succeeds.

- [ ] **Step 5: Commit**

```bash
git add web/src
git commit -m "feat: land on a draft-sync screen, move the research tool to /legacy

The site is a draft helper now, so the first thing it asks for is the draft
you want to follow. The player table, profiles and predicted board are
unchanged behind /legacy -- this is a re-parenting, not a rewrite."
```

---

### Task 3: The live draft room

**Files:**
- Create: `web/src/pages/LiveDraft.tsx`
- Modify: `web/src/App.tsx` (swap the placeholder), `web/src/App.css`

**Interfaces:**
- Consumes: `GET /api/live/state`.

Layout, from the reviewed mockup: a status strip (whose turn, pick number,
clock), the board grid on the left, a recommendation rail on the right, and
the existing `PlayerProfile` opening as a sidebar when a player is clicked.

- [ ] **Step 1: Build the page**

Poll `/api/live/state` every 2500ms. Render:
- **Strip**: `on_the_clock === my_slot` drives the "you're up" treatment;
  show `picks_made`, and the pick number.
- **Rail**: `candidates[0]` is the call — name, position, team, and the
  reason built from its `ev`, the runner-up's `ev`, and both `applied_pct`
  values. Remaining candidates listed with cost against the leader and their
  survival percentage.
- **Staleness**: when `stale` is true, say the board may be behind rather
  than showing its numbers as current.
- **Recomputing**: when `candidates_as_of_pick < picks_made`, mark the list
  as recomputing.
- **Unmapped**: when `unmapped_picks` is non-empty, surface it — those
  players are still on the board and should not be.

- [ ] **Step 2: Verify the build**

Run: `cd web && npx tsc -b && npm run build`

- [ ] **Step 3: Commit**

```bash
git add web/src
git commit -m "feat: the live draft room"
```

---

## Self-Review

**Spec coverage.** The home page asks for a draft URL (Task 2), syncs to it
(Task 1), supports real and mock drafts through one path (Task 1, asserted by
test), and the existing UI moves to `/legacy` (Task 2).

**Known gap, deliberate.** `POST /api/live/select` — the draft button — is not
in this plan. It needs the listener proven against a live draft first, and
this plan's whole point is getting to the screen where that can be observed.
It follows as its own task once a mock run confirms picks land.

**Risk carried forward.** Navigating directly to a draft room URL may not work
if ESPN requires the waiting-room flow to enter. `_is_waiting_room` warns about
the wrong URL, but whether the right one can be entered cold is unverified and
is the first thing a mock run will tell us.
