/*
 * Draft Helper bookmarklet -- readable source of truth.
 *
 * The compact, draggable version lives in web/src/lib/bookmarklet.ts (the
 * install screen renders it as a `javascript:` link). Keep the two in sync:
 * this file is what a human edits and reviews; that one is what ships.
 *
 * What it does, and why this is the whole trick:
 *
 *   Runs ON the ESPN draft page (fantasy.espn.com origin). ESPN's draftSecurity
 *   endpoint mints a per-draft token, and its CORS allows the fantasy.espn.com
 *   origin to read the response with credentials (verified: the endpoint returns
 *   `access-control-allow-origin: https://fantasy.espn.com` +
 *   `access-control-allow-credentials: true`). So a bookmarklet -- which runs in
 *   that origin -- can mint the token while the browser attaches the account
 *   session cookie (espn_s2) automatically. espn_s2 is HttpOnly: the bookmarklet
 *   never SEES it, it just rides along on the fetch and stays on ESPN.
 *
 *   Then it opens Draft Helper in a new window with only the throwaway token and
 *   the public ids in the URL hash. Nothing is pasted, nothing is stored, and
 *   the account session never leaves the ESPN tab.
 *
 * To point it at a deployed app instead of local dev, change APP (and rebuild
 * the compact copy in web/src/lib/bookmarklet.ts).
 */
(async () => {
  const APP = 'http://localhost:5173'
  const ESPN = 'https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl'

  const u = location.href
  const league = (u.match(/leagueId=(\d+)/) || [])[1]
  const team = (u.match(/teamId=(\d+)/) || [])[1]
  // memberId carries the SWID directly on a draft-room URL; fall back to the
  // SWID cookie (not HttpOnly, so document.cookie can read it) otherwise.
  let swid = (u.match(/memberId=(\{[^}]+\})/) || [])[1]
  const season = (u.match(/seasonId=(\d+)/) || [])[1]
    || String(new Date().getFullYear())
  if (!swid) {
    const m = document.cookie.match(/SWID=([^;]+)/)
    swid = m ? decodeURIComponent(m[1]) : null
  }

  if (!league || !team) {
    alert('Open your ESPN draft room first, then click Draft Helper.\n'
      + '(This page has no leagueId/teamId in its address.)')
    return
  }
  if (!swid) {
    alert("Couldn't find your ESPN id — make sure you're signed in to ESPN.")
    return
  }

  // credentials:'include' attaches espn_s2 automatically; ESPN's CORS allows
  // this origin to read the response back. The token is a bare integer body.
  const url = `${ESPN}/seasons/${season}/segments/0/leagues/${league}`
    + `/teams/${team}/draftSecurity`
  let token
  try {
    const r = await fetch(url, {
      credentials: 'include',
      headers: { 'x-fantasy-source': 'kona' },
    })
    if (!r.ok) {
      alert(r.status === 401 || r.status === 403
        ? "ESPN didn't recognise your session — sign in to ESPN and retry."
        : `ESPN returned ${r.status}. Is the draft actually open?`)
      return
    }
    token = (await r.text()).trim()
  } catch (e) {
    alert("Couldn't reach ESPN: " + e.message)
    return
  }

  // Only the throwaway token + public ids travel, and they travel into a
  // window this click opened -- in the hash, so the token never lands in a
  // server access log.
  const q = new URLSearchParams({
    leagueId: league, teamId: team, swid, token, season,
  })
  window.open(`${APP}/#${q.toString()}`, 'DraftHelper', 'width=1280,height=900')
})()
