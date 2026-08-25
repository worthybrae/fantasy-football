/*
 * Draft Assistant bookmarklet -- readable source of truth.
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
 *   session cookie (espn_s2) automatically.
 *
 *   espn_s2 IS NOT HttpOnly, whatever this file used to say. Measured in a
 *   real browser against a real account: `document.cookie` on
 *   fantasy.espn.com returns it, and Chrome reports `httpOnly=false,
 *   sameSite=Lax`. The old comment here claimed the opposite and the security
 *   reasoning downstream was resting on it, which is why the correction is
 *   spelled out rather than quietly edited.
 *
 *   That is what makes "keep me signed in" possible at all (see below): the
 *   session can be handed to the helper, which then mints tokens for every
 *   later draft by itself. `sameSite=Lax` is the OTHER half of the wall and
 *   still stands -- the browser will not attach these cookies to a fetch from
 *   the helper's own origin, so a button on that page can never do this job
 *   and this bookmarklet is still the courier.
 *
 *   Then it opens Draft Assistant in a new window, passing what it found in
 *   the URL HASH -- which browsers do not send to servers, so none of it
 *   lands in an access log on the way in.
 *
 * WHAT TRAVELS, AND THE CHANGE OF POSTURE IT REPRESENTS. This used to send a
 * throwaway per-draft token and the public ids, and nothing else. It now also
 * sends `espn_s2`, the account session itself, because that is the difference
 * between a tool you click a bookmark for every single draft and one that
 * knows your leagues, joins them on a button, and re-mints its own token when
 * the old one dies mid-draft. The helper encrypts it, keys the row by an HMAC
 * of the SWID, hands this browser an httpOnly cookie as the only way back to
 * it, and forgets the whole thing after thirty days -- see
 * pipeline/credentials.py, which was written for this and has been waiting
 * for a caller.
 *
 * NO DRAFT ROOM REQUIRED any more either. On a draft-room URL this still
 * mints a token and joins that draft immediately. On any other ESPN page it
 * connects the account and lets the app list the drafts to join, which is
 * the difference between "find the right tab first" and "click it whenever
 * you think of it".
 *
 * APP IS NOT A CONSTANT IN THE SHIPPED COPY. This readable file hard-codes
 * localhost because it is what a developer runs; the string the install page
 * actually renders is built per request from `window.location.origin` (see
 * `bookmarkletFor` in web/src/lib/bookmarklet.ts).
 *
 * That is not a nicety. A baked-in localhost is right for exactly one person
 * -- whoever is running this repo -- and silently wrong for every visitor to
 * a deployed copy: their bookmark would open THEIR OWN localhost, where
 * nothing is listening, and the click would appear to do nothing at all. It
 * also decides where an ESPN session gets sent, and the only origin that may
 * ever receive one is the origin the user chose to install from.
 */
(async () => {
  const APP = 'http://localhost:5173'
  const ESPN = 'https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl'

  const u = location.href
  const league = (u.match(/leagueId=(\d+)/) || [])[1]
  const team = (u.match(/teamId=(\d+)/) || [])[1]
  // The account session. Readable here and nowhere else: this code runs on
  // ESPN's origin, and `sameSite=Lax` keeps the same cookie off every fetch
  // the helper's own page could make.
  const s2 = (document.cookie.match(/(?:^|;\s*)espn_s2=([^;]+)/) || [])[1]
  // memberId carries the SWID directly on a draft-room URL; fall back to the
  // SWID cookie otherwise. Neither ESPN cookie is HttpOnly (measured), so
  // document.cookie can read both.
  let swid = (u.match(/memberId=(\{[^}]+\})/) || [])[1]
  const season = (u.match(/seasonId=(\d+)/) || [])[1]
    || String(new Date().getFullYear())
  if (!swid) {
    const m = document.cookie.match(/SWID=([^;]+)/)
    swid = m ? decodeURIComponent(m[1]) : null
  }

  if (!swid) {
    alert("Couldn't find your ESPN id — make sure you're signed in to ESPN.")
    return
  }

  // NOT ON A DRAFT ROOM. Nothing to mint and nothing to join, but the session
  // is right here -- so hand it over and let the app show which of this
  // account's drafts can be joined. This is the ordinary path now; the one
  // below is the shortcut for somebody already sitting in the room.
  if (!league || !team) {
    if (!s2) {
      alert("Couldn't find your ESPN session — sign in to ESPN and try again.")
      return
    }
    const q = new URLSearchParams({ swid, s2, season })
    window.open(`${APP}/#${q.toString()}`, 'DraftAssistant', 'width=1280,height=900')
    return
  }

  // ESPN's draftSecurity endpoint mints the token the socket opens with.
  // credentials:'include' attaches espn_s2 automatically; ESPN's CORS allows
  // this origin to read the response back. The token is a bare integer body
  // -- and a SIGNED one: a real mock draft answered -1872384467.
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
  // Empty on a page where the cookie was not readable, which the app treats
  // as "join this draft and connect nothing" -- the behaviour this
  // bookmarklet had before it could offer more.
  if (s2) q.set('s2', s2)
  window.open(`${APP}/#${q.toString()}`, 'DraftAssistant', 'width=1280,height=900')
})()
