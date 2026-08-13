// Service worker: does the actual work when the popup asks.
//
// Why the work lives here rather than in the popup or a content script:
//   - The popup window is destroyed the moment it loses focus, which can
//     abort an in-flight fetch. The service worker outlives it.
//   - A content script runs in ESPN's page and is subject to ESPN's CSP and
//     DOM changes. The service worker runs in the extension's own context
//     with our host_permissions, so it can call ESPN's API directly and is
//     not tied to ESPN's markup, which changes without notice.
//
// The extension's one advantage over a bookmarklet is used here: chrome.cookies
// can read the session cookies (including HttpOnly) given host permission, so
// there is nothing for the user to drag or paste. But it never SENDS espn_s2 --
// it uses it only to fetch a per-draft token from ESPN, and forwards that.

const APP = "http://localhost:8000";
const ESPN_API = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl";

// Pull the ids out of a draft-room URL. memberId carries the SWID directly,
// so we never have to read the SWID cookie -- one less credential handled.
function parseDraftUrl(url) {
  const league = (url.match(/leagueId=(\d+)/) || [])[1];
  const team = (url.match(/teamId=(\d+)/) || [])[1];
  const swid = (url.match(/memberId=(\{[^}]+\})/) || [])[1];
  const season = (url.match(/seasonId=(\d+)/) || [])[1]
    || String(new Date().getFullYear());
  return { league, team, swid, season };
}

async function readSwidCookie() {
  // Fallback when the URL has no memberId (e.g. the waiting-room URL). The
  // SWID cookie is not HttpOnly, but chrome.cookies reads it either way.
  const c = await chrome.cookies.get({
    url: "https://fantasy.espn.com", name: "SWID",
  });
  return c ? c.value : null;
}

async function syncDraft(tabUrl) {
  const { league, team, season } = parseDraftUrl(tabUrl);
  let { swid } = parseDraftUrl(tabUrl);
  if (!league || !team) {
    return { ok: false, error: "That tab is not an ESPN draft room. Open "
      + "your draft and click again." };
  }
  if (!swid) swid = await readSwidCookie();
  if (!swid) {
    return { ok: false, error: "Could not find your ESPN id. Make sure you "
      + "are signed in to ESPN in this browser." };
  }

  // The token fetch. credentials:"include" attaches espn_s2 automatically
  // because espn.com is in host_permissions. ESPN's CORS allows reading the
  // response back for the fantasy.espn.com origin (verified against the live
  // headers), which is the whole reason this works without a browser window.
  const tokenUrl = `${ESPN_API}/seasons/${season}/segments/0/leagues/`
    + `${league}/teams/${team}/draftSecurity`;
  let token;
  try {
    const r = await fetch(tokenUrl, {
      credentials: "include",
      headers: { "x-fantasy-source": "kona" },
    });
    if (!r.ok) {
      return { ok: false, error: r.status === 401 || r.status === 403
        ? "ESPN did not recognise your session. Sign in to ESPN and retry."
        : `ESPN returned ${r.status}. Is the draft actually open?` };
    }
    token = (await r.text()).trim();
  } catch (e) {
    return { ok: false, error: "Could not reach ESPN: " + e.message };
  }

  // Forward ONLY the throwaway token (plus the public ids). espn_s2 never
  // leaves the browser. host_permissions lets this POST to the app without a
  // CORS preflight dance.
  try {
    await fetch(`${APP}/api/live/token`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        leagueId: league, teamId: team, swid, token, season,
      }),
    });
  } catch (e) {
    return { ok: false, error: "Got the token, but the Draft Helper is not "
      + "running at " + APP + ". Start it and retry." };
  }
  return { ok: true, league, team };
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type === "SYNC") {
    syncDraft(msg.url).then(sendResponse);
    return true; // keep the message channel open for the async reply
  }
});
