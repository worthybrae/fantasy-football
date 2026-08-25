// The Draft Assistant bookmarklet, as the exact `javascript:` string the connect
// screen renders for the user to drag to their bookmarks bar.
//
// Readable source of truth: /bookmarklet/bookmarklet.js. This is the shipped,
// comment-free, single-expression copy -- comment-free on purpose: some
// bookmark managers collapse newlines to spaces on save, which would turn any
// `//` line comment into "comment out the rest of the bookmarklet."
//
// Backslashes are doubled because this is a string literal: '\\d+' is the three
// characters \d+ at runtime, which is the regex the bookmarklet actually runs.
// Keep APP_BASE and BOOKMARKLET in step if this is ever pointed at a deployed
// app instead of local dev.
//
// WHAT IT CARRIES NOW: the account session (`espn_s2`) as well as the ids,
// because that is what lets the helper list your leagues and join them
// without another click. It is readable here and only here -- this runs on
// ESPN's origin, and `sameSite=Lax` keeps the same cookie off every fetch the
// helper's own page could make. On a draft-room URL it still mints a token
// and joins immediately; anywhere else on ESPN it connects the account and
// lets the app offer the drafts.

// WHERE THE BOOKMARKLET SENDS PEOPLE, and why it is not a constant.
//
// This used to bake in `http://localhost:5173`. That is correct for exactly
// one user -- whoever is running the repo -- and silently wrong for everybody
// else: a visitor dragging the button off a hosted page would install a
// bookmark that opens THEIR OWN localhost, where nothing is listening. The
// click would appear to do nothing at all.
//
// So the target is the origin the page was served from, resolved when the
// link is rendered. Drag it from localhost and it points at localhost; drag
// it from a deployed site and it points there. It also matters for what
// travels: the bookmarklet now carries an ESPN session, and the only origin
// that may ever receive one is the origin the user chose to install from.
export const APP_BASE = 'http://localhost:5173'

/** The `javascript:` string, aimed at `origin`.
 *
 *  `origin` is a bare scheme+host (`window.location.origin`), embedded as a
 *  JSON string literal so a stray quote in it could not close the attribute
 *  the link is rendered into. */
export function bookmarkletFor(origin: string): string {
  return BOOKMARKLET_TEMPLATE.replace('__APP__', JSON.stringify(origin || APP_BASE))
}

const BOOKMARKLET_TEMPLATE =
  'javascript:(async()=>{' +
  'var A=__APP__,' +
  'E="https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl",' +
  'u=location.href,' +
  'l=(u.match(/leagueId=(\\d+)/)||[])[1],' +
  't=(u.match(/teamId=(\\d+)/)||[])[1],' +
  's=(u.match(/memberId=(\\{[^}]+\\})/)||[])[1],' +
  'y=(u.match(/seasonId=(\\d+)/)||[])[1]||String(new Date().getFullYear());' +
  'if(!s){var m=document.cookie.match(/SWID=([^;]+)/);s=m?decodeURIComponent(m[1]):null}' +
  'var c=(document.cookie.match(/(?:^|;\\s*)espn_s2=([^;]+)/)||[])[1];' +
  'if(!s){alert("Cannot find your ESPN id. Make sure you are signed in to ESPN.");return}' +
  'if(!l||!t){if(!c){alert("Cannot find your ESPN session. Sign in to ESPN and try again.");return}' +
  'window.open(A+"/#"+new URLSearchParams({swid:s,s2:c,season:y}).toString(),' +
  '"DraftAssistant","width=1280,height=900");return}' +
  'var k;try{' +
  'var r=await fetch(E+"/seasons/"+y+"/segments/0/leagues/"+l+"/teams/"+t+"/draftSecurity",' +
  '{credentials:"include",headers:{"x-fantasy-source":"kona"}});' +
  'if(!r.ok){alert(r.status===401||r.status===403?' +
  '"ESPN did not recognise your session. Sign in to ESPN and retry.":' +
  '"ESPN returned "+r.status+". Is the draft actually open?");return}' +
  'k=(await r.text()).trim()' +
  '}catch(e){alert("Cannot reach ESPN: "+e.message);return}' +
  'var q=new URLSearchParams({leagueId:l,teamId:t,swid:s,token:k,season:y});' +
  'if(c)q.set("s2",c);' +
  'window.open(A+"/#"+q.toString(),"DraftAssistant","width=1280,height=900")' +
  '})()'

/** The local-development spelling, kept so a reader can eyeball the shipped
 *  string and so the readable source in /bookmarklet stays comparable. Every
 *  caller in the app uses `bookmarkletFor(window.location.origin)`. */
export const BOOKMARKLET = bookmarkletFor(APP_BASE)
