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

export const APP_BASE = 'http://localhost:5173'

export const BOOKMARKLET =
  'javascript:(async()=>{' +
  'var A="' + APP_BASE + '",' +
  'E="https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl",' +
  'u=location.href,' +
  'l=(u.match(/leagueId=(\\d+)/)||[])[1],' +
  't=(u.match(/teamId=(\\d+)/)||[])[1],' +
  's=(u.match(/memberId=(\\{[^}]+\\})/)||[])[1],' +
  'y=(u.match(/seasonId=(\\d+)/)||[])[1]||String(new Date().getFullYear());' +
  'if(!s){var m=document.cookie.match(/SWID=([^;]+)/);s=m?decodeURIComponent(m[1]):null}' +
  'if(!l||!t){alert("Open your ESPN draft room first, then click Draft Assistant.");return}' +
  'if(!s){alert("Cannot find your ESPN id. Make sure you are signed in to ESPN.");return}' +
  'var k;try{' +
  'var r=await fetch(E+"/seasons/"+y+"/segments/0/leagues/"+l+"/teams/"+t+"/draftSecurity",' +
  '{credentials:"include",headers:{"x-fantasy-source":"kona"}});' +
  'if(!r.ok){alert(r.status===401||r.status===403?' +
  '"ESPN did not recognise your session. Sign in to ESPN and retry.":' +
  '"ESPN returned "+r.status+". Is the draft actually open?");return}' +
  'k=(await r.text()).trim()' +
  '}catch(e){alert("Cannot reach ESPN: "+e.message);return}' +
  'var q=new URLSearchParams({leagueId:l,teamId:t,swid:s,token:k,season:y});' +
  'window.open(A+"/#"+q.toString(),"DraftAssistant","width=1280,height=900")' +
  '})()'
