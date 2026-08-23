# Holding other people's ESPN sessions

**Design, 2026-08-23.** The landing page sells "$4.99 a real draft", which
means strangers connect their ESPN accounts to a server this project runs.
This is the design for holding those credentials without the holding becoming
the product's biggest liability.

## What is actually being stored

`espn_s2` plus `SWID` is a full ESPN session cookie pair. It is not scoped to
fantasy, cannot be limited to read-only, and ESPN offers no per-application
revocation -- a user's only remedy is to log out of ESPN everywhere. It is
long-lived, measured in months.

The long life is the reason to be careful, not the reason to relax. A
short-lived token in a leaked database is close to worthless by the time
anyone reads it. These stay valid.

## What we do not get

ESPN's fan API (`fan.api.espn.com/apis/v2/fans/{SWID}`) returns no email and
no name -- only `dmaId`, account creation date, and insider/premium flags,
verified against a real account. So:

- there is no way to contact a user, ever;
- there is no recovery path if they lose their browser session;
- a stored credential nobody comes back for can only be reaped by a clock.

Every decision below follows from those three facts.

## Identity

`SWID` identifies the user. It is NOT a secret: it rides in the invite POST's
query string, in the draft socket's JOIN url, and in anything either gets
pasted into. A server that returns a stored ESPN session to whoever presents a
SWID is an account-takeover endpoint.

So SWID is the row key and a browser-held secret is the authorization. SWID is
the username; an opaque, `httpOnly` session cookie is the password.

## Storage

Two tables.

**`espn_credential`** -- one row per ESPN user.

| column | contents |
|---|---|
| `id` | `HMAC-SHA256(swid, KEY)` -- the lookup key. The raw SWID is never a column. |
| `blob` | Fernet-encrypted `{"swid": ..., "espn_s2": ...}` |
| `key_version` | integer, so the key can be rotated without a flag day |
| `created_at`, `last_used_at`, `expires_at` | `expires_at` defaults to 30 days after `last_used_at` |

**`espn_session`** -- one row per browser.

| column | contents |
|---|---|
| `id` | `HMAC-SHA256(cookie_value, KEY)` -- the raw cookie is never stored |
| `credential_id` | FK to `espn_credential.id`, cascading delete |
| `created_at`, `last_used_at`, `expires_at` | |

Two tables rather than one because a user connecting from a second browser
must not silently evict the first.

`KEY` lives in the environment and is never written to the database. The
consequence that matters: a stolen database file is inert. Not "tokens
encrypted, user list readable" -- there is no readable user list, because the
SWIDs are inside the encrypted blob and the row keys are HMACs of them.

**Dependency.** This adds `cryptography` to `requirements.txt`, the first new
dependency this project has taken. Rolling our own AEAD to avoid it would be
a worse trade by a wide margin.

## Connect

The bookmarklet already POSTs `{leagueId, teamId, swid, token}` to
`/api/live/connect-token`. That endpoint gains three jobs:

1. encrypt `{swid, espn_s2}` and upsert the credential row;
2. mint a 256-bit opaque session value, set it `httpOnly; Secure;
   SameSite=Lax`, and store only its HMAC;
3. never echo the token -- not in the response body, not in an error, not in
   a log.

**Plain HTTP fails closed.** Both the token and the session cookie cross the
wire. If the request is not TLS the endpoint refuses rather than degrading: a
`Secure` cookie silently not being set is precisely the failure nobody
notices until it matters. Local development gets an explicit, separately
named env opt-out, so the switch that makes dev work cannot be the switch
that disables the check in production.

## Use, and deletion

Every ESPN call on a user's behalf resolves cookie -> session -> credential,
decrypts in memory, and passes through `pipeline/redact.py` before anything
reaches a log. `redact` already exists and already registers the live SWID and
`espn_s2`; every new log path must go through it.

- **Disconnect** deletes the session row. The browser is logged out.
- **Disconnect everywhere** deletes the credential row and cascades.
- **A 401 from ESPN deletes the credential immediately.** Keeping a
  known-dead session on disk has no upside and is pure liability.
- **`expires_at` is the only reaper** for a user who clears cookies and never
  returns. Given there is no email, without it the database accumulates
  credentials belonging to people who cannot be reached.

## What the user is told

Both, because they serve different readers at different moments.

**At connect, one plain sentence** where it is actually read: what is stored,
that it is encrypted, what it allows, how long it is kept, and a visible
disconnect control. No modal, no legalese.

**A policy page** linked from the connect screen and from checkout, for
anyone who wants the detail -- and because charging money for a service that
holds session credentials without a written position is not defensible.

## Testing

The tests that matter here are the breach-class ones, and they are the point
of this section rather than an afterthought:

- a database dump plus the code, WITHOUT the key, yields nothing: no SWIDs,
  no tokens, no enumerable user list
- a request with no cookie is refused; a request with a well-formed but wrong
  cookie is refused; a cookie belonging to a deleted session is refused
- presenting a valid SWID with no cookie is refused -- the account-takeover
  case this design exists to prevent
- disconnect deletes; disconnect-everywhere cascades; a 401 deletes
- the TTL reaps an idle credential
- plain HTTP is refused unless the dev opt-out is explicitly set
- every new log path is covered by `redact` -- assert against a real
  exception carrying a token in a URL, the way the existing redact tests do

## Order of work

The credential-free lobby widget (`GET /api/lobby`, a live strip on the
landing page) is independent of all of this, useful immediately, and commits
us to nothing. It ships first.

This subsystem ships second, backend before frontend, because the disclosure
copy should describe what the code actually does rather than the reverse.
