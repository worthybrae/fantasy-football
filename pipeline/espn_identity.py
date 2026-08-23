"""Proving that whoever sent us an ESPN session actually owns the account.

WHY THIS MODULE EXISTS. `SWID` is public. It rides in the mock lobby's invite
POST query string and in the draft socket's JOIN url, so anyone who has seen a
user's draft link has it (`pipeline/redact.py`'s docstring is a catalogue of
the places it escapes). A credential store that takes `{swid, espn_s2}` from a
request and writes the row keyed on the CLIENT'S `swid` field therefore has no
authorization on its write path at all: a stranger who knows a victim's SWID
can post their own `espn_s2` under it and silently replace the victim's stored
session. The victim's browser then holds a cookie pointing at the ATTACKER'S
ESPN account, and the first feature that acts on a resolved credential acts on
the wrong account entirely.

So the store never keys a row on what the client claimed. It keys on what ESPN
says the presented cookie belongs to, which is what this module goes and asks.

THE ENDPOINT. `fan.api.espn.com/apis/v2/fans/{SWID}` is the account profile.
It is the one ESPN endpoint that answers for the account rather than for a
league, and it carries no email or name (verified against a real account:
`dmaId`, a creation date, and insider/premium flags), which is exactly why
this project cannot contact its users -- see `pipeline/credentials.py`.

TWO REQUESTS, NOT ONE, AND THE SECOND IS THE POINT. The obvious check is "GET
the profile with these cookies, confirm the id comes back". That check is only
worth something if the endpoint REFUSES the same request without cookies -- if
it happens to be public, an attacker's request would sail through it, because
the id in the response is just the id from the url. Rather than assume, this
module measures: it repeats the request with no cookies, and if the anonymous
attempt returns the same account, the endpoint has proved nothing and
verification FAILS CLOSED with a message saying so. That turns an assumption
about ESPN's behaviour, which nobody here can hold still, into a runtime check
that breaks loudly if it ever stops being true.

`fetch` is injectable for exactly one reason: so the whole of this logic is
testable with no network, no ESPN, and no real credential.
"""
from __future__ import annotations

import json
import re
from urllib.parse import quote, unquote

from pipeline import redact

FAN_BASE = "https://fan.api.espn.com/apis/v2/fans"

# Same shape as everywhere else ESPN publishes it: 8-4-4-4-12 hex.
_GUID = re.compile(r"^[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}$")


class OwnershipUnproven(RuntimeError):
    """The presented cookie was not shown to belong to the claimed account.

    Raised for every distinguishable cause -- ESPN refused, ESPN answered for
    somebody else, ESPN could not be reached, the endpoint turned out to
    authenticate nothing -- because the caller's response is the same in all
    of them (refuse to store), and a caller that could tell them apart could
    be used to probe which SWIDs exist.
    """


def canonical_swid(value) -> str:
    """One spelling per account, so one account is one row.

    ESPN, its own URLs, a pasted draft link and a browser cookie jar between
    them produce `{GUID}`, `GUID`, `%7BGUID%7D`, lower case, upper case, and
    the occasional trailing space. Hashed as-is that is five row keys for one
    person, each holding a live ESPN session, and a "disconnect everywhere"
    that clears exactly one of them. So every SWID is canonicalised to
    `{UPPERCASE-GUID}` before it is hashed or stored.

    A value that is not GUID-shaped is stripped and returned unchanged rather
    than forced into a shape it does not have: ESPN changing its id format
    should degrade to "one spelling, exactly as sent", not to a mangled key.
    """
    text = unquote(str(value or "")).strip()
    body = text.strip("{}").strip()
    if _GUID.match(body):
        return "{" + body.upper() + "}"
    return text


def fan_url(swid: str) -> str:
    """The profile url for one account.

    The SWID goes in the PATH, braces percent-encoded, which is also why any
    exception carrying this url carries a credential-shaped value -- see
    `redact`, which already scrubs `%7B...%7D`.
    """
    return f"{FAN_BASE}/{quote(canonical_swid(swid), safe='')}"


def http_fetch():
    """A `fetch(url, cookies) -> (status, text)` callable hitting ESPN.

    Mirrors `pipeline.espn_teams.http_fetch` and `pipeline.draft_socket`'s,
    but takes its cookies per call rather than closing over the one saved
    login: this module is asked about a DIFFERENT person's session on every
    request, and a fetcher that quietly attached the owner's own cookies would
    verify the owner's account every time and approve every attacker.

    Never raises for a status: a 401 is data here (it is the ordinary answer
    for a session that has expired), so the status is returned and the caller
    decides. A transport failure does raise, and `verify_account` turns it
    into a refusal.
    """
    import httpx

    def fetch(url: str, cookies: dict | None):
        response = httpx.get(url, cookies=cookies or {}, timeout=10.0,
                             headers={"User-Agent": "Mozilla/5.0"})
        return response.status_code, response.text
    return fetch


def _reported_id(body: str):
    """The account id in a fan-profile body, or None if it does not carry one.

    Tolerant about WHERE the id sits (`id`, `guid`, `swid`, or a nested
    `profile`) because this is a read of somebody else's JSON and the shape is
    not ours to pin -- but strict about there being one, since "no id" must
    never read as "verified".
    """
    try:
        payload = json.loads(body or "")
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    for holder in (payload, payload.get("profile") or {}):
        if not isinstance(holder, dict):
            continue
        for field in ("id", "guid", "swid", "SWID"):
            value = holder.get(field)
            if isinstance(value, str) and value.strip():
                return canonical_swid(value)
    return None


def verify_account(swid: str, espn_s2: str, fetch=None) -> str:
    """The account ESPN attributes to this cookie. Raises if it cannot say.

    Returns the CANONICAL SWID, and the caller is expected to key its storage
    on the return value rather than on what it was handed -- the whole point
    is that the client's claim is not evidence.

    Fails closed on every uncertainty, including ESPN being unreachable. This
    runs on a write path that stores a full account session for thirty days;
    "we could not check, so we stored it" is not an available answer.
    """
    fetch = fetch or http_fetch()
    claimed = canonical_swid(swid)
    if not claimed or not espn_s2:
        raise OwnershipUnproven("no account was named")
    url = fan_url(claimed)

    try:
        status, body = fetch(url, {"espn_s2": str(espn_s2), "SWID": claimed})
    except Exception as exc:      # noqa: BLE001 -- any transport failure
        # `redacted_error` and not `exc`: the url this failed on has the SWID
        # in its path, and httpx wraps the whole url in its message.
        raise OwnershipUnproven(
            f"ESPN could not be reached to verify the account "
            f"({redact.redacted_error(exc)})") from None
    if status != 200:
        raise OwnershipUnproven(
            f"ESPN did not accept that session (HTTP {status})")
    reported = _reported_id(body)
    if reported is None:
        raise OwnershipUnproven("ESPN's answer named no account")
    if reported != claimed:
        # The interesting failure: the cookie works, for somebody else.
        raise OwnershipUnproven("that session belongs to a different account")

    # THE CONTROL. If the same request answers identically with no cookies at
    # all, then the first request proved only that the url was well formed.
    try:
        anon_status, anon_body = fetch(url, None)
    except Exception:      # noqa: BLE001 -- unreachable anonymously is fine:
        # it is evidence the endpoint is not open, which is what we wanted.
        anon_status, anon_body = 599, ""
    if anon_status == 200 and _reported_id(anon_body) == claimed:
        raise OwnershipUnproven(
            "ESPN's account endpoint answered without a session, so it cannot "
            "prove who owns this one -- refusing to store the credential")
    return reported
