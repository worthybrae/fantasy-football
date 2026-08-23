"""Keep the ESPN login out of anything this project prints.

WHY A MODULE AND NOT A CAREFUL CALL SITE. `SWID` is one of the two values
`draft_socket.load_cookies` treats as the saved login, and it is the SOLE
credential on the draft-socket handshake (see `run_socket_listener`: the
socket authenticates on `SWID` alone, espn_s2 was never required). It is also
carried in the query string of two URLs this project mints -- the mock
lobby's invite POST (`memberId=`) and the draft socket's own JOIN (`4=` and,
again, inside `5=`) -- so ANY exception that formats its request URL prints
the credential. httpx's `HTTPStatusError` does exactly that: `str(exc)` is
"Client error '400 Bad Request' for url 'https://...memberId={...}'".

That mattered enough to write down because of where those lines land.
`pipeline.mock_farm` runs unattended all night and its log is, in its own
docstring's words, the only account of that night anyone reads in the
morning -- which makes it the single most likely thing in the repo to be
pasted into a chat window or an issue. A room filling between the directory
read and the join is an ORDINARY race the farm expects and recovers from, so
the leak did not need anything to go wrong to fire.

The fix is deliberately not "remember to redact at the three call sites that
format an exception today". Call sites accumulate; one of them will forget.
So this module offers a scrubber and `redacting`, which wraps a print-like
callable ONCE at the top of a run -- after which no line printed anywhere
below it can carry the login, including a `traceback.format_exc()` nobody
thought about.

TWO LAYERS, because either alone has a hole:

  - PATTERNS. SWID is a brace-wrapped GUID, and the shapes it takes in a URL
    are enumerable (raw, percent-encoded, and named by parameter). This layer
    works in a process that never loaded a saved cookie at all -- api/live.py
    takes a stranger's swid off a pasted draft URL and never calls
    `load_cookies`.
  - THE LITERAL VALUE. `remember_secret` records the actual cookie the moment
    it is read, so a login whose shape is not one of the above -- ESPN
    changing the format, an account with an odd id -- is still scrubbed. This
    is the layer that makes "cannot reach the log" a fact about the value
    rather than a bet on a regex.

Redaction is IDEMPOTENT: every rule refuses to match its own placeholder, so
wrapping an already-wrapped `out` (which `play_draft` does when `farm` calls
it) cannot mangle a line into `memberId=<swid><swid>`.
"""
import re
from urllib.parse import quote

# What a scrubbed credential reads as. Angle brackets on purpose: they are
# excluded from every value-matching character class below, which is what
# makes re-redacting an already-redacted line a no-op.
SWID_PLACEHOLDER = "<swid>"
SECRET_PLACEHOLDER = "<redacted>"

# The GUID body ESPN's SWID has always had: 8-4-4-4-12 hex, dash separated.
_GUID = r"[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}"

_RULES = (
    # Percent-encoded, which is the form `espn_mock_lobby.invite_url` puts in
    # `memberId` -- the braces are not legal in a query string unescaped, so
    # `quote` turns them into %7B/%7D. Case-insensitive because nothing
    # guarantees which case a client emits the escape in.
    (re.compile(rf"(?i)%7B{_GUID}%7D"), SWID_PLACEHOLDER),
    # The cookie's own form, braces and all. This is what
    # `draft_socket.socket_url` puts in `4=` and inside `5=`, and what a
    # TOKEN frame carries.
    (re.compile(rf"\{{{_GUID}\}}"), SWID_PLACEHOLDER),
    # Bare, for anywhere the braces were stripped on the way through.
    (re.compile(_GUID), SWID_PLACEHOLDER),
    # By parameter name, for a value that is not GUID-shaped at all. `+`
    # rather than `*` so an already-redacted `memberId=<swid>` does not match
    # its own empty value and double up; `<` and `>` are out of the class for
    # the same reason. `&` and `;` are out because they SEPARATE values -- a
    # cookie header keeps its shape, it just loses the secret.
    (re.compile(r"(?i)\b(memberId|swid|espn_s2)=[^&;\s\"'<>]+"),
     lambda m: f"{m.group(1)}="
               + (SWID_PLACEHOLDER if m.group(1).lower() in ("memberid", "swid")
                  else SECRET_PLACEHOLDER)),
)

# Literal credential values seen this process, and their percent-encoded
# forms. A set rather than a single value because a process may hold more
# than one (the saved cookie plus a swid read off a pasted draft URL), and
# because registering the same one twice must be free.
_SECRETS: set = set()


def remember_secret(value) -> None:
    """Record a literal credential so `redact` scrubs it wherever it appears.

    Called from `draft_socket.load_cookies` -- the one place the saved login
    is read -- so every tool that authenticates gets this layer without
    asking for it. Short values are ignored: a one- or two-character secret
    is not a real credential, and scrubbing every occurrence of it would
    shred ordinary log lines.
    """
    text = str(value or "")
    if len(text) < 8:
        return
    _SECRETS.add(text)
    # The same value as it appears inside a URL. `invite_url` percent-encodes
    # with an empty `safe`, so that is the encoding to match.
    _SECRETS.add(quote(text, safe=""))


def redact(value) -> str:
    """`value` as text, with any credential in it replaced.

    Takes anything, not just a string, because the things printed here are
    exceptions and frames as often as they are messages, and a caller that
    has to remember to stringify first is a caller that can forget.
    """
    text = value if isinstance(value, str) else str(value)
    # Literals first: they are exact, and doing them before the patterns
    # means a secret that IS GUID-shaped is scrubbed the same way either way.
    for secret in _SECRETS:
        if secret and secret in text:
            text = text.replace(secret, SECRET_PLACEHOLDER)
    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)
    return text


def redacted_error(exc) -> str:
    """One line describing a failed HTTP call, safe to print.

    An httpx `HTTPStatusError` stringifies to a sentence wrapped around the
    whole request URL. What a reader of an overnight log actually needs from
    it is the status code -- 400 (the room filled) and 401 (the cookies
    expired) call for completely different mornings -- so this leads with
    that and appends the URL scrubbed, rather than printing ESPN's prose.

    Duck-typed on `response`/`request` rather than importing httpx: this
    module is imported by a login script that has no business pulling an
    HTTP client in, and anything else raising an exception with those two
    attributes is describing the same thing.
    """
    status = getattr(getattr(exc, "response", None), "status_code", None)
    url = (getattr(getattr(exc, "request", None), "url", None)
           or getattr(getattr(exc, "response", None), "url", None))
    if status is not None:
        where = f" for {redact(url)}" if url else ""
        return f"{type(exc).__name__}: HTTP {status}{where}"
    return f"{type(exc).__name__}: {redact(exc)}"


def redacting(out):
    """A print-like callable that scrubs everything handed to it.

    The whole point of the wrapper is that it is applied ONCE, at the top of
    a run, rather than at each call site: a future `out(...)` added anywhere
    below it is covered without its author having to know this module exists.
    """
    def write(*args):
        out(*(redact(a) for a in args))
    return write
