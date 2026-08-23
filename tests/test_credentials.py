"""The breach-class tests for `pipeline/credentials.py`.

These are not a formality around a storage layer. The subsystem holds full
ESPN account sessions belonging to people this project has no way to contact
-- ESPN's fan API returns no email -- so there is no version of a leak here
that ends in an apology and a password reset. What is testable in advance is
the shape of the damage, and that is what every test below asserts:

  * a stolen database file, plus this repository, minus the key, is inert;
  * a SWID is not a password, no matter how it is presented;
  * everything that should delete a credential does, promptly;
  * nothing keeps one longer than the clock allows.

Entirely offline. The only ESPN artefacts are invented strings chosen to be
unmistakable in a hex dump (see FAKE_SWID / FAKE_S2), and nothing here opens a
socket, a browser, or the repository's real database.
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from pipeline import credentials as cred
from pipeline import redact

# Deliberately shaped like the real things -- a brace-wrapped GUID and a long
# opaque cookie -- so that a test asserting "this string is not in the file"
# is asserting something about the storage rather than about the fixture being
# too short to find. Both are invented; neither has ever been a real session.
FAKE_SWID = "{7A1F9C34-BEEF-4D01-9A55-C0FFEE001122}"
FAKE_S2 = "AEBz" + "QqcustodyFIXTUREnotarealsessionvalue" * 3 + "%2Fend"

# A second, different account, for the tests that need to prove one user's
# row cannot be reached with another user's material.
OTHER_SWID = "{11112222-3333-4444-5555-666677778888}"
OTHER_S2 = "AEBz" + "OTHERcustodyFIXTUREvalue" * 3


def _key():
    return cred.generate_key()


@pytest.fixture
def store(tmp_path):
    """A store on a throwaway file with an explicit, throwaway key.

    Explicit rather than environment-driven: these tests mint and rotate keys,
    and a test that has to mutate `os.environ` to do that can leak the mutation
    into the next one. `close_all` afterwards because DuckDB holds a file lock
    and the inert-dump tests read the file's raw bytes.
    """
    made = cred.CredentialStore(path=str(tmp_path / "custody.duckdb"),
                                keys=f"1:{_key()}", out=lambda *a: None)
    yield made
    cred.close_all()


def _files(store):
    """Every byte DuckDB wrote, database and write-ahead log alike.

    The WAL matters: a row that has not been checkpointed yet lives there in
    full, so checking only the .duckdb file would be a test that passes for
    the wrong reason.
    """
    base = Path(store.path)
    return [p for p in base.parent.iterdir()
            if p.name.startswith(base.name)]


def _raw_bytes(store):
    cred.close_all()
    blob = b""
    for path in _files(store):
        blob += path.read_bytes()
    return blob


# --- The one property this module exists to provide --------------------------

def test_a_stolen_database_without_the_key_yields_nothing_usable(store):
    """Database file + this repository + no key = nothing.

    The three things an attacker would want, checked against the real bytes on
    disk rather than against query results (a query is what the code does; the
    bytes are what the thief has):

      1. the SWIDs -- absent, because they are inside the encrypted blob;
      2. the espn_s2 tokens -- absent, same reason;
      3. an enumerable user list -- absent, because the row keys are HMACs
         under a key that is not in the file. An attacker who already HOLDS a
         SWID (they are public; they ride in draft URLs) cannot even test it
         for membership, which is the difference between "encrypted store" and
         "encrypted store that still tells you who its customers are".
    """
    store.connect(FAKE_SWID, FAKE_S2)
    store.connect(OTHER_SWID, OTHER_S2)
    raw = _raw_bytes(store)

    for secret in (FAKE_SWID, FAKE_S2, OTHER_SWID, OTHER_S2):
        assert secret.encode() not in raw, "a credential is readable on disk"
    # The GUID body alone, in case the braces were ever stripped on the way in.
    assert FAKE_SWID.strip("{}").encode() not in raw

    # An attacker with the file, the code, and a candidate SWID. They can
    # compute any unkeyed digest they like; none of them is the row key.
    import hashlib
    for guess in (FAKE_SWID, FAKE_SWID.strip("{}"), FAKE_SWID.lower()):
        assert hashlib.sha256(guess.encode()).hexdigest().encode() not in raw
    # And with a key of their own, which is the closest they can get: HMAC is
    # keyed, so a wrong key produces an id that matches nothing.
    thief = cred.CredentialStore(path=store.path, keys=f"1:{_key()}",
                                 out=lambda *a: None)
    assert thief.resolve(FAKE_SWID) is None
    thief_id = thief._row_id(FAKE_SWID, thief.current_key)
    assert thief_id.encode() not in raw


def test_the_columns_themselves_carry_no_plaintext(store):
    """Every stored value is an HMAC, a Fernet token, or a timestamp.

    A separate assertion from the byte scan above, and worth making: the byte
    scan proves the fixtures are absent, while this proves the SHAPE is right,
    so a future column added in plaintext fails here rather than passing
    quietly because nobody's fixture happened to land in it.
    """
    store.connect(FAKE_SWID, FAKE_S2)
    conn = cred._connect(store.path)
    for row in conn.execute("SELECT * FROM espn_credential").fetchall():
        row_id, blob, version, *stamps = row
        assert len(row_id) == 64 and int(row_id, 16) >= 0   # hex HMAC
        assert blob.startswith("gAAAAA")                    # Fernet, versioned
        assert isinstance(version, int)
        assert all(isinstance(s, datetime) for s in stamps)
    for row in conn.execute("SELECT * FROM espn_session").fetchall():
        row_id, credential_id, version, *stamps = row
        assert len(row_id) == 64 and len(credential_id) == 64
        assert int(row_id, 16) >= 0 and int(credential_id, 16) >= 0


def test_the_key_is_never_written_beside_the_data(store):
    """The whole inert-dump claim reduces to this: the key is not in the
    thing that gets stolen. Asserted directly, because it is the assumption
    every other test in this file rests on."""
    material = store.keys[1]
    store.connect(FAKE_SWID, FAKE_S2)
    raw = _raw_bytes(store)
    assert material.index_key not in raw
    assert store._keys_spec.split(":", 1)[1].encode() not in raw


def test_a_missing_key_is_a_refusal_and_never_a_plaintext_fallback(tmp_path):
    """A deployment that forgot to set the key must break, loudly.

    The tempting alternative -- store it in the clear and warn -- would mean a
    misconfiguration silently produces exactly the file this whole design
    exists to make worthless.
    """
    broken = cred.CredentialStore(path=str(tmp_path / "c.duckdb"), keys="",
                                  out=lambda *a: None)
    with pytest.raises(cred.CustodyUnavailable):
        broken.connect(FAKE_SWID, FAKE_S2)
    with pytest.raises(cred.CustodyUnavailable):
        broken.resolve("anything")
    assert not (tmp_path / "c.duckdb").exists(), \
        "a keyless store created a database file anyway"


# --- Authorization: the cookie, and only the cookie --------------------------

def test_a_valid_swid_with_no_cookie_is_refused(store):
    """THE ACCOUNT-TAKEOVER CASE, and the reason the design is shaped this way.

    SWID is public: it rides in the mock lobby's invite POST query string and
    in the draft socket's JOIN url, so anyone who has seen a user's draft link
    has it. Presenting it -- in any form, as if it were the cookie -- must get
    nothing back, and there must be no other call that takes it either.
    """
    minted = store.connect(FAKE_SWID, FAKE_S2)
    assert store.resolve(minted.cookie) is not None      # control

    for shape in (FAKE_SWID, FAKE_SWID.strip("{}"), FAKE_SWID.lower(),
                  minted.credential_id):
        assert store.resolve(shape) is None, \
            "an identifier was accepted as authorization"

    # And there is no swid-keyed lookup to find. This is a real assertion, not
    # a stylistic one: the failure mode is somebody adding a convenience
    # `resolve_swid` later, and this is what tells them not to.
    assert not [name for name in dir(store)
                if "swid" in name.lower() and not name.startswith("__")]


def test_no_cookie_a_wrong_cookie_and_a_dead_cookie_are_all_refused(store):
    """The three ways a request arrives without a valid session.

    All three return None rather than distinguishing themselves. A caller that
    cannot tell "never existed" from "was deleted" cannot be turned into an
    oracle for which sessions this server holds.
    """
    minted = store.connect(FAKE_SWID, FAKE_S2)

    assert store.resolve("") is None
    assert store.resolve(None) is None
    # Well-formed but wrong: the right length, the right alphabet, not ours.
    import secrets as _secrets
    assert store.resolve(_secrets.token_urlsafe(32)) is None
    # A cookie whose session has been deleted.
    assert store.disconnect(minted.cookie) is True
    assert store.resolve(minted.cookie) is None
    # And one whose credential has been deleted out from under it.
    again = store.connect(FAKE_SWID, FAKE_S2)
    store.forget_credential(again.credential_id)
    assert store.resolve(again.cookie) is None


def test_one_users_cookie_never_reaches_another_users_credential(store):
    """Two accounts, two cookies, no crossover."""
    mine = store.connect(FAKE_SWID, FAKE_S2)
    theirs = store.connect(OTHER_SWID, OTHER_S2)
    assert store.resolve(mine.cookie).swid == FAKE_SWID
    assert store.resolve(theirs.cookie).swid == OTHER_SWID
    assert mine.credential_id != theirs.credential_id


def test_the_stored_session_comes_back_exactly_as_it_went_in(store):
    """The round trip, so that "encrypted" does not quietly become
    "encrypted and subtly mangled" -- an espn_s2 that survives storage with
    its percent-encoding altered is a credential that no longer works."""
    minted = store.connect(FAKE_SWID, FAKE_S2)
    resolved = store.resolve(minted.cookie)
    assert resolved.swid == FAKE_SWID
    assert resolved.espn_s2 == FAKE_S2


# --- Two tables, so a second browser does not evict the first ----------------

def test_a_second_browser_does_not_log_the_first_one_out(store):
    """The entire reason `espn_session` is a separate table. The first browser
    is frequently the one with the draft open."""
    first = store.connect(FAKE_SWID, FAKE_S2)
    second = store.connect(FAKE_SWID, FAKE_S2)
    assert first.cookie != second.cookie
    assert first.credential_id == second.credential_id   # one user, one row
    assert store.resolve(first.cookie) is not None
    assert store.resolve(second.cookie) is not None
    assert store.counts() == {"credentials": 1, "sessions": 2}


def test_disconnect_ends_one_browser_and_disconnect_everywhere_cascades(store):
    """The two ways out, and the difference between them."""
    first = store.connect(FAKE_SWID, FAKE_S2)
    second = store.connect(FAKE_SWID, FAKE_S2)

    assert store.disconnect(first.cookie) is True
    assert store.resolve(first.cookie) is None
    assert store.resolve(second.cookie) is not None, \
        "disconnecting one browser logged another one out"
    assert store.counts() == {"credentials": 1, "sessions": 1}

    third = store.connect(FAKE_SWID, FAKE_S2)
    assert store.disconnect_everywhere(second.cookie) is True
    assert store.resolve(second.cookie) is None
    assert store.resolve(third.cookie) is None, "the cascade missed a browser"
    assert store.counts() == {"credentials": 0, "sessions": 0}


def test_disconnect_everywhere_needs_a_cookie_of_its_own(store):
    """It is a deletion, so it is authorised like every other read: by the
    cookie. A SWID must not be able to log anybody out either."""
    store.connect(FAKE_SWID, FAKE_S2)
    assert store.disconnect_everywhere(FAKE_SWID) is False
    assert store.counts()["credentials"] == 1


# --- A 401 is a deletion, not a retry ----------------------------------------

class _FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


class _FakeRequest:
    def __init__(self, url):
        self.url = url


class _FakeHTTPStatusError(Exception):
    """Shaped like httpx's, which is what an ESPN call would actually raise.

    Its `str()` is the dangerous part and is copied faithfully: httpx wraps
    the WHOLE request url in its message, and a request made on a stored
    credential's behalf carries both the SWID and espn_s2 in that url.
    """

    def __init__(self, status_code, url):
        self.response = _FakeResponse(status_code)
        self.request = _FakeRequest(url)
        super().__init__(
            f"Client error '{status_code} Unauthorized' for url '{url}'")


def _espn_url():
    return ("https://fantasy.espn.com/apis/v3/games/ffl/seasons/2026/"
            f"segments/0/leagues/1?memberId={FAKE_SWID}&espn_s2={FAKE_S2}")


def test_a_401_from_espn_deletes_the_credential(store):
    """No retry, no backoff, no keeping it around in case.

    A 401 means ESPN has already invalidated the session; there is no refresh
    flow that could revive it. Keeping the row would mean holding a known-dead
    account session on disk, which is liability with the upside removed.
    """
    minted = store.connect(FAKE_SWID, FAKE_S2)
    exc = _FakeHTTPStatusError(401, _espn_url())

    assert store.forget_if_unauthorized(minted.credential_id, exc) is True
    assert store.resolve(minted.cookie) is None
    assert store.counts() == {"credentials": 0, "sessions": 0}


def test_any_other_status_leaves_the_credential_alone(store):
    """A 500 or a 429 is ESPN having a bad minute. Deleting a working session
    over one would mean a user has to reconnect because a server hiccuped, and
    they would have no idea why."""
    minted = store.connect(FAKE_SWID, FAKE_S2)
    for status in (400, 403, 429, 500, 503):
        exc = _FakeHTTPStatusError(status, _espn_url())
        assert store.forget_if_unauthorized(minted.credential_id, exc) is False
    assert store.resolve(minted.cookie) is not None


# --- The clock is the only other reaper --------------------------------------

def test_the_ttl_reaps_an_idle_credential(store):
    """The only thing that ever removes a credential nobody comes back for.

    There is no email, so there is no "we noticed you have not been around"
    message and no way to ask. Without this the table would accumulate live
    account sessions belonging to people who cannot be reached, permanently.
    """
    start = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)
    minted = store.connect(FAKE_SWID, FAKE_S2, now=start)

    just_inside = start + timedelta(days=cred.DEFAULT_TTL_DAYS) - timedelta(hours=1)
    assert store.resolve(minted.cookie, now=just_inside) is not None

    # ...and that resolve just slid the window, so expiry is now measured from
    # it. Idle, not absolute: an active user is never logged out by the TTL.
    past = just_inside + timedelta(days=cred.DEFAULT_TTL_DAYS, hours=1)
    assert store.reap(now=past) == {"credentials": 1, "sessions": 1}
    assert store.resolve(minted.cookie) is None
    assert store.counts() == {"credentials": 0, "sessions": 0}


def test_an_expired_credential_cannot_be_resolved_even_if_nobody_reaped(store):
    """The TTL must not depend on a scheduled job somebody remembered to
    write. `resolve` reaps first, so the expiry is enforced by the read path
    itself and a deployment with no cron still honours it."""
    start = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)
    minted = store.connect(FAKE_SWID, FAKE_S2, now=start)
    later = start + timedelta(days=cred.DEFAULT_TTL_DAYS + 1)
    assert store.resolve(minted.cookie, now=later) is None
    assert store.counts() == {"credentials": 0, "sessions": 0}


def test_an_active_credential_is_never_reaped(store):
    """The counterpart, so the reaper cannot be made safe by being too keen."""
    start = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)
    minted = store.connect(FAKE_SWID, FAKE_S2, now=start)
    for day in range(0, 90, 7):
        assert store.resolve(minted.cookie,
                             now=start + timedelta(days=day)) is not None
    assert store.reap(now=start + timedelta(days=90)) == {
        "credentials": 0, "sessions": 0}


def test_a_lone_idle_browser_is_reaped_without_taking_the_account_with_it(store):
    """Sessions expire on their own clock too: one browser going quiet logs
    that browser out, it does not delete the credential another browser is
    still using."""
    start = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)
    idle = store.connect(FAKE_SWID, FAKE_S2, now=start)
    active = store.connect(FAKE_SWID, FAKE_S2, now=start)
    for day in range(0, 60, 7):
        store.resolve(active.cookie, now=start + timedelta(days=day))
    assert store.resolve(idle.cookie) is None
    assert store.resolve(active.cookie) is not None


# --- Rotation ----------------------------------------------------------------

def test_a_row_written_under_key_version_1_survives_version_2(tmp_path):
    """Rotation without a flag day, which is the only kind that gets done.

    A row records the version it was written under; lookups try every known
    version; new writes use the highest. So adding a key is a deploy, and the
    old rows keep working until their owners next connect -- at which point
    they move to the new key, taking their other browsers with them rather
    than being evicted.
    """
    path = str(tmp_path / "custody.duckdb")
    key1, key2 = _key(), _key()

    old = cred.CredentialStore(path=path, keys=f"1:{key1}", out=lambda *a: None)
    first_browser = old.connect(FAKE_SWID, FAKE_S2)
    second_browser = old.connect(FAKE_SWID, FAKE_S2)

    # Version 2 arrives. Nothing has been rewritten.
    rotated = cred.CredentialStore(path=path, keys=f"1:{key1},2:{key2}",
                                   out=lambda *a: None)
    assert rotated.current_key.version == 2
    resolved = rotated.resolve(first_browser.cookie)
    assert resolved is not None, "a version-1 row stopped resolving"
    assert resolved.swid == FAKE_SWID and resolved.espn_s2 == FAKE_S2

    conn = cred._connect(path)
    assert conn.execute(
        "SELECT key_version FROM espn_credential").fetchone()[0] == 1

    # The user reconnects. Their row moves to version 2 -- and their OTHER
    # browser, whose session id was HMAC-ed under version 1, still works.
    rotated.connect(FAKE_SWID, FAKE_S2)
    assert conn.execute(
        "SELECT key_version FROM espn_credential").fetchone()[0] == 2
    assert conn.execute("SELECT count(*) FROM espn_credential").fetchone()[0] == 1
    assert rotated.resolve(second_browser.cookie) is not None, \
        "rotation orphaned a browser it should have repointed"
    cred.close_all()


def test_retiring_a_key_makes_its_rows_unreadable_rather_than_dangerous(tmp_path):
    """Completing a rotation -- removing the old key -- is what actually ends
    the old key's blast radius.

    Two halves, and the second is the one worth having a test for.

    A key that is simply gone takes its rows out of reach entirely: the SESSION
    ids were HMAC-ed under it too, so the cookie no longer finds anything and
    there is nothing to report. That is the ordinary end of a rotation.

    Rolling one BACK is different, and it is the case that can actually be hit
    by accident (a bad key deploy reverted). The session is found under the key
    that is still present, and its credential names a version that is not.
    Nothing is deleted: the likeliest cause is a misconfigured environment, and
    reacting to that by wiping every stored credential would be far worse than
    the outage. The row is left for the clock and the reason is logged.
    """
    path = str(tmp_path / "custody.duckdb")
    key1, key2 = _key(), _key()
    lines = []

    old = cred.CredentialStore(path=path, keys=f"1:{key1}", out=lambda *a: None)
    minted = old.connect(FAKE_SWID, FAKE_S2)

    # Version 2 arrives and the user reconnects, so the credential moves to
    # version 2 while this browser's session id stays a version-1 HMAC.
    rotated = cred.CredentialStore(path=path, keys=f"1:{key1},2:{key2}",
                                   out=lambda *a: None)
    rotated.connect(FAKE_SWID, FAKE_S2)
    assert rotated.resolve(minted.cookie) is not None

    # Version 2 is rolled back. The session still resolves; its credential
    # does not.
    reverted = cred.CredentialStore(path=path, keys=f"1:{key1}",
                                    out=lines.append)
    assert reverted.resolve(minted.cookie) is None
    assert reverted.counts()["credentials"] == 1     # left for the clock
    assert any("key version 2" in line for line in lines), \
        "an unreadable credential passed without a word"

    # And the plain case: a key that is gone with nothing left that knows it.
    gone = cred.CredentialStore(path=path, keys=f"9:{_key()}",
                                out=lambda *a: None)
    assert gone.resolve(minted.cookie) is None
    cred.close_all()


# --- Transport ---------------------------------------------------------------

def test_plain_http_is_refused_unless_the_dev_opt_out_is_set():
    """FAILS CLOSED, and the degraded mode is why it has to.

    The session cookie is set `Secure`, and a `Secure` cookie on a plaintext
    connection is silently DROPPED by the browser. So "allow it and warn"
    means the credential crossed the wire in the clear AND the user appears
    not to be connected -- a failure that looks like a bug, gets retried, and
    transmits it again.
    """
    cred.require_secure_transport("https", env={})          # fine

    with pytest.raises(cred.InsecureTransport):
        cred.require_secure_transport("http", env={})

    # The opt-out is explicit, and it is its OWN switch.
    cred.require_secure_transport("http", env={cred.ALLOW_PLAINTEXT_ENV: "1"})
    # Nothing else turns it off: not a generic dev flag, not a debug flag.
    for decoy in ("DEV", "DEBUG", "ENV", "NODE_ENV"):
        with pytest.raises(cred.InsecureTransport):
            cred.require_secure_transport("http", env={decoy: "1"})


def test_x_forwarded_proto_is_not_believed_unless_a_deployment_says_so():
    """The header is just a header -- a client can send it. Trusting it by
    default would make the plain-HTTP refusal something any attacker could
    opt out of by adding one line to their request."""
    forged = {"scheme": "http", "forwarded_proto": "https"}
    with pytest.raises(cred.InsecureTransport):
        cred.require_secure_transport(forged["scheme"],
                                      forged["forwarded_proto"], env={})
    # A deployment that terminates TLS at a proxy it controls opts in.
    cred.require_secure_transport(
        forged["scheme"], forged["forwarded_proto"],
        env={cred.TRUST_FORWARDED_PROTO_ENV: "1"})


# --- Redaction ---------------------------------------------------------------

def test_the_401_log_path_scrubs_a_token_carried_in_a_url(store):
    """Every new log path goes through `pipeline/redact.py`, asserted against
    a real exception carrying real-shaped credentials in its url -- the same
    way the existing redact tests do, because that is the leak that actually
    happens: httpx stringifies to a sentence wrapped around the whole request
    url, and this module's one log line is a reaction to exactly that
    exception."""
    lines = []
    talkative = cred.CredentialStore(path=store.path, keys=store._keys_spec,
                                     out=lines.append)
    minted = talkative.connect(FAKE_SWID, FAKE_S2)
    talkative.forget_if_unauthorized(minted.credential_id,
                                     _FakeHTTPStatusError(401, _espn_url()))

    printed = "\n".join(lines)
    assert printed, "the 401 path printed nothing at all"
    assert "401" in printed                       # still says what happened
    assert FAKE_S2 not in printed
    assert FAKE_SWID not in printed
    assert "7A1F9C34" not in printed
    assert "espn_s2=" not in printed or redact.SECRET_PLACEHOLDER in printed


def test_the_custody_cookie_is_scrubbed_by_name(store):
    """The session cookie is the PASSWORD to a stored ESPN account session, so
    it is worth more in a log than the espn_s2 beside it. It is not
    GUID-shaped and it is not registered as a literal (see the rule's comment
    in pipeline/redact.py for why a multi-tenant server must not do that), so
    the name rule is the whole defence."""
    minted = store.connect(FAKE_SWID, FAKE_S2)
    header = f"Cookie: {cred.COOKIE_NAME}={minted.cookie}; theme=dark"
    scrubbed = redact.redact(header)
    assert minted.cookie not in scrubbed
    assert "theme=dark" in scrubbed, "redaction ate an unrelated cookie"
    assert redact.redact(scrubbed) == scrubbed, "redaction is not idempotent"


def test_a_stored_credential_never_reaches_a_log_by_accident(store):
    """The blob is the one value in this module that is safe to print, and
    nothing prints it. Asserted by driving every path that logs at all and
    checking what came out."""
    lines = []
    noisy = cred.CredentialStore(path=store.path, keys=store._keys_spec,
                                 out=lines.append)
    minted = noisy.connect(FAKE_SWID, FAKE_S2)
    noisy.resolve(minted.cookie)
    noisy.disconnect(minted.cookie)
    noisy.reap()
    printed = "\n".join(lines)
    assert FAKE_S2 not in printed and FAKE_SWID not in printed


# --- Housekeeping ------------------------------------------------------------

def test_half_a_session_is_refused_rather_than_half_stored(store):
    """A row with a SWID and no espn_s2 carries every custody obligation and
    none of the usefulness."""
    for swid, s2 in ((FAKE_SWID, ""), ("", FAKE_S2), ("", "")):
        with pytest.raises(ValueError):
            store.connect(swid, s2)
    assert store.counts() == {"credentials": 0, "sessions": 0}


def test_reconnecting_replaces_the_stored_session_rather_than_adding_one(store):
    """ESPN reissues espn_s2; the second connect must overwrite the first,
    not leave a stale copy of a session we would otherwise hold forever."""
    first = store.connect(FAKE_SWID, FAKE_S2)
    replacement = FAKE_S2 + "ROTATEDBYESPN"
    second = store.connect(FAKE_SWID, replacement)
    assert store.counts()["credentials"] == 1
    assert store.resolve(second.cookie).espn_s2 == replacement
    assert store.resolve(first.cookie).espn_s2 == replacement


def test_the_minted_cookie_is_never_stored_anywhere(store):
    """It is a password. It exists in the response, in the browser, and
    nowhere else -- so it cannot be read back out of the database by anyone,
    including us."""
    minted = store.connect(FAKE_SWID, FAKE_S2)
    raw = _raw_bytes(store)
    assert minted.cookie.encode() not in raw
    assert len(minted.cookie) >= 40      # 256 bits, url-safe base64


def test_the_blob_is_the_only_place_the_pair_lives(store):
    """A direct check that decryption is what produces the credential, rather
    than the store keeping a convenient copy somewhere."""
    minted = store.connect(FAKE_SWID, FAKE_S2)
    conn = cred._connect(store.path)
    blob = conn.execute("SELECT blob FROM espn_credential").fetchone()[0]
    payload = json.loads(store.current_key.fernet.decrypt(blob.encode()))
    assert payload == {"swid": FAKE_SWID, "espn_s2": FAKE_S2}
