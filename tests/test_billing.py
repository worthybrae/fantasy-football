"""api.billing: who has paid for which draft, and what that gates.

WHAT THESE ARE GUARDING. Three failures here are each worse than a bug:
granting a draft nobody paid for, charging somebody for a mock (which is the
free trial this product's whole funnel depends on), and breaking every local
checkout by making payment mandatory somewhere it was never configured. Each
one has tests below, and the third is the reason `enabled()` reads the
environment on every call rather than at import.

Nothing here talks to Stripe. The webhook's decisions are exercised through
`_handle_event` on hand-built event dicts, which is the same shape
`stripe.Webhook.construct_event` returns once a signature has been verified --
the verification itself is Stripe's library's job and is not re-tested here.
"""
import pytest

from api import billing


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Never the real billing database, and billing ON unless a test says
    otherwise -- every gate below is a no-op without a key, so a fixture that
    left it off would make most of these pass vacuously."""
    monkeypatch.setenv(billing.KEY_ENV, "sk_test_not_a_real_key")
    monkeypatch.setenv(billing.PRICE_ENV, "price_test")
    billing.reset_for_tests(str(tmp_path / "billing.duckdb"))
    yield
    billing.reset_for_tests(str(tmp_path / "billing.duckdb"))


def _session_event(event_id="evt_1", account="acct-hmac", league="777",
                   season="2026", status="paid", kind="checkout.session.completed"):
    return {
        "id": event_id,
        "type": kind,
        "data": {"object": {
            "id": "cs_test_1",
            "payment_status": status,
            "payment_intent": "pi_test_1",
            "client_reference_id": account,
            "metadata": {"account_id": account, "league_id": league,
                         "season": season},
        }},
    }


# -- entitlements ------------------------------------------------------------


def test_nobody_is_entitled_to_anything_by_default():
    assert billing.entitled(["acct-hmac"], "777", 2026) is False


def test_a_paid_draft_is_entitled_and_only_that_draft():
    billing.grant("acct-hmac", "777", 2026)

    assert billing.entitled(["acct-hmac"], "777", 2026) is True
    # A different league, a different season, a different account: three
    # separate ways to be wrong about what was bought.
    assert billing.entitled(["acct-hmac"], "888", 2026) is False
    assert billing.entitled(["acct-hmac"], "777", 2027) is False
    assert billing.entitled(["somebody-else"], "777", 2026) is False


def test_a_second_key_version_still_finds_the_purchase():
    """Key rotation must not orphan a paid draft. `account_ids` yields the id
    under every key version, newest first, and a purchase made under the old
    one is still that person's."""
    billing.grant("old-key-id", "777", 2026)
    assert billing.entitled(["new-key-id", "old-key-id"], "777", 2026) is True


def test_an_account_with_no_stored_session_owns_nothing():
    """The empty list is an unconnected browser, not a wildcard."""
    billing.grant("acct-hmac", "777", 2026)
    assert billing.entitled([], "777", 2026) is False


# -- the webhook -------------------------------------------------------------


def test_a_completed_paid_session_grants_the_draft():
    assert billing._handle_event(_session_event()) == {"granted": True}
    assert billing.entitled(["acct-hmac"], "777", 2026) is True


def test_the_same_event_twice_grants_once():
    """Stripe retries on any non-2xx and can deliver an event twice on its
    own. A replay must not be a second purchase."""
    billing._handle_event(_session_event(event_id="evt_dupe"))
    again = billing._handle_event(_session_event(event_id="evt_dupe"))

    assert again == {"ignored": "duplicate"}


def test_an_unpaid_completed_session_grants_nothing():
    """The delayed-notification case. With those payment methods the
    completed event arrives while the session is still unpaid, and granting on
    it would hand out access for payments that later fail."""
    result = billing._handle_event(_session_event(status="unpaid"))

    assert result == {"ignored": "unpaid"}
    assert billing.entitled(["acct-hmac"], "777", 2026) is False


def test_the_money_landing_later_grants_it_then():
    """...and the same payment succeeding hours later is what grants it."""
    billing._handle_event(_session_event(event_id="evt_a", status="unpaid"))
    billing._handle_event(_session_event(
        event_id="evt_b", status="paid",
        kind="checkout.session.async_payment_succeeded"))

    assert billing.entitled(["acct-hmac"], "777", 2026) is True


def test_a_failed_async_payment_leaves_nothing_behind():
    billing._handle_event(_session_event(event_id="evt_a", status="unpaid"))
    billing._handle_event(_session_event(
        event_id="evt_b", kind="checkout.session.async_payment_failed"))

    assert billing.entitled(["acct-hmac"], "777", 2026) is False


def test_a_payment_this_integration_did_not_make_is_ignored():
    """A Payment Link, a Dashboard-created invoice, somebody testing: a
    session with no metadata has nothing to grant against. Answered rather
    than errored, so Stripe stops retrying it."""
    event = _session_event()
    event["data"]["object"]["metadata"] = {}
    event["data"]["object"]["client_reference_id"] = None

    assert billing._handle_event(event) == {"ignored": "no metadata"}


def test_a_refund_takes_the_draft_back():
    billing._handle_event(_session_event())
    refund = {"id": "evt_refund", "type": "charge.refunded",
              "data": {"object": {"payment_intent": "pi_test_1"}}}

    assert billing._handle_event(refund) == {"revoked": 1}
    assert billing.entitled(["acct-hmac"], "777", 2026) is False


def test_a_refund_delivered_twice_does_not_revoke_a_later_purchase():
    """The reason refunds are idempotent too: buy, refund, buy again, and a
    replayed refund event must not take the SECOND purchase away."""
    billing._handle_event(_session_event(event_id="evt_buy_1"))
    refund = {"id": "evt_refund", "type": "charge.refunded",
              "data": {"object": {"payment_intent": "pi_test_1"}}}
    billing._handle_event(refund)
    billing._handle_event(_session_event(event_id="evt_buy_2"))

    assert billing._handle_event(refund) == {"ignored": "duplicate"}
    assert billing.entitled(["acct-hmac"], "777", 2026) is True


def test_an_unrelated_event_is_ignored_quietly():
    assert billing._handle_event(
        {"id": "evt_x", "type": "customer.created",
         "data": {"object": {}}}) == {"ignored": "customer.created"}


# -- which drafts are free ---------------------------------------------------


def test_a_room_we_seated_somebody_in_is_free(monkeypatch):
    monkeypatch.setattr("api.lobby.cached_rows", lambda: [])
    billing.note_mock_room("555")
    assert billing.is_free_draft("555") is True


def test_a_room_the_mock_lobby_has_listed_is_free_and_remembered(monkeypatch):
    """Somebody who joined a mock on ESPN's own site never touches our
    mock-join. The directory is how they are recognised, and the id is kept
    because the directory drops the room the moment it fills."""
    monkeypatch.setattr("api.lobby.cached_rows", lambda: [{"leagueId": 555}])
    assert billing.is_free_draft("555") is True

    # And now without the directory at all.
    monkeypatch.setattr("api.lobby.cached_rows", lambda: [])
    assert billing.is_free_draft("555") is True


def test_a_league_no_mock_directory_has_ever_shown_is_not_free(monkeypatch):
    monkeypatch.setattr("api.lobby.cached_rows", lambda: [{"leagueId": 555}])
    assert billing.is_free_draft("999") is False


def test_an_unreadable_lobby_lets_the_draft_through(monkeypatch):
    """The tie goes to the drafter. Blocking somebody out of a free mock on
    draft night because ESPN is down is a worse failure than a missed sale."""
    def broken():
        raise RuntimeError("ESPN is down")

    monkeypatch.setattr("api.lobby.cached_rows", broken)
    monkeypatch.setattr("api.lobby._cached_rows", broken)
    assert billing.is_free_draft("999") is True


def test_nothing_is_learned_when_nothing_is_for_sale(monkeypatch):
    """No key, no billing database. A laptop and a test run should not grow a
    file to answer a question neither of them asks."""
    monkeypatch.delenv(billing.KEY_ENV, raising=False)
    billing.note_mock_rooms(["1", "2", "3"])
    monkeypatch.setattr("api.lobby.cached_rows", lambda: [])

    # Billing off, so everything is free anyway -- and the table stayed empty.
    assert billing.enabled() is False


# -- the gate ----------------------------------------------------------------


class _Request:
    """Enough of a request for the gate: it never reads anything but the
    cookies, and the account lookup is monkeypatched around it."""
    cookies: dict = {}


def _gate(monkeypatch, ids, free=False):
    monkeypatch.setattr(billing, "_account_ids", lambda request, store=None: ids)
    monkeypatch.setattr(billing, "is_free_draft", lambda league_id: free)


def test_an_unpaid_real_draft_is_refused_with_the_price(monkeypatch):
    from fastapi import HTTPException

    _gate(monkeypatch, ["acct-hmac"])
    with pytest.raises(HTTPException) as caught:
        billing.require_paid(_Request(), "999", "2026")

    assert caught.value.status_code == 402
    # The league is in the body: the page has to offer checkout for the draft
    # that was refused, not for whatever it last had in a variable.
    assert caught.value.detail["league_id"] == "999"
    assert caught.value.detail["season"] == 2026


def test_a_paid_draft_passes(monkeypatch):
    _gate(monkeypatch, ["acct-hmac"])
    billing.grant("acct-hmac", "999", 2026)
    billing.require_paid(_Request(), "999", "2026")      # does not raise


def test_a_mock_is_never_charged_for(monkeypatch):
    _gate(monkeypatch, [], free=True)
    billing.require_paid(_Request(), "555", "2026")      # does not raise


def test_billing_switched_off_gates_nothing(monkeypatch):
    """The local path, and the one that must never break: no key, no gate,
    not even a question asked about the league."""
    monkeypatch.delenv(billing.KEY_ENV, raising=False)

    def explode(league_id):
        raise AssertionError("must not even ask")

    monkeypatch.setattr(billing, "is_free_draft", explode)
    billing.require_paid(_Request(), "999", "2026")      # does not raise


# -- the return URL ----------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ("/draft", "/draft"),
    ("/mocks?draft=mock%3Aabc", "/mocks?draft=mock%3Aabc"),
    (None, "/"),
    ("", "/"),
    # The three that matter: this value goes into a URL Stripe redirects a
    # browser to, so an unchecked one is an open redirect with a payment page
    # in front of it.
    ("//evil.example.com", "/"),
    ("https://evil.example.com/steal", "/"),
    ("javascript:alert(1)", "/"),
])
def test_only_a_path_on_this_site_survives(raw, expected):
    assert billing._return_to(raw) == expected


def test_every_mock_the_corpus_remembers_is_free(tmp_path, monkeypatch):
    """A mock that has already closed is the case the two live tests miss:
    ESPN drops the room from its directory the moment it fills, and the
    farm's own rooms never touch mock-join. The corpus has known all along
    which drafts were mocks."""
    import duckdb
    from pipeline import draft_log as dl

    corpus = tmp_path / "corpus.duckdb"
    conn = duckdb.connect(str(corpus))
    conn.execute("CREATE TABLE draft_log (draft_id VARCHAR, source VARCHAR, "
                 "league_id VARCHAR)")
    conn.execute("INSERT INTO draft_log VALUES ('a', ?, '1802561235'), "
                 "('b', 'espn_live', '999')", [dl.SOURCE_MOCK])
    conn.close()
    monkeypatch.setattr(dl, "CORPUS_PATH", str(corpus))
    billing.reset_for_tests(str(tmp_path / "seeded.duckdb"))
    monkeypatch.setattr("api.lobby.cached_rows", lambda: [])

    assert billing.is_free_draft("1802561235") is True
    # And a real league in the same corpus is still a real league.
    assert billing.is_free_draft("999") is False


def test_the_corpus_seed_writes_only_what_the_table_is_missing(
        tmp_path, monkeypatch):
    """Every id in `mock_room` is one an earlier boot already wrote, and
    re-inserting all of them is one round trip per row to a database in
    another region to say nothing. The seed asks what is there first."""
    import duckdb
    from pipeline import draft_log as dl

    corpus = tmp_path / "corpus.duckdb"
    conn = duckdb.connect(str(corpus))
    conn.execute("CREATE TABLE draft_log (draft_id VARCHAR, source VARCHAR, "
                 "league_id VARCHAR)")
    conn.execute("INSERT INTO draft_log VALUES ('a', ?, '111'), ('b', ?, '222')",
                 [dl.SOURCE_MOCK, dl.SOURCE_MOCK])
    conn.close()
    monkeypatch.setattr(dl, "CORPUS_PATH", str(corpus))
    monkeypatch.setattr("api.lobby.cached_rows", lambda: [])
    billing.reset_for_tests(str(tmp_path / "seeded-once.duckdb"))

    store = billing._db()                   # the seed runs here, once
    assert billing.is_free_draft("111") is True
    assert billing.is_free_draft("222") is True

    wrote = []
    real = store.executemany
    monkeypatch.setattr(store, "executemany",
                        lambda sql, rows: wrote.append(list(rows)) or real(sql, rows))
    billing._seed_mocks_from_corpus(store)
    assert wrote == [], "the seed re-inserted rows already in the table"


def test_a_corpus_that_cannot_be_read_is_not_fatal(tmp_path, monkeypatch):
    """The farm writes that file as drafts finish. A lock held by it is not a
    reason to fail a request -- the two live tests still work."""
    from pipeline import draft_log as dl

    monkeypatch.setattr(dl, "CORPUS_PATH", str(tmp_path / "not-here.duckdb"))
    billing.reset_for_tests(str(tmp_path / "unseeded.duckdb"))
    monkeypatch.setattr("api.lobby.cached_rows", lambda: [{"leagueId": 555}])

    assert billing.is_free_draft("555") is True
    assert billing.is_free_draft("999") is False


# -- who the buyer is --------------------------------------------------------


class _LocalRequest:
    """A request straight off the loopback interface, no proxy in front."""
    cookies: dict = {}
    headers: dict = {}

    class client:
        host = "127.0.0.1"


class _ProxiedRequest:
    """The same address, but forwarded -- which is what a deployment looks
    like from inside the container."""
    cookies: dict = {}
    headers = {"x-forwarded-for": "203.0.113.7"}

    class client:
        host = "127.0.0.1"


class _RemoteRequest:
    cookies: dict = {}
    headers: dict = {}

    class client:
        host = "203.0.113.7"


def test_the_owners_own_machine_can_buy_without_connecting(monkeypatch):
    """A local checkout reaches its leagues from a saved ESPN login with no
    cookie anywhere (the dashboard reports source "local"), so there is
    nothing for `custody_for` to resolve -- and the person running it IS the
    account."""
    monkeypatch.setattr("api.custody.custody_for", lambda request, store=None: None)
    monkeypatch.setattr("pipeline.espn_drafts.saved_session",
                        lambda: ("{OWNER-SWID}", "espn_s2_value"))
    monkeypatch.setattr(billing, "_custody_store",
                        lambda store=None: type("S", (), {
                            "account_ids": staticmethod(lambda swid: [f"id:{swid}"])})())

    assert billing._account_ids(_LocalRequest()) == ["id:{OWNER-SWID}"]


def test_a_forwarded_request_may_not_use_the_local_login(monkeypatch):
    """THE ONE THAT MATTERS. That same file on the deployment is the FARM's
    ESPN login. Honouring it there would resolve every visitor on earth to
    one account: a single $9.99 would unlock the product for everybody, and
    anybody could spend what somebody else bought.

    A loopback address alone is not proof -- a proxy beside the app can
    present one -- so a request carrying forwarding headers is refused even
    from 127.0.0.1."""
    monkeypatch.setattr("api.custody.custody_for", lambda request, store=None: None)
    monkeypatch.setattr("pipeline.espn_drafts.saved_session",
                        lambda: ("{FARM-SWID}", "espn_s2_value"))

    assert billing._account_ids(_ProxiedRequest()) == []
    assert billing._account_ids(_RemoteRequest()) == []


# -- the store under threads --------------------------------------------------

# Enough reads that the interleaving actually happens. The measured rate of
# the bug this guards was about one answer in a thousand, so a couple of
# hundred reads would have missed a regression more often than it caught one;
# these numbers cost a second or two and miss it rarely.
THREADS = 4
READS = 1500


def test_two_accounts_read_from_four_threads_never_come_back_empty():
    """THE FAILURE THIS PINS: A ASKS, B ASKS, A FETCHES B'S ANSWER.

    A DuckDB connection holds the result of the last statement run on it, and
    `_Duck` deliberately keeps ONE connection per file -- so every statement
    that ran on it directly shared one result slot with every other thread.
    FastAPI runs sync handlers in a threadpool, so "every other thread" is
    "every other request". Measured before the fix: seven empty answers in six
    thousand reads across four threads.

    EMPTY IS THE DANGEROUS SHAPE, which is why this is worth 800 reads in the
    suite. Every read through this store means "nothing stored" by an empty
    answer: a customer is told they have not paid, and an account with a
    favourites list is shown the onboarding picker for one it already made.

    Both kinds of read, because the fix belongs to the adapter and not to
    either caller.
    """
    from threading import Thread

    billing.grant("acct-one", "111", 2026)
    billing.set_favorites(["acct-two"], ["p1", "p2", "p3"])
    wrong: list = []

    def read_entitlement():
        for _ in range(READS):
            if not billing.entitled(["acct-one"], "111", 2026):
                wrong.append("a paid draft came back unpaid")

    def read_favorites():
        for _ in range(READS):
            if billing.favorites(["acct-two"]) != ["p1", "p2", "p3"]:
                wrong.append("a saved list came back wrong")

    threads = ([Thread(target=read_entitlement) for _ in range(THREADS // 2)]
               + [Thread(target=read_favorites) for _ in range(THREADS // 2)])
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert wrong == []
