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


def _gate(monkeypatch, ids, free=False, open_founders=False,
          source=billing.CONNECTED):
    """The gate with its two lookups faked, and THE FOUNDING PERIOD CLOSED by
    default.

    That last part is why this helper takes a keyword nothing passed before.
    While `FOUNDERS_OPEN` is True every draft is free, so a gate test that
    left it alone would pass whatever the rest of the function did -- the
    price, the entitlement and the 402 would all be dead code under it. The
    tests that are ABOUT the open period say so explicitly.

    `account_context` rather than `_account_ids`, because the source is half
    the answer now: the same ids off the loopback login are read from and
    never given a seat.
    """
    monkeypatch.setattr(billing, "FOUNDERS_OPEN", open_founders)
    monkeypatch.setattr(billing, "account_context",
                        lambda request, store=None: (ids, source))
    monkeypatch.setattr(billing, "is_free_draft", lambda league_id: free)


def test_an_unpaid_real_draft_is_refused_with_the_price(monkeypatch):
    """Once the founding period is over. `acct-hmac` never claimed a seat and
    the list is closed, so this is the paywall doing its job."""
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

    # The seed runs on its own thread now (see `_db`), so the question this
    # test asks only has an answer once it has finished.
    billing._db()
    assert billing._seed_done.wait(30)

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

    store = billing._db()                   # the seed is claimed here, once
    assert billing._seed_done.wait(30)
    assert billing.is_free_draft("111") is True
    assert billing.is_free_draft("222") is True

    inserts = _record_inserts(store, monkeypatch)
    billing._seed_mocks_from_corpus(store)
    assert inserts == [], "the seed re-inserted rows already in the table"


def _record_inserts(store, monkeypatch) -> list:
    """Every INSERT the seed sends the store, in order. The SELECT it does
    first goes through the same method, so this keeps only the writes."""
    sent: list = []
    real = store.execute

    def watched(sql, params=()):
        if sql.lstrip().upper().startswith("INSERT"):
            sent.append(sql)
        return real(sql, params)

    monkeypatch.setattr(store, "execute", watched)
    return sent


def test_the_seed_writes_the_backlog_in_statements_not_in_rows(
        tmp_path, monkeypatch):
    """WHERE THE 24 SECONDS WENT. The seed used `executemany`, and both
    adapters implement that the honest way -- one execution per row -- so a
    corpus that had run 854 rooms ahead of the table was 854 statements, each
    its own commit and so its own fsync on a network volume. That is minutes
    of latency hiding behind a method name, and on 2026-08-27 the first
    request of a fresh container wore 24,026 ms of it.

    Nine hundred rooms rather than a handful, because a per-row write passes
    any test small enough not to notice."""
    import duckdb
    from pipeline import draft_log as dl

    corpus = tmp_path / "corpus.duckdb"
    conn = duckdb.connect(str(corpus))
    conn.execute("CREATE TABLE draft_log (draft_id VARCHAR, source VARCHAR, "
                 "league_id VARCHAR)")
    conn.executemany(
        "INSERT INTO draft_log VALUES (?, ?, ?)",
        [[f"d{i}", dl.SOURCE_MOCK, str(i)] for i in range(900)])
    conn.close()
    monkeypatch.setattr(dl, "CORPUS_PATH", str(corpus))
    billing.reset_for_tests(str(tmp_path / "backlog.duckdb"))

    store = billing._db()
    assert billing._seed_done.wait(30)

    # Every room landed...
    assert store.execute("SELECT count(*) FROM mock_room")[0][0] == 900
    # ...and a re-seed of the same 900 against an empty table takes a couple
    # of statements, not nine hundred of them.
    store.execute("DELETE FROM mock_room")
    inserts = _record_inserts(store, monkeypatch)
    billing._seed_mocks_from_corpus(store)
    assert store.execute("SELECT count(*) FROM mock_room")[0][0] == 900
    assert len(inserts) <= 3, f"{len(inserts)} statements for 900 rooms"


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


# -- founders -----------------------------------------------------------------
#
# THE PROMISE UNDER TEST. The first `FOUNDERS_LIMIT` accounts to connect draft
# free forever. Three ways to break it, and each has a test below: handing out
# a hundred and first seat (a promise made to somebody it was not meant for,
# which cannot be withdrawn), losing a seat somebody already holds (a promise
# broken), and letting the founder clause be tidied underneath the
# open-period one, which would take every founder's seat away on the day the
# paywall comes back and on no day before it.


def _limit(monkeypatch, seats: int) -> None:
    monkeypatch.setenv(billing.FOUNDERS_LIMIT_ENV, str(seats))


def test_the_default_limit_is_a_hundred(monkeypatch):
    monkeypatch.delenv(billing.FOUNDERS_LIMIT_ENV, raising=False)
    assert billing.founders_limit() == 100
    _limit(monkeypatch, 7)
    assert billing.founders_limit() == 7


def test_a_nonsense_limit_is_the_default_rather_than_a_crash(monkeypatch):
    """This is read on the path that opens a draft room. A typo in a
    deployment variable must not be able to take that down."""
    monkeypatch.setenv(billing.FOUNDERS_LIMIT_ENV, "one hundred")
    assert billing.founders_limit() == 100


def test_the_first_account_is_founder_one(monkeypatch):
    _limit(monkeypatch, 3)
    assert billing.claim_founder(["acct-one"]) == 1
    assert billing.is_founder(["acct-one"]) is True
    assert billing.founders_left() == 2


def test_claiming_twice_is_the_same_seat(monkeypatch):
    """Every caller is a route that runs on every page load. A second visit
    must not be a second row, or a hundred pageviews would be a hundred
    founders."""
    _limit(monkeypatch, 3)
    first = billing.claim_founder(["acct-one"])

    assert billing.claim_founder(["acct-one"]) == first
    assert billing.founders_taken() == 1


def test_seats_are_handed_out_in_order_and_then_run_out(monkeypatch):
    _limit(monkeypatch, 3)
    ordinals = [billing.claim_founder([f"acct-{i}"]) for i in range(5)]

    assert ordinals == [1, 2, 3, None, None]
    assert billing.founders_taken() == 3
    assert billing.founders_left() == 0
    # And the two who missed out are not founders by any other route.
    assert billing.is_founder(["acct-3"]) is False
    assert billing.founder_ordinal(["acct-4"]) is None


def test_an_unconnected_browser_claims_nothing(monkeypatch):
    """The empty list is somebody who has not connected, not a wildcard --
    the same rule `entitled` follows. A seat handed to nobody is a seat
    nobody can ever be given."""
    _limit(monkeypatch, 3)
    assert billing.claim_founder([]) is None
    assert billing.founders_taken() == 0


def test_a_rotated_key_finds_its_seat_and_does_not_take_a_second(monkeypatch):
    """Rotating the custody key computes a different id for the same person.
    The row keeps the id it was written under, so a read that only looked at
    the newest would both lose the founder his seat AND spend another one on
    him."""
    _limit(monkeypatch, 3)
    billing.claim_founder(["old-key-id"])

    assert billing.claim_founder(["new-key-id", "old-key-id"]) == 1
    assert billing.founders_taken() == 1
    assert billing.is_founder(["new-key-id", "old-key-id"]) is True


def test_a_closed_list_hands_out_nothing_more(monkeypatch):
    _limit(monkeypatch, 100)
    monkeypatch.setattr(billing, "FOUNDERS_OPEN", False)

    assert billing.claim_founder(["latecomer"]) is None
    assert billing.founders_taken() == 0


def test_a_limit_of_zero_sells_nothing_for_free(monkeypatch):
    """The switch that turns the offer off without a deploy."""
    _limit(monkeypatch, 0)
    assert billing.claim_founder(["acct-one"]) is None
    assert billing.founders_left() == 0


def test_a_store_that_cannot_be_written_costs_the_request_nothing(monkeypatch):
    """`claim_founder_quietly` is what the routes call. The claim is a side
    effect of requests that exist to answer other questions, and a billing
    store that is briefly away is not a reason to fail one of them."""
    def broken():
        raise billing.StoreError("the entitlement store is not usable")

    monkeypatch.setattr(billing, "_db", broken)
    assert billing.claim_founder_quietly(["acct-one"]) is None


# -- founders at the gate ------------------------------------------------------


def test_while_the_founding_period_is_open_nobody_pays(monkeypatch):
    """The paywall is off. A real league, an account that has bought nothing,
    and no refusal."""
    _gate(monkeypatch, ["acct-hmac"], open_founders=True)
    billing.require_paid(_Request(), "999", "2026")      # does not raise


def test_opening_a_real_draft_takes_a_seat(monkeypatch):
    _limit(monkeypatch, 3)
    _gate(monkeypatch, ["acct-hmac"], open_founders=True)

    billing.require_paid(_Request(), "999", "2026")

    assert billing.founder_ordinal(["acct-hmac"]) == 1


def test_a_founder_still_drafts_free_after_the_paywall_comes_back(monkeypatch):
    """THE ORDERING THIS WHOLE FEATURE RESTS ON.

    Billing on, the founding period over, nothing bought -- which is a 402 for
    everybody else in this file. The founder clause sits above the
    open-period one in `require_paid` precisely so that it is still there on
    the day that constant flips, and this is the test that notices if it is
    ever tidied below it.
    """
    _limit(monkeypatch, 3)
    billing.claim_founder(["acct-founder"])
    _gate(monkeypatch, ["acct-founder"])                 # FOUNDERS_OPEN False

    assert billing.enabled() is True
    assert billing.entitled(["acct-founder"], "999", 2026) is False
    billing.require_paid(_Request(), "999", "2026")      # does not raise


def test_a_stranger_pays_once_the_period_is_over(monkeypatch):
    """The same request from somebody who never claimed a seat, so the two
    tests differ by one row in one table."""
    from fastapi import HTTPException

    _limit(monkeypatch, 3)
    billing.claim_founder(["acct-founder"])
    _gate(monkeypatch, ["acct-stranger"])

    with pytest.raises(HTTPException) as caught:
        billing.require_paid(_Request(), "999", "2026")
    assert caught.value.status_code == 402


def test_a_closed_list_does_not_let_the_gate_hand_out_a_seat(monkeypatch):
    """The gate claims, so it is also a way to get one. It must stop being
    one at the same moment every other caller does."""
    from fastapi import HTTPException

    _limit(monkeypatch, 3)
    _gate(monkeypatch, ["acct-stranger"])

    with pytest.raises(HTTPException):
        billing.require_paid(_Request(), "999", "2026")
    assert billing.founders_taken() == 0


# -- the hundred and first ----------------------------------------------------

# Enough threads and enough accounts that the interleaving actually happens.
# The window is one statement wide, so a serial run of this test proves
# nothing; twelve threads racing for eight seats is where a count taken before
# somebody else's insert would show up.
CLAIM_THREADS = 12
CLAIM_SEATS = 8


def test_a_dozen_threads_racing_cannot_overfill_the_list(monkeypatch):
    """THE FAILURE THIS PINS: TWO PEOPLE COUNT 99 AND BOTH INSERT.

    FastAPI runs every sync handler in a threadpool, so two people connecting
    in the same millisecond is the ordinary case rather than a rare one. A
    count read before somebody else's insert lands is a hundred and first
    founder -- a promise made to somebody it was not meant for, which cannot
    be taken back afterwards, which is why this is worth a dozen threads in
    the suite.
    """
    from threading import Barrier, Thread

    _limit(monkeypatch, CLAIM_SEATS)
    start = Barrier(CLAIM_THREADS)
    got: list = []

    def claim(n):
        start.wait()
        got.append(billing.claim_founder([f"racer-{n}"]))

    threads = [Thread(target=claim, args=(n,)) for n in range(CLAIM_THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    seats = sorted(o for o in got if o is not None)
    # Exactly the seats there were, each handed out once, and nobody else got
    # anything at all.
    assert seats == list(range(1, CLAIM_SEATS + 1))
    assert billing.founders_taken() == CLAIM_SEATS
    assert billing.founders_left() == 0


# -- one decision, four callers ------------------------------------------------
#
# WHAT WENT WRONG BEFORE THIS SECTION EXISTED. `require_paid` learned about
# founders and the open founding period; the room's state payload, the status
# endpoint and the checkout did not. Three of the four callers therefore
# answered "you have not paid" about a draft the fourth was letting through
# free -- a locked room, a checkout button, and $9.99 taken for something
# nobody was going to charge for. `free_reason` is the one decision now, and
# the tests below are the three callers that used to have their own.


@pytest.fixture
def api(monkeypatch):
    """The three billing endpoints on an app of their own.

    Not `create_app`: these routes are mounted by `register_billing_routes`
    and read nothing but the request, so a whole server would be fifteen
    routers and a board build to answer three questions.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    with TestClient(billing.register_billing_routes(FastAPI())) as client:
        yield client


def _status(api, league="999", season="2026"):
    return api.get(f"/api/billing/status?leagueId={league}&season={season}")


def _checkout(api, league="999", season="2026"):
    return api.post("/api/billing/checkout",
                    json={"leagueId": league, "season": season})


def _fake_stripe(monkeypatch):
    """A Stripe that always opens a session, so a test can tell "refused
    before Stripe" from "refused by Stripe"."""
    session = type("S", (), {"url": "https://checkout.stripe.test/pay"})()
    sessions = type("Sessions", (), {
        "create": staticmethod(lambda params=None, options=None: session)})()
    checkout = type("Checkout", (), {"sessions": sessions})()
    client = type("Client", (), {
        "v1": type("V1", (), {"checkout": checkout})()})()
    monkeypatch.setattr(billing, "_client", lambda: client)


def test_the_status_endpoint_charges_nobody_while_the_period_is_open(
        monkeypatch, api):
    _gate(monkeypatch, ["acct-hmac"], open_founders=True)

    body = _status(api).json()

    assert body["required"] is False and body["entitled"] is True
    assert body["reason"] == billing.FREE_FOUNDERS_OPEN


def test_the_status_endpoint_keeps_a_founder_free_after_the_period(
        monkeypatch, api):
    billing.claim_founder(["acct-founder"])
    _gate(monkeypatch, ["acct-founder"])

    body = _status(api).json()

    assert body["required"] is False and body["entitled"] is True
    assert body["reason"] == billing.FREE_FOUNDER


def test_the_status_endpoint_still_asks_a_stranger_to_pay(monkeypatch, api):
    _gate(monkeypatch, ["acct-stranger"])

    body = _status(api).json()

    assert body["required"] is True and body["entitled"] is False
    assert "reason" not in body


def test_the_status_endpoint_still_says_mock(monkeypatch, api):
    """The one reason this payload has always carried."""
    _gate(monkeypatch, ["acct-stranger"], free=True)

    assert _status(api).json()["reason"] == "mock"


def test_a_checkout_is_refused_for_a_draft_that_is_already_free(
        monkeypatch, api):
    """THE ONE THAT TAKES MONEY. A checkout offered for a draft the gate lets
    through is $9.99 charged for nothing."""
    _fake_stripe(monkeypatch)
    _gate(monkeypatch, ["acct-hmac"], open_founders=True)

    res = _checkout(api)

    assert res.status_code == 409
    assert "founding offer" in res.json()["detail"]


def test_a_founder_is_refused_a_checkout_after_the_period_too(
        monkeypatch, api):
    _fake_stripe(monkeypatch)
    billing.claim_founder(["acct-founder"])
    _gate(monkeypatch, ["acct-founder"])

    res = _checkout(api)

    assert res.status_code == 409
    assert "founding member" in res.json()["detail"]


def test_a_paid_draft_is_still_refused_in_its_own_words(monkeypatch, api):
    _fake_stripe(monkeypatch)
    _gate(monkeypatch, ["acct-hmac"])
    billing.grant("acct-hmac", "999", 2026)

    res = _checkout(api)

    assert res.status_code == 409
    assert res.json()["detail"] == "This draft is already paid for."


def test_a_mock_checkout_is_still_a_400(monkeypatch, api):
    """Not the 409 above: a mock is not "already free for you", it is free for
    everybody and was never for sale."""
    _fake_stripe(monkeypatch)
    _gate(monkeypatch, ["acct-hmac"], free=True)

    res = _checkout(api)

    assert res.status_code == 400
    assert res.json()["detail"] == "Mock drafts are free."


def test_a_checkout_with_nothing_to_attach_to_is_a_401(monkeypatch, api):
    _fake_stripe(monkeypatch)
    _gate(monkeypatch, [], source=None)

    assert _checkout(api).status_code == 401


def test_a_stranger_with_the_period_over_still_reaches_stripe(
        monkeypatch, api):
    """The refusals above are about drafts that are free, not a blanket no:
    somebody who owes $9.99 still gets a checkout."""
    _fake_stripe(monkeypatch)
    _gate(monkeypatch, ["acct-stranger"])

    res = _checkout(api)

    assert res.status_code == 200
    assert res.json()["url"] == "https://checkout.stripe.test/pay"


# -- the machine we are running on --------------------------------------------


def test_the_local_login_is_labelled_as_local(monkeypatch):
    """`account_context` says WHERE an id came from, and that is the whole
    mechanism behind the rule below."""
    monkeypatch.setattr("api.custody.custody_for",
                        lambda request, store=None: None)
    monkeypatch.setattr("pipeline.espn_drafts.saved_session",
                        lambda: ("{OWNER-SWID}", "espn_s2_value"))
    monkeypatch.setattr(billing, "_custody_store",
                        lambda store=None: type("S", (), {
                            "account_ids": staticmethod(
                                lambda swid: [f"id:{swid}"])})())

    assert billing.account_context(_LocalRequest()) == (
        ["id:{OWNER-SWID}"], billing.LOCAL)
    # And the connected path is labelled too, or the rule below would refuse
    # everybody a seat.
    monkeypatch.setattr(
        billing, "custody_for",
        lambda request, store=None: type("R", (), {"swid": "{X}"})())
    assert billing.account_context(_LocalRequest())[1] == billing.CONNECTED


def test_the_owners_own_machine_drafts_free_and_takes_no_seat(monkeypatch):
    """WHY THIS IS A RULE AND NOT A DETAIL. The saved ESPN login is honoured
    only on a request that did not come off a network -- a checkout, `make
    up`, the owner's laptop. The id it resolves to is computed from THAT
    machine's custody key, and with the deployment's `.env` loaded the store
    it would be written to is the shared Postgres: a founder seat spent on a
    row nobody can ever be matched to, and one fewer for a real reader. The
    first draft room ever opened locally would have taken #1.
    """
    _limit(monkeypatch, 3)
    _gate(monkeypatch, ["owner-id"], source=billing.LOCAL, open_founders=True)

    billing.require_paid(_Request(), "999", "2026")      # does not raise
    assert billing.founders_taken() == 0


def test_the_owners_own_machine_is_free_after_the_period_too(monkeypatch):
    """It is not the founding offer that makes it free -- there is simply
    nobody to charge on the machine the server is running on."""
    _limit(monkeypatch, 3)
    _gate(monkeypatch, ["owner-id"], source=billing.LOCAL)

    assert billing.free_reason(["owner-id"], "999", 2026,
                               billing.LOCAL) == billing.FREE_LOCAL
    billing.require_paid(_Request(), "999", "2026")      # does not raise
    assert billing.founders_taken() == 0


# -- an ordinal is an identity ------------------------------------------------


def test_a_seat_given_back_does_not_lend_its_number_to_the_next_person(
        monkeypatch):
    """One past the HIGHEST, not one past the count. They agree until a row
    leaves the table -- a seat withdrawn by hand, a test cleaning up -- and
    after that the count would hand out a number somebody already has. With
    `ordinal` unique that is not two founders numbered 2, it is a claim that
    silently does nothing and an account left with no seat at all."""
    _limit(monkeypatch, 5)
    for i in range(3):
        billing.claim_founder([f"acct-{i}"])
    billing._db().execute("DELETE FROM founder WHERE account_id = ?",
                          ["acct-1"])

    assert billing.founders_taken() == 2
    assert billing.claim_founder(["acct-new"]) == 4


def test_two_founders_cannot_be_given_the_same_number(monkeypatch):
    """The constraint, not the lock. A duplicate ordinal is a badge that lies
    to one of the two people holding it, and the process lock is the only
    thing that would stand between the table and that pair without this."""
    _limit(monkeypatch, 5)
    billing.claim_founder(["acct-one"])

    with pytest.raises(billing.StoreError):
        billing._db().execute(
            "INSERT INTO founder (account_id, ordinal, granted_at) "
            "VALUES (?, ?, ?)", ["acct-two", 1, billing._now()])


# -- a misconfiguration says so ------------------------------------------------


def test_a_negative_limit_is_the_default_and_is_said_out_loud(
        monkeypatch, capsys):
    """`-1` read as "no seats" would switch the offer off and look exactly
    like it working. Zero is the deliberate way to do that; a negative number
    is a typo."""
    monkeypatch.setenv(billing.FOUNDERS_LIMIT_ENV, "-1")

    assert billing.founders_limit() == 100

    said = capsys.readouterr().out
    assert billing.FOUNDERS_LIMIT_ENV in said and "negative" in said


def test_a_bad_limit_is_said_once_and_not_once_a_poll(monkeypatch, capsys):
    """This is read every time a room polls its state. A line a second for as
    long as the typo lasts is a log nobody can read."""
    monkeypatch.setenv(billing.FOUNDERS_LIMIT_ENV, "one hundred")

    for _ in range(5):
        assert billing.founders_limit() == 100

    assert capsys.readouterr().out.count("\n") == 1


def test_a_seat_that_could_not_be_claimed_is_said_once(monkeypatch, capsys):
    """Swallowed silently, a store that had stopped accepting claims would
    look exactly like a hundred seats already gone: every visitor told
    nothing, no error anywhere, and the offer quietly dead."""
    def broken():
        raise billing.StoreError("the entitlement store is not usable")

    monkeypatch.setattr(billing, "_db", broken)

    assert billing.claim_founder_quietly(["acct-one"]) is None
    assert billing.claim_founder_quietly(["acct-one"]) is None

    said = capsys.readouterr().out
    assert "founder seat could not be claimed" in said
    assert said.count("\n") == 1
