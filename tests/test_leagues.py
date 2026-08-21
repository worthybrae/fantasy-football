def test_table_split_is_complete_and_disjoint():
    """Every table the app uses is classified exactly once. A table in
    neither set would silently be treated as universal by provisioning and
    leak one league's data into every other; a table in both is a
    contradiction the code cannot honour."""
    from pipeline.db import UNIVERSAL_TABLES, LEAGUE_TABLES
    assert UNIVERSAL_TABLES.isdisjoint(LEAGUE_TABLES)
    # The draft corpus (pipeline/draft_log.py) is in NEITHER set, on purpose:
    # it lives in its own database. Both sets here describe the contents of a
    # league file, and provisioning COPIES the universal ones into each new
    # league -- right for reference data a refresh rewrites wholesale, wrong
    # for a corpus that grows, which would fork into a private history per
    # league and never share a draft between them.
    from pipeline.draft_log import CORPUS_TABLES
    assert UNIVERSAL_TABLES.isdisjoint(CORPUS_TABLES)
    assert LEAGUE_TABLES.isdisjoint(CORPUS_TABLES)
    # The league-specific set, exact -- adding a per-league table without
    # listing it here is the bug this pins.
    assert LEAGUE_TABLES == frozenset({
        "league", "draft_picks", "draft_teams", "draft_order",
        "manager_profiles", "manager_tendencies", "drafted",
        "sim_board", "sim_results", "sim_survival"})


def test_default_league_resolves_to_the_existing_database():
    """The single-user path must not move. The default league is today's
    data/nfl.duckdb, so nothing about the current app changes."""
    from pipeline.leagues import DEFAULT_LEAGUE, league_db_path
    from pipeline.db import DEFAULT_PATH
    assert league_db_path(DEFAULT_LEAGUE) == DEFAULT_PATH


def test_a_real_league_gets_its_own_file_under_the_leagues_root():
    """A league id other than the configured default -- 909090, not
    53929318, which is now DEFAULT_LEAGUE_ID (see the two tests below)."""
    from pipeline.leagues import league_db_path
    p = league_db_path("909090", root="/tmp/lg")
    assert p == "/tmp/lg/909090.duckdb"


def test_the_configured_default_league_id_also_resolves_to_the_existing_database():
    """The existing user's own league -- data/nfl.duckdb already holds its
    712 draft_picks and fitted managers -- must resolve exactly like the
    __default__ sentinel, not cold-start into a fresh per-league file just
    because it arrived as a real, parsed ESPN league id (which is the only
    way it ever arrives: parse_league_id returns digits, never
    __default__)."""
    from pipeline.leagues import DEFAULT_LEAGUE_ID, league_db_path
    from pipeline.db import DEFAULT_PATH
    assert DEFAULT_LEAGUE_ID == "53929318"
    assert league_db_path(DEFAULT_LEAGUE_ID) == DEFAULT_PATH


def test_a_league_id_unequal_to_the_default_gets_its_own_path():
    from pipeline.leagues import league_db_path
    assert league_db_path("999") != league_db_path("53929318")


def test_league_id_is_sanitised_into_the_filename():
    """A league id reaches this from a pasted URL. It must not be able to
    escape the leagues directory via path characters."""
    from pipeline.leagues import league_db_path
    p = league_db_path("../../etc/passwd", root="/tmp/lg")
    assert p.startswith("/tmp/lg/")
    assert ".." not in p.split("/tmp/lg/")[1]


def test_provision_copies_universal_tables_but_not_league_ones(tmp_path):
    """A provisioned league can build a board (universal data present) but
    starts with no draft history of its own (league tables empty), which is
    the cold-start case the model already handles."""
    from pipeline.db import get_conn, write_table, read_table
    from pipeline.leagues import provision_league
    import pandas as pd

    shared = str(tmp_path / "universal.duckdb")
    conn = get_conn(shared)
    write_table(conn, "weekly", pd.DataFrame([{"player_id": "p1", "season": 2025}]))
    write_table(conn, "draft_picks", pd.DataFrame([{"season": 2025, "overall_pick": 1}]))
    conn.close()

    path = provision_league("999", universal_path=shared, root=str(tmp_path / "lg"))
    lg = get_conn(path)
    # universal came across
    assert not read_table(lg, "weekly").empty
    # league-specific did NOT -- this league has its own (empty) history
    assert read_table(lg, "draft_picks").empty
    lg.close()


def test_provision_is_idempotent(tmp_path):
    """Connecting twice must not reprovision.

    `drafted` surviving alone doesn't pin this -- it's a LEAGUE_TABLE, and
    the copy loop only ever touches UNIVERSAL_TABLES, so it would survive
    even with the idempotency guard deleted. What the guard actually
    protects is any UNIVERSAL table already in the league file: without it,
    a reconnect re-copies from the shared source and clobbers whatever has
    changed there since provisioning. A sentinel written into a universal
    table (`weekly`), absent from the shared source, pins that.
    """
    from pipeline.db import get_conn, write_table, read_table
    from pipeline.leagues import provision_league
    import pandas as pd

    shared = str(tmp_path / "universal.duckdb")
    conn = get_conn(shared)
    write_table(conn, "weekly", pd.DataFrame([{"player_id": "p1", "season": 2025}]))
    conn.close()

    path = provision_league("999", universal_path=shared, root=str(tmp_path / "lg"))
    lg = get_conn(path)
    lg.execute("INSERT INTO drafted VALUES ('p1', 1)")
    # Overwrite the copied-in universal table with a value the shared source
    # does not have. A reprovision that skips the guard re-copies "weekly"
    # from the source and wipes this.
    write_table(lg, "weekly", pd.DataFrame([{"player_id": "sentinel", "season": 9999}]))
    lg.close()

    again = provision_league("999", universal_path=shared, root=str(tmp_path / "lg"))
    assert again == path
    lg = get_conn(path)
    assert not read_table(lg, "drafted").empty      # the in-progress draft survived
    assert read_table(lg, "weekly")["player_id"].tolist() == ["sentinel"]  # not re-copied
    lg.close()


def test_two_leagues_have_independent_drafted_tables(tmp_path):
    """The whole point. A pick marked in league A must be invisible to
    league B -- they are different files, so a global `drafted` table can no
    longer merge two live drafts into one."""
    from pipeline.db import get_conn, read_table
    from pipeline.leagues import provision_league

    shared = str(tmp_path / "universal.duckdb")
    get_conn(shared).close()
    root = str(tmp_path / "lg")

    a = get_conn(provision_league("aaa", universal_path=shared, root=root))
    b = get_conn(provision_league("bbb", universal_path=shared, root=root))
    a.execute("INSERT INTO drafted VALUES ('gibbs', 1)")

    assert read_table(a, "drafted")["player_id"].tolist() == ["gibbs"]
    assert read_table(b, "drafted").empty       # league B never saw it
    a.close()
    b.close()
