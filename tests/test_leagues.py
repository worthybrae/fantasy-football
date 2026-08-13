def test_table_split_is_complete_and_disjoint():
    """Every table the app uses is classified exactly once. A table in
    neither set would silently be treated as universal by provisioning and
    leak one league's data into every other; a table in both is a
    contradiction the code cannot honour."""
    from pipeline.db import UNIVERSAL_TABLES, LEAGUE_TABLES
    assert UNIVERSAL_TABLES.isdisjoint(LEAGUE_TABLES)
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
    from pipeline.leagues import league_db_path
    p = league_db_path("53929318", root="/tmp/lg")
    assert p == "/tmp/lg/53929318.duckdb"


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
    """Connecting twice must not reprovision and wipe a draft in progress."""
    from pipeline.db import get_conn, write_table, read_table
    from pipeline.leagues import provision_league
    import pandas as pd

    shared = str(tmp_path / "universal.duckdb")
    get_conn(shared).close()
    path = provision_league("999", universal_path=shared, root=str(tmp_path / "lg"))
    lg = get_conn(path)
    lg.execute("INSERT INTO drafted VALUES ('p1', 1)")
    lg.close()

    again = provision_league("999", universal_path=shared, root=str(tmp_path / "lg"))
    assert again == path
    lg = get_conn(path)
    assert not read_table(lg, "drafted").empty      # the in-progress draft survived
    lg.close()
