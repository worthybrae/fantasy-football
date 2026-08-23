"""Every draft this tool has ever seen, kept as facts rather than conclusions.

WHY THIS IS NOT A TABLE OF TENDENCIES. The obvious design is to compute each
owner's traits at the end of a draft and store those. E006 is the argument
against it: of the four traits that design would have stored, two (does he
draft growth, does he draft steady players) turned out to be unmeasurable, and
the one that carries the most signal -- how many tight ends a manager ends up
with -- was not among them and was found only afterwards, by re-reading the
raw picks in a window nobody had thought to use. A corpus of computed traits
would have had to be thrown away and re-collected. A corpus of picks would
not. So this module stores what happened; `owner_profile` is derived from it
and can be dropped and rebuilt at any time.

WHY THE CORPUS HAS ITS OWN DATABASE. `pipeline.db` splits tables into
per-league and shared, and `pipeline.leagues.provision` COPIES the shared ones
into each new league file. Copy semantics are right for reference data that a
refresh rewrites wholesale -- every league can hold its own snapshot of what
Ja'Marr Chase did in week 4 -- and wrong for a corpus that grows: a draft
recorded in one league would never reach another, and each league file would
drift into its own private history. So this lives in a file of its own,
belonging to neither set, and every league reads and writes the same one.
That is also the file you would sync to build a population baseline across
users, which is the whole reason it exists: E006 measured that eight managers
over six seasons cannot resolve an effect smaller than about r = 0.4, so the
answer to most of its open questions is more drafts, from anywhere.

WHAT A DRAFT IS HERE. A pool snapshot plus an ordered list of picks. Storing
the pool once per draft rather than the available set per pick is what makes
that affordable -- 250 rows instead of 250 x 130 -- and it loses nothing,
because availability at pick N is the pool minus the first N picks. Anything
we later wish we had asked (what was still on the board when he reached, how
far past his ADP the alternative was) is recoverable by replay.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

# A draft nobody can be identified in still counts. Mock opponents have no
# identity that survives the session, so their picks carry an owner key that
# is unique to the draft: they can never accumulate a personal profile, but
# they DO feed the population baseline, which is the thing most drafts are
# useful for. `is_anonymous` is what a profile builder filters on.
ANONYMOUS_PREFIX = "anon:"

# Its own file, env-overridable so a test never writes the real corpus and a
# second instance can point at a copy. Deliberately NOT under pipeline.db's
# LEAGUE_TABLES/UNIVERSAL_TABLES classification -- those describe the contents
# of a league database, and these tables are not in one.
CORPUS_PATH = os.environ.get("DRAFT_CORPUS_PATH", "data/draft_corpus.duckdb")
CORPUS_TABLES = ("draft_log", "draft_log_pick", "draft_log_pool")

SOURCE_LIVE = "espn_live"
SOURCE_HISTORY = "espn_history"
SOURCE_MOCK = "mock"

# ORDER IS THE STORED COLUMN ORDER, and every column added after a table first
# shipped goes on the END of its list, because DuckDB's `ALTER TABLE ... ADD
# COLUMN` appends. `record` still names the columns it writes (see the INSERTs
# there) so that a frame and a table which have drifted apart fail loudly
# rather than writing every value one column to the left -- silently, into the
# one table in this project that cannot be rebuilt.
_DRAFT_COLUMNS = ["draft_id", "source", "league_id", "season", "recorded_at",
                  "teams", "rounds", "my_slot", "scoring_json",
                  "settings_json", "human_seats"]
_PICK_COLUMNS = ["draft_id", "pick_no", "round", "slot", "owner_key",
                 "is_anonymous", "player_id", "position", "adp_rank",
                 "proj_points", "autodrafted", "had_owner",
                 "seconds_to_pick", "clock_seconds"]
_POOL_COLUMNS = ["draft_id", "player_id", "position", "team", "adp_rank",
                 "proj_points", "espn_rank", "espn_proj", "bye"]


def corpus_conn(path: str | None = None):
    """Open (creating if needed) the corpus database, schema ready."""
    import duckdb
    target = path or CORPUS_PATH
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(target)
    ensure_schema(conn)
    return conn


def ensure_schema(conn) -> None:
    """Create the corpus tables if they are absent.

    Plain DDL rather than `write_table`, because these accumulate: every other
    table in this project is rebuilt wholesale from a source that still exists,
    and a draft is the one thing that does not -- a mock that is not written
    down when it happens is gone. CREATE OR REPLACE anywhere near these would
    silently destroy the only copy.
    """
    conn.execute("""CREATE TABLE IF NOT EXISTS draft_log (
        draft_id VARCHAR PRIMARY KEY, source VARCHAR, league_id VARCHAR,
        season INTEGER, recorded_at TIMESTAMP, teams INTEGER, rounds INTEGER,
        my_slot INTEGER, scoring_json VARCHAR, settings_json VARCHAR)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS draft_log_pick (
        draft_id VARCHAR, pick_no INTEGER, round INTEGER, slot INTEGER,
        owner_key VARCHAR, is_anonymous BOOLEAN, player_id VARCHAR,
        position VARCHAR, adp_rank DOUBLE, proj_points DOUBLE,
        PRIMARY KEY (draft_id, pick_no))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS draft_log_pool (
        draft_id VARCHAR, player_id VARCHAR, position VARCHAR, team VARCHAR,
        adp_rank DOUBLE, proj_points DOUBLE,
        PRIMARY KEY (draft_id, player_id))""")
    # Added after the tables above went live, so an existing corpus file
    # reaches this as an ALTER rather than a CREATE. Whether ESPN's own
    # engine made the pick for that seat rather than a person -- known for a
    # live-recorded draft, unrecoverable for one backfilled from a `drafted`
    # table that carries no such flag (see pipeline.mock_backfill), so NULL
    # is the honest answer there rather than a guessed default. IF NOT
    # EXISTS makes this idempotent across every future ensure_schema call,
    # the same as the CREATE TABLEs above.
    conn.execute(
        "ALTER TABLE draft_log_pick ADD COLUMN IF NOT EXISTS autodrafted BOOLEAN")
    # Whether the seat that made this pick had a real ESPN member attached to
    # it when the draft opened, read from the league's own `?view=mTeam`
    # while the room was still live (mock leagues 404 once the draft ends, so
    # there is no later chance to ask). Per-PICK rather than a fourth table
    # keyed by (draft_id, slot): the fit walks pick rows one at a time and a
    # per-slot table would buy one normalised column at the cost of another
    # join in the hot loop and another table in a corpus that cannot be
    # rebuilt -- and `slot` is already carried per pick for exactly the same
    # reason. NULL where nobody asked ESPN: every backfilled pick, and every
    # farmed draft whose mTeam read failed.
    conn.execute("ALTER TABLE draft_log_pick "
                 "ADD COLUMN IF NOT EXISTS had_owner BOOLEAN")
    # How many seats in the room held a real person when the draft opened,
    # OUR OWN SEAT NOT COUNTED -- the farm's seat always has an owner (we
    # joined with a member id) and it is a bot, so counting it would make
    # every room look one person more human than it was. Stored per draft
    # even though it is derivable from `had_owner`, because the query it
    # exists for ("fit only on drafts with at least N people in them", "was
    # the overnight window emptier than the evening one") is a per-draft
    # filter and should not have to aggregate 128 pick rows to ask.
    conn.execute(
        "ALTER TABLE draft_log ADD COLUMN IF NOT EXISTS human_seats INTEGER")
    # THE BOARD THE ROOM WAS ACTUALLY LOOKING AT. Everything stored above
    # prices a player against the consensus market (`adp_rank`), and the
    # measurement in docs/superpowers/specs/2026-08-23-best-opponent-model-
    # design.md says that is the wrong list: a drafter in an ESPN room reads
    # ESPN's own ranking, on screen, sorted by ESPN's rank, and 23.1% of
    # human picks are the top name on THAT list against 15.9% on the market
    # one. A model that cannot see the list its subjects are reading is
    # guessing at their input. So the pool snapshot gains ESPN's rank and
    # ESPN's displayed projection alongside the market ones, and `bye`,
    # which several of the new candidate features (bye conflicts against the
    # roster already drafted) need and which no other stored column implies.
    #
    # NULL is a real answer here and has two distinct causes: a draft
    # recorded before these columns existed (fillable, and
    # pipeline.backfill_espn_board fills it -- see that module for why
    # today's board is an honest source for a preseason quantity), and a
    # player ESPN has no ranking for at all (not fillable, and not a defect:
    # ESPN publishes 500 rows and the pool is larger).
    conn.execute(
        "ALTER TABLE draft_log_pool ADD COLUMN IF NOT EXISTS espn_rank DOUBLE")
    conn.execute(
        "ALTER TABLE draft_log_pool ADD COLUMN IF NOT EXISTS espn_proj DOUBLE")
    conn.execute(
        "ALTER TABLE draft_log_pool ADD COLUMN IF NOT EXISTS bye INTEGER")
    # HOW LONG THE PICK TOOK, and how long it was allowed to take. ESPN's
    # socket has always carried both -- `SELECTING <teamId> <clock_ms>` opens
    # a turn and the team's next `SELECTED` closes it -- and we threw the
    # timing away, keeping only the order. It is the most behavioral thing in
    # the stream: a three-second pick is somebody clicking the top of the
    # list, a sixty-second pick is somebody deliberating, and an autodrafted
    # pick sits exactly at the clock's expiry. `clock_seconds` is stored
    # beside it rather than assumed to be 30, because it is a room setting
    # and "took 25 seconds" means opposite things on a 30-second clock and a
    # 90-second one.
    #
    # UNRECOVERABLE FOR EVERY DRAFT ALREADY RECORDED. Nothing on disk holds
    # the frame arrival times of a draft that has already been played, so
    # those rows stay NULL forever -- never 0, never a mean, never the clock
    # length. A default here would be indistinguishable from a real
    # instantaneous pick to every reader, and this column exists precisely to
    # tell those two apart.
    conn.execute("ALTER TABLE draft_log_pick "
                 "ADD COLUMN IF NOT EXISTS seconds_to_pick DOUBLE")
    conn.execute("ALTER TABLE draft_log_pick "
                 "ADD COLUMN IF NOT EXISTS clock_seconds DOUBLE")


def draft_id_for(source: str, league_id, season, started_at=None) -> str:
    """A stable id, so recording the same draft twice is not two drafts.

    A completed league draft is identified by its league and season: re-import
    it and it must land on the same row. A mock has no such identity -- two
    mocks in the same league on the same day are different drafts -- so its id
    takes the start time as well.
    """
    parts = [source, str(league_id or ""), str(season or "")]
    if started_at is not None:
        parts.append(started_at.isoformat() if hasattr(started_at, "isoformat")
                     else str(started_at))
    digest = hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]
    return f"{source}:{digest}"


def anonymous_key(draft_id: str, slot: int) -> str:
    """Owner key for a seat nobody can be identified in."""
    return f"{ANONYMOUS_PREFIX}{draft_id}:{slot}"


@dataclass
class DraftRecord:
    """One draft, ready to write. `picks` and `pool` are frames or None."""
    source: str
    league_id: str | None = None
    season: int | None = None
    teams: int | None = None
    rounds: int | None = None
    my_slot: int | None = None
    human_seats: int | None = None
    scoring_json: str | None = None
    settings_json: str | None = None
    started_at: datetime | None = None
    picks: pd.DataFrame | None = None
    pool: pd.DataFrame | None = None
    draft_id: str | None = None
    recorded_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc))

    def resolved_id(self) -> str:
        return self.draft_id or draft_id_for(
            self.source, self.league_id, self.season, self.started_at)


def record(conn, draft: DraftRecord) -> str:
    """Write one draft, replacing any previous copy of the SAME draft.

    Idempotent by draft_id rather than append-only: re-importing a league's
    history, or reconnecting to a draft that is still running, must not
    accumulate duplicate copies of the same picks. Delete-then-insert, so a
    draft that was recorded mid-way and is now complete ends up with the full
    pick list rather than a merge of two partial ones.

    ALL OF IT IN ONE TRANSACTION, which matters more here than anywhere else
    in this project. Every other table is rebuilt wholesale from a source
    that still exists; a draft is the one thing that cannot be re-derived
    (see the module docstring). So a delete that lands and an insert that
    then fails would not be a failed write, it would be the only copy of a
    draft destroyed by a re-record -- and that is reachable without anything
    exotic: `make mock-backfill` re-run over an already-recorded file after a
    board change that makes `_picks_frame`'s merge emit a duplicate
    `(draft_id, pick_no)` gets the delete, then the primary key rejects the
    insert. Wrapped, the failure costs the re-record and leaves what was
    already written exactly as it was. Same idiom as
    `espn_live.replace_drafted`, for the same reason.

    `ensure_schema` stays OUTSIDE the transaction: it is idempotent DDL that
    is either already true or wanted regardless of whether this particular
    draft writes.
    """
    ensure_schema(conn)
    draft_id = draft.resolved_id()

    head = pd.DataFrame([{
        "draft_id": draft_id, "source": draft.source,
        "league_id": None if draft.league_id is None else str(draft.league_id),
        "season": draft.season, "recorded_at": draft.recorded_at,
        "teams": draft.teams, "rounds": draft.rounds, "my_slot": draft.my_slot,
        "scoring_json": draft.scoring_json,
        "settings_json": draft.settings_json,
        "human_seats": draft.human_seats}], columns=_DRAFT_COLUMNS)
    picks = _shape(draft.picks, _PICK_COLUMNS, draft_id)
    pool = _shape(draft.pool, _POOL_COLUMNS, draft_id)

    try:
        conn.execute("BEGIN TRANSACTION")
        for table in ("draft_log_pick", "draft_log_pool", "draft_log"):
            conn.execute(f"DELETE FROM {table} WHERE draft_id = ?", [draft_id])
        # BY NAME, not by position. `_shape` builds each frame in the stored
        # column order, so `SELECT *` worked -- right up until a schema
        # addition landed while a process holding the OLD module was still
        # running: its frame is a column short, the table is a column longer,
        # and a positional insert either errors on the count or, if two
        # additions ever cancel out, writes the wrong value into every field
        # after the first difference. Naming them costs one f-string and makes
        # the mismatch a loud failure of that one write instead.
        conn.execute(f"INSERT INTO draft_log ({', '.join(_DRAFT_COLUMNS)}) "
                     "SELECT * FROM head")
        if not picks.empty:
            conn.execute(f"INSERT INTO draft_log_pick "
                         f"({', '.join(_PICK_COLUMNS)}) SELECT * FROM picks")
        if not pool.empty:
            conn.execute(f"INSERT INTO draft_log_pool "
                         f"({', '.join(_POOL_COLUMNS)}) SELECT * FROM pool")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return draft_id


def _shape(df, columns, draft_id) -> pd.DataFrame:
    """Force a frame into the stored column order, filling what it omits."""
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)
    out = df.copy()
    out["draft_id"] = draft_id
    for col in columns:
        if col not in out.columns:
            out[col] = None
    return out[columns]


def picks(conn, source: str | None = None) -> pd.DataFrame:
    """Every logged pick, optionally from one source."""
    ensure_schema(conn)
    sql = ("SELECT p.* FROM draft_log_pick p "
           "JOIN draft_log d USING (draft_id)")
    if source:
        return conn.execute(sql + " WHERE d.source = ?", [source]).df()
    return conn.execute(sql).df()


def summary(conn) -> pd.DataFrame:
    """Drafts held, by source -- what the corpus actually contains."""
    ensure_schema(conn)
    return conn.execute("""
        SELECT d.source, count(DISTINCT d.draft_id) AS drafts,
               count(p.pick_no) AS picks,
               count(DISTINCT CASE WHEN NOT p.is_anonymous THEN p.owner_key END)
                   AS known_owners
        FROM draft_log d LEFT JOIN draft_log_pick p USING (draft_id)
        GROUP BY d.source ORDER BY d.source""").df()


def _with_historic_adp(league_conn, picks: pd.DataFrame) -> pd.DataFrame:
    """Attach each pick's market rank AS OF its own draft.

    Keyed exactly as `scoring.draft_model` keys it, so the corpus and the
    model agree about which ADP row is which player -- two different name
    matchers would put a pick at two different ranks and only one of them
    could be right.
    """
    from scoring.board import _ADP_POSITION_ALIASES, _norm_name
    from scoring.draft_model import _match_keys
    adp = league_conn.execute("SELECT * FROM historic_adp").df()
    if adp.empty:
        return picks.assign(_adp_rank=None)
    adp = adp.assign(norm=adp["adp_name"].map(_norm_name),
                     position=adp["position"].replace(_ADP_POSITION_ALIASES))
    adp = adp.assign(key=_match_keys(adp, "adp_name"))
    adp = adp[adp["key"].notna()].sort_values("adp_rank").drop_duplicates(
        ["season", "key"], keep="first")
    out = picks.assign(key=_match_keys(picks, "player_name", "nfl_team"))
    out = out.merge(adp[["season", "key", "adp_rank"]].rename(
        columns={"adp_rank": "_adp_rank"}), on=["season", "key"], how="left")
    return out


def backfill_history(league_conn, corpus, league_id: str | None = None) -> int:
    """Fold this league's imported draft history into the corpus.

    So that profiles are built from ONE table. `draft_picks` is the ESPN
    import's own shape and stays exactly as it is -- the model and the
    simulator read it -- but a profile builder that had to union it with the
    live log would grow two code paths that drift, and the whole point of the
    corpus is that a pick is a pick whatever draft it came from.

    Reads the league database and writes the corpus -- two connections,
    because they are two files (see this module's docstring).

    Owner keys here are the manager names ESPN reports, which are stable
    within a league and NOT across leagues: two different people called "Mike"
    in two leagues would merge. Keyed on league as well for that reason --
    ESPN's member GUIDs would be better and `draft_teams` does not carry them.
    """
    ensure_schema(corpus)
    src = league_conn.execute("SELECT * FROM draft_picks").df()
    teams = league_conn.execute("SELECT * FROM draft_teams").df()
    if src.empty:
        return 0
    if not teams.empty:
        src = src.merge(teams[["season", "team_id", "manager"]],
                        on=["season", "team_id"], how="left")
    else:
        src["manager"] = None
    src = _with_historic_adp(league_conn, src)
    league = str(league_id or "")
    n = 0
    for season, rows in src.groupby("season"):
        draft_id = draft_id_for(SOURCE_HISTORY, league, int(season))
        slots = rows["team_id"] if "team_id" in rows else None
        picks_df = pd.DataFrame({
            "pick_no": rows["overall_pick"].astype("Int64"),
            "round": rows["round"].astype("Int64"),
            "slot": slots.astype("Int64") if slots is not None else None,
            "owner_key": [
                f"espn:{league}:{m}" if isinstance(m, str) and m
                else anonymous_key(draft_id, int(s) if s == s else 0)
                for m, s in zip(rows["manager"], rows.get("team_id", rows["overall_pick"]))],
            "is_anonymous": [not (isinstance(m, str) and m) for m in rows["manager"]],
            "player_id": rows.get("player_id"),
            "position": rows["position"],
            # Resolved HERE rather than left for a reader to join later. The
            # market rank that makes a pick mean anything is the one that
            # stood at that draft, and it is the least recoverable thing about
            # it -- live ADP moves daily, and a draft contributed by another
            # user arrives without their ADP tables. Storing it is what lets a
            # pick be read by anyone who has the corpus and nothing else.
            "adp_rank": rows["_adp_rank"].to_numpy(),
            "proj_points": None,
        })
        record(corpus, DraftRecord(
            source=SOURCE_HISTORY, league_id=league, season=int(season),
            teams=int(rows["team_id"].nunique()) if "team_id" in rows else None,
            rounds=int(rows["round"].max()),
            draft_id=draft_id, picks=picks_df))
        n += 1
    return n


def main(argv: list[str]) -> int:
    """Backfill this league's history into the corpus and report what it holds.

    Run: python -m pipeline.draft_log [league_id]
    """
    from pipeline.db import get_conn
    league_id = argv[1] if len(argv) > 1 else ""
    league = get_conn()
    corpus = corpus_conn()
    try:
        n = backfill_history(league, corpus, league_id=league_id)
        print(f"backfilled {n} drafts into {CORPUS_PATH}")
        print(summary(corpus).to_string(index=False))
    finally:
        corpus.close()
        league.close()
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv))
