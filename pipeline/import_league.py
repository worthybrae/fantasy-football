"""Import ESPN league history. Run: python -m pipeline.import_league <league>"""
import sys

import pandas as pd

from pipeline import sources
from pipeline.db import get_conn, write_table, record_freshness
from pipeline.espn_league import (EspnClient, import_seasons, parse_league_id,
                                  validate_import)
from scoring.board import _ADP_POSITION_ALIASES, _ADP_TEAM_ALIASES
from scoring.config import CURRENT_SEASON

_HISTORIC_ADP_COLUMNS = ["season", "adp_name", "position", "team", "adp_rank"]


def normalize_historic_adp(df: pd.DataFrame) -> pd.DataFrame:
    """Put an ADP feed into the board's position and team vocabulary.

    `sources.parse_adp` maps DEF -> DST but leaves kickers as the feed's own
    "PK", and gives every team abbreviation exactly as the feed spells it.
    `draft_picks` comes from ESPN, whose positions are K/DST and whose teams
    are nflverse abbreviations, so writing the feed straight through means no
    K pick and no DST pick can ever match an ADP row -- roughly 2 of every 15
    picks in this league, guaranteed unmatched, dragging the import's ADP
    match rate down far enough to trip its own low-match warning and blame
    the data for a code bug. It also leaves `pos_K` and `pos_DST` as all-zero
    columns in the feature matrix.

    `team` is carried through because DSTs join on it (see
    `board.adp_match_key`); the same two alias tables the board already
    applies to the live ADP feed are applied here.
    """
    out = df.copy()
    out["position"] = out["position"].replace(_ADP_POSITION_ALIASES)
    if "team" not in out.columns:
        out["team"] = pd.NA
    out["team"] = out["team"].replace(_ADP_TEAM_ALIASES)
    return out


def build_historic_espn(frames: dict) -> pd.DataFrame:
    """One dense 1..N ranking per season from ESPN's own rankings.

    `espn_ppr_rank` is the ordering, not `espn_adp`: the rank column is
    reliable in every season checked, the ADP column is not (see
    `sources.espn_adp_is_usable`). ADP breaks ties only where it is usable.
    Re-ranking densely rather than passing ESPN's raw rank through means
    the scale matches `historic_adp`'s, so the two references are
    comparable.
    """
    rows = []
    for season, df in sorted(frames.items()):
        if df.empty or "espn_ppr_rank" not in df.columns:
            continue
        usable = sources.espn_adp_is_usable(df)
        ranked = df.dropna(subset=["espn_ppr_rank"]).copy()
        ranked["_adp"] = (pd.to_numeric(ranked["espn_adp"], errors="coerce")
                          if usable else 0.0)
        ranked = ranked.sort_values(["espn_ppr_rank", "_adp"]).reset_index(drop=True)
        ranked["espn_rank"] = ranked.index + 1
        ranked["season"] = season
        ranked["adp_usable"] = usable
        rows.append(ranked[["season", "espn_name", "position",
                            "espn_rank", "adp_usable"]])
    cols = ["season", "espn_name", "position", "espn_rank", "adp_usable"]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python -m pipeline.import_league <league-url-or-id>")
        return 2
    league_id = parse_league_id(argv[1])
    conn = get_conn()
    with EspnClient() as client:
        summary = import_seasons(conn, league_id, CURRENT_SEASON, client.get_json)

    # Historical ADP is what makes "reach" measurable: a pick only means
    # something against where the market had that player THAT year.
    frames = []
    for season in summary["seasons"]:
        try:
            df = sources.fetch_adp(season)
        except Exception as e:
            print(f"  WARN historic ADP {season}: {e}")
            continue
        df = normalize_historic_adp(df)
        df = df.sort_values("adp").reset_index(drop=True)
        df["adp_rank"] = df.index + 1
        df["season"] = season
        frames.append(df[_HISTORIC_ADP_COLUMNS])
    if frames:
        rows = pd.concat(frames, ignore_index=True)
        write_table(conn, "historic_adp", rows)
        record_freshness(conn, "historic_adp", True, len(rows))

    # ESPN's own historical rankings -- what the league's managers actually
    # saw on draft day, as opposed to the FFC consensus above.
    espn_frames = {}
    for season in summary["seasons"]:
        try:
            espn_frames[season] = sources.fetch_espn_adp(season, limit=500)
        except Exception as e:
            print(f"  WARN historic ESPN {season}: {e}")
    historic_espn = build_historic_espn(espn_frames)
    if not historic_espn.empty:
        write_table(conn, "historic_espn", historic_espn)
        record_freshness(conn, "historic_espn", True, len(historic_espn))
        unusable = sorted(historic_espn[~historic_espn["adp_usable"]]["season"].unique())
        if unusable:
            print(f"  note: ESPN ADP unusable for {unusable} "
                  "(reset to a constant) -- ranked on espn_ppr_rank instead")

    print(f"Imported {summary['picks']} picks across "
          f"{len(summary['seasons'])} seasons:")
    print("\n".join(validate_import(conn)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
