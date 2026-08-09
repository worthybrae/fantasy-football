"""Import ESPN league history. Run: python -m pipeline.import_league <league>"""
import sys

import pandas as pd

from pipeline import sources
from pipeline.db import get_conn, write_table, record_freshness
from pipeline.espn_league import (EspnClient, import_seasons, parse_league_id,
                                  validate_import)
from scoring.config import CURRENT_SEASON


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
        df = df.sort_values("adp").reset_index(drop=True)
        df["adp_rank"] = df.index + 1
        df["season"] = season
        frames.append(df[["season", "adp_name", "position", "adp_rank"]])
    if frames:
        rows = pd.concat(frames, ignore_index=True)
        write_table(conn, "historic_adp", rows)
        record_freshness(conn, "historic_adp", True, len(rows))

    print(f"Imported {summary['picks']} picks across "
          f"{len(summary['seasons'])} seasons:")
    print("\n".join(validate_import(conn)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
