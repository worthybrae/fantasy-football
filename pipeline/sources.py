import nfl_data_py as nfl
import pandas as pd
import requests

ADP_URL = "https://fantasyfootballcalculator.com/api/v1/adp/ppr"
WEEKLY_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{year}.parquet"

def fetch_weekly(years):
    # nfl.import_weekly_data reads from the old `player_stats` release, which
    # nflverse has stopped publishing for the current season. The weekly
    # player stats now live under the `stats_player` release tag instead.
    df = pd.concat([pd.read_parquet(WEEKLY_URL.format(year=year)) for year in years])
    if "recent_team" not in df.columns and "team" in df.columns:
        df = df.rename(columns={"team": "recent_team"})
    if "season_type" in df.columns:
        df = df[df["season_type"] == "REG"]
    return df

def fetch_snap_counts(years):
    return nfl.import_snap_counts(years)

def fetch_depth_charts(season):
    return nfl.import_depth_charts([season])

def fetch_schedules(season):
    return nfl.import_schedules([season])

def parse_adp(payload: dict) -> pd.DataFrame:
    rows = [{"adp_name": p["name"],
             "position": "DST" if p["position"] == "DEF" else p["position"],
             "team": p.get("team"), "adp": p["adp"]}
            for p in payload.get("players", [])]
    return pd.DataFrame(rows, columns=["adp_name", "position", "team", "adp"])

def fetch_adp(year: int, teams: int = 12) -> pd.DataFrame:
    resp = requests.get(ADP_URL, params={"teams": teams, "year": year}, timeout=30)
    resp.raise_for_status()
    return parse_adp(resp.json())
