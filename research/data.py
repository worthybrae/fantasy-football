"""Shared inputs for experiments, loaded once.

The historical odds workbook is the one input that is NOT reproducible from
a URL: aussportsbetting.com serves 403 to anything scripted, so the file is
downloaded by hand and lives wherever the operator put it. `odds()` looks in
a few obvious places and says plainly what is missing rather than failing
deep inside a merge.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# Every name the workbook has used, including relocations, to our abbreviations.
TEAM_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Oakland Raiders": "LV",
    "Los Angeles Chargers": "LAC", "San Diego Chargers": "LAC",
    "Los Angeles Rams": "LA", "St Louis Rams": "LA", "St. Louis Rams": "LA",
    "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN", "New England Patriots": "NE",
    "New Orleans Saints": "NO", "New York Giants": "NYG", "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN",
    "Washington Redskins": "WAS", "Washington Football Team": "WAS",
    "Washington Commanders": "WAS",
}

# Committed alongside the code rather than downloaded on demand: the source
# serves 403 to anything scripted (see odds_path's message), so a fetch step
# would make every experiment that uses it unreproducible. It is 830 KB, and
# `data/` proper is git-ignored because it holds the DuckDB.
_SEARCH = (
    os.environ.get("NFL_ODDS_XLSX"),
    str(Path(__file__).resolve().parent / "data" / "nfl.xlsx"),
    "~/Downloads/nfl.xlsx",
)


def odds_path() -> Path:
    for candidate in _SEARCH:
        if not candidate:
            continue
        p = Path(candidate).expanduser()
        if p.exists():
            return p
    raise FileNotFoundError(
        "historical odds workbook not found. Download it by hand from "
        "https://www.aussportsbetting.com/data/historical-nfl-results-and-odds-data/ "
        "(the site refuses scripted requests), then either put it at "
        "data/odds/nfl.xlsx or set NFL_ODDS_XLSX to its path.")


def odds() -> pd.DataFrame:
    """Regular-season games with opening and closing lines, our team codes.

    `week` is derived by bucketing each game seven days from the season's
    first game -- good enough for a multi-week average and wrong for anything
    that needs a specific week, which is recorded on every experiment that
    uses it rather than hidden here.
    """
    df = pd.read_excel(odds_path(), sheet_name="Data")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    # The NFL season straddles the new year: January and February games belong
    # to the season that started the previous autumn.
    df["season"] = np.where(df["Date"].dt.month <= 2,
                            df["Date"].dt.year - 1, df["Date"].dt.year)
    df = df[df["Playoff Game?"] != 1].copy()
    opener = df.groupby("season")["Date"].transform("min")
    df["week"] = ((df["Date"] - opener).dt.days // 7) + 1
    df["home"] = df["Home Team"].map(TEAM_ABBR)
    df["away"] = df["Away Team"].map(TEAM_ABBR)
    unmapped = sorted(set(df.loc[df.home.isna(), "Home Team"].dropna()))
    if unmapped:
        raise ValueError(f"unmapped team names in the odds workbook: {unmapped}")
    return df.dropna(subset=["home", "away"])
