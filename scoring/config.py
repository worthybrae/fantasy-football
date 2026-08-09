CURRENT_SEASON = 2026
# Deep history for profiles, game logs, and stat twins. The draft board
# scores on the RECENCY_WEIGHTS window only (see build_board).
HISTORY_SEASONS = list(range(2016, 2026))
RECENCY_WEIGHTS = {2025: 0.5, 2024: 0.3, 2023: 0.2}

# League: 8 teams, QB/2RB/2WR/TE/2Flex(W-R-T)/K/DST, PPR, 5 bench
LEAGUE_TEAMS = 8

DEFAULT_WEIGHTS = {
    "production": 0.35, "role": 0.25, "environment": 0.20,
    "schedule": 0.10, "durability": 0.10,
}

# starters*8 + flex allocation (16 flex slots ~ RB 6 / WR 8 / TE 2) + 1 buffer for single-slot
REPLACEMENT_RANK = {"QB": 9, "RB": 22, "WR": 24, "TE": 10, "K": 9, "DST": 9}

# How the league's FLEX slots historically get filled, by position. Used to
# derive REPLACEMENT_RANK from roster shape instead of hardcoding it.
FLEX_SHARES = {"RB": 0.375, "WR": 0.5, "TE": 0.125}
