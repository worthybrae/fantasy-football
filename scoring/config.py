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

# starters*8 + flex allocation (16 flex slots ~ RB 6 / WR 8 / TE 2) + 1 buffer
# for single-slot -- except K and DST, see STREAMED_REPLACEMENT_RANK below.
REPLACEMENT_RANK = {"QB": 9, "RB": 22, "WR": 24, "TE": 10, "K": 3, "DST": 3}

# How the league's FLEX slots historically get filled, by position. Used to
# derive REPLACEMENT_RANK from roster shape instead of hardcoding it.
FLEX_SHARES = {"RB": 0.375, "WR": 0.5, "TE": 0.125}

# Replacement rank for the two positions nobody holds past the week they use
# them. A CALIBRATION, not a derivation: there is no roster-shape arithmetic
# behind these two numbers, they are a judgement about how kickers and
# defenses are actually managed, and they should be re-argued rather than
# re-derived if they ever look wrong.
#
# The roster-shape rule gives K and DST `teams * 1 + 1` = 9 in this league:
# the best one nobody has drafted. That is the right REPLACEMENT for a
# position you hold all season, and the wrong one for a position you stream:
# with 24 kickers and 26 defenses on the board and 8 of each rostered,
# two-thirds of both positions sits on waivers every week of the season, and
# preseason kicker/defense projections carry so little week-to-week signal
# that the free option you can pick up for the matchup is, in expectation,
# near the top of the position rather than 9th. What a draft pick at K or
# DST actually buys is the edge over THAT, and rank 3 is the flat stand-in
# for it: the top two carry a real (if small) edge, everything past them is
# a stream.
#
# Measured on the real board (data/nfl.duckdb, 249 players, 8 teams), what
# this does to the OVER REPL column the room shows:
#
#     rank 9 (before)   Brandon Aubrey +25.6   Houston Defense +26.8
#     rank 3 (after)    Brandon Aubrey +12.0   Houston Defense  +4.8
#
# The +26/+27 read like a genuine starting receiver -- George Pickens, the
# 15th WR on this board, is +26.3 and DeVonta Smith +26.0 -- for a pick that
# is worth a late round at most.
#
# Note the DIRECTION, since it is the opposite of the obvious one: a DEEPER
# replacement rank makes `vor_points` BIGGER, not smaller (`apply_vor`
# differences against the player AT the rank, so a worse baseline flatters
# everyone above it). Rank 16 would have printed +34.9/+43.5 and rank 24
# +50.1/+69.3 -- further from what the owner asked for, not closer.
#
# Only K and DST. Every other position's rank still comes from roster shape,
# untouched -- see LeagueSettings.replacement_ranks.
STREAMED_REPLACEMENT_RANK = {"K": 3, "DST": 3}

# How much a position is worth to a roster that already holds `counts`.
# Multiplies the value a pick gains over waiting (see scoring/gain.py), so a
# position at its roster cap contributes nothing however good the player is.
# Starting values, to calibrate against replayed drafts -- not derived.
#
# "deferred" is an open STARTER slot that is not yet worth filling: a
# position the roster rules let you hold exactly one of (so a second is
# never a bench asset -- `draft_sim._roster_cap` caps K and DST at 1, and
# they are the only positions where the cap equals the starter count) while
# the roster still has more picks left than unfilled starter slots. A
# CALIBRATION, like the four above it, and re-argued rather than re-derived
# if it ever looks wrong:
#
#   * BELOW "bench" (0.35), because bench depth is a player you might
#     actually start after an injury, and a second kicker is a player you
#     can never start at all -- the pick buys the slot and nothing else, and
#     the slot can be bought with any later pick just as well.
#   * ABOVE "capped" (0.0), because the slot IS one you have to fill, so the
#     row keeps a real if small number instead of collapsing into the tie
#     that put a defense and a kicker in the top fifteen in the first place.
#
# What it is worth on the real board (data/nfl.duckdb, 8 teams, 15 rounds,
# slot 2): the best kicker's gain in round 9 is +3.32 at a full starter
# weight and +0.50 at this one, against a whole-board top-15 spread of +4.6
# to -0.5 that round. The weight alone therefore CANNOT keep him off the
# list -- at those magnitudes any positive number is a top-15 number -- and
# that is exactly why scoring/gain.py orders the deferred block last as
# well as scaling it. The weight makes the number honest; the ordering is
# what answers the owner's "no kicker in the top fifteen in round 3".
NEED_WEIGHTS = {"starter": 1.0, "flex": 0.75, "deferred": 0.15,
                "bench": 0.35, "capped": 0.0}
