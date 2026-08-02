# Board Redesign + Market Consensus — Design

**Date:** 2026-08-02
**Status:** Approved by user (pro-terminal direction, consensus + spread market display)

## Purpose

Two coupled upgrades: (1) replace the single FantasyFootballCalculator ADP with a
multi-source market consensus (ESPN ADP + FantasyPros expert consensus + FFC),
and (2) a full visual redesign of the board in a dense "pro terminal" style.

## Part 1 — Market consensus

### Sources (all verified live for 2026, free, keyless)

| Source | What it is | Fetch |
| --- | --- | --- |
| FFC (existing) | Mock-draft ADP, 12-team PPR | existing `fetch_adp` |
| ESPN | Mock-draft ADP + PPR rank | `lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{year}/segments/0/leaguedefaults/3?view=kona_player_info` with `X-Fantasy-Filter` header (`limit` 500); fields `fullName`, `defaultPositionId` (1=QB,2=RB,3=WR,4=TE,5=K,16=DST), `ownership.averageDraftPosition`, `draftRanksByRankType.PPR.rank`, `player.id` (= espn_id) |
| FantasyPros | 81-expert consensus rank (ECR), tiers, expert spread | extract `var ecrData = {...};` JSON from `fantasypros.com/nfl/rankings/ppr-cheatsheets.php`; fields `player_name`, `player_team_id`, `player_position_id`, `rank_ecr`, `rank_ave`, `rank_std`, `tier` |
| Sleeper | ID crosswalk only (NOT a rank source) | `api.sleeper.app/v1/players/nfl`; keep active players' `gsis_id`, `espn_id`, `full_name`, `position`, `team` |

New DuckDB tables: `espn_adp`, `fp_ecr`, `sleeper_ids`. Refresh gains three jobs
(same per-source failure isolation + freshness rows). ESPN/FantasyPros are
undocumented endpoints: fetches set a browser-ish User-Agent, tolerate schema
drift by failing that source only.

### Consensus math (`scoring/market.py`, pure)

- Per source, produce an overall rank per player:
  FFC `adp` → rank by ascending adp; ESPN `averageDraftPosition` → rank
  ascending; FantasyPros `rank_ecr` used directly.
- Join to board players: ESPN via `sleeper_ids` crosswalk (`gsis_id ↔ espn_id`),
  name-norm + position fallback for players missing from the crosswalk;
  FantasyPros and FFC via existing `_norm_name` + position (DST by team).
- `market_rank` = mean of available source ranks (1–3 sources); None if zero.
- `market_spread` = max − min of available source ranks (None with <2 sources).
- Per-source detail kept: `market_sources` = `{ffc: rank|None, espn: rank|None,
  fp: rank|None, fp_tier: int|None}`.
- Board changes: `adp` column replaced by `market_rank` (float, 1 decimal),
  new `market_spread`, `market_sources` object; `edge` = `market_rank` − board
  rank (same sign convention: positive = market undervalues). `adp_rank`
  intermediate goes away. K/DST join as today (team-based for DST).
- Profile drawer: header chip shows market rank; a small "Market" section lists
  the three per-source values + FP tier.
- API: `/api/players` rows and `/api/players/{id}/profile` header carry the new
  fields. No new endpoints.

## Part 2 — Pro-terminal UI redesign

Direction: dense, fast, scannable — Bloomberg/Linear energy. Implementation
loads the frontend-design and dataviz skills before writing UI code.

### Layout

- **Top bar** (replaces current header): app title left; center: player search
  input (fuzzy substring on name/team, filters the board live, `/` focuses,
  `Esc` clears); right: per-source freshness dots (compact, tooltip detail) and
  data age.
- **Left rail** (collapsible): weight sliders + hide-drafted toggle, restyled;
  collapse to icons to maximize board width.
- **Board**: sticky column header; tight rows (~32px); tabular numerals
  everywhere (bundled mono/font via npm package — self-hosted, offline-safe;
  no CDN fonts).

### Visual system

- Design tokens in CSS variables: background layers, border, text hierarchy
  (primary/secondary/faint), accent, and a fixed position palette (QB/RB/WR/TE/
  K/DST each one hue) used consistently in board badges, drawer, and chart.
- Position shown as small colored badge, not plain text.
- Tier boundaries: grouped subtle background bands per tier (within the current
  sort), replacing hairline rules.
- Edge: signed chip (`+14` green / `−9` red, neutral gray near zero).
- Market spread: small "±N" muted suffix on the market rank when spread ≥ 12
  ranks (sources disagree); hover reveals per-source values (title tooltip).
- VOR: subtle horizontal micro-bar behind the number (scaled within position).
- Drafted rows: dimmed + struck, ✓ button becomes undo; row stays in place
  unless "hide drafted".
- Drawer inherits all tokens/badges (no structural drawer redesign).

### Keyboard

`/` focus search · `↑/↓` move row selection (visible highlight) · `Enter` open
profile for selection · `D` toggle drafted on selection · `Esc` close drawer /
clear search. Selection follows the visible (filtered/sorted) row order.

### Out of scope

New pages/routing, state libraries, drawer content redesign, virtualized
scrolling, light theme.

## Error handling

- A market source failing → dropped from consensus, freshness dot red, board
  still renders (existing degradation pattern).
- Search matching zero players → empty-state row, filters/tabs unaffected.

## Testing

- pytest: per-source rank conversion, consensus mean/spread with 1/2/3 sources,
  crosswalk join + name fallback, DST/K joins, edge sign, API shape update.
- Existing board/profile tests updated for the new columns.
- Frontend: typecheck + build + browser verification (search, keyboard nav,
  drafted flow, drawer inherit).
