# More market sources: MFL ADP + CBS Top 200 folded into consensus

**Date:** 2026-08-02 · **Status:** approved

## Scope

User asked for "as many as possible" additional ranking sources, folded
into the Mkt consensus (not per-source columns). Feasibility probes:

| Source | Verdict |
|---|---|
| MFL ADP | ✅ public JSON API (`export?TYPE=adp` + `TYPE=players` for names) |
| CBS Top 200 PPR | ✅ scrapeable; full names live in href slugs (visible names are abbreviated) |
| NFL.com | ❌ both fantasy API generations 404; page is client-rendered |
| Underdog | ❌ endpoint 404 |
| Yahoo | ❌ OAuth-only |
| Sleeper | ❌ no public ADP endpoint |
| Boris Chen | ❌ files blocked; FantasyPros-derived (already have FP) |

## Design

1. `pipeline/sources.py`: `fetch_mfl_adp(year)` (two calls: adp + player
   directory; join on MFL id; "Last, First" → "First Last"; PK→K; team-
   defense rows dropped — players only) → `[mfl_name, position, mfl_rank]`
   ranked by averagePick. `fetch_cbs()` parses the top-200 page's
   player-row blocks: rank div, full name from the `/nfl/players/<id>/<slug>/`
   href slug, position from the team/position span → `[cbs_name, position,
   cbs_rank]`.
2. Refresh jobs `mfl_adp`, `cbs_ranks` (new tables).
3. `scoring/market.py`: generic name+position rank joiner (same dedupe rules
   as the FP path); `_RANK_COLS` gains `mfl_rank`, `cbs_rank`, so
   `market_rank` (mean, skipna) and `market_spread` absorb them
   automatically. `market_sources` gains `mfl`, `cbs`. `add_market` takes
   the new frames as keyword args defaulting to None (missing tables
   degrade to NaN columns).
4. Frontend: `MarketSources` type + Mkt tooltip + drawer Market chips gain
   MFL and CBS entries. No new columns (user chose fold-in + tooltip).

## Testing

TDD: parse_mfl (name flip, PK alias, defense dropped), parse_cbs (slug
names, rank order), market fold-in math with 5 sources, degradation with
tables absent. Frontend tsc + temp-stack screenshot with real pulls.
