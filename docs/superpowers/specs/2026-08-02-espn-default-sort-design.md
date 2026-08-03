# ESPN PPR Rank as Default Board Order — Design

**Date:** 2026-08-02
**Status:** Approved by user

## Purpose

The board opens sorted by ESPN's PPR expert ranking — the lens league-mates
see — with our model's rank/VOR/edge as the visible disagreement. Our columns
and sorting stay fully available; only the default changes.

## Design

- **Backend:** `scoring/market.py`'s existing ESPN join (crosswalk + name
  fallback) additionally carries `espn_ppr_rank` (already ingested in the
  `espn_adp` table, currently unused) onto the board as a nullable float
  column `espn_ppr_rank`, dropped intermediates unchanged. Column joins the
  board contract, API payload, and profile header automatically.
- **Frontend:** new "ESPN" column (0-decimal rank, "—" when null); default
  sorting state becomes `[{ id: 'espn_ppr_rank', desc: false }]` with an
  explicit sortingFn placing nulls LAST in both directions. All other columns
  keep current sorting behavior. Tier bands keep their existing gating
  (our-rank sort + single-position tab).
- **Types:** `Player.espn_ppr_rank: number | null`.

## Testing

- market tests: espn_ppr_rank carried through the crosswalk path and the
  name-fallback path; None when ESPN table empty/unmatched.
- board column-contract + API shape tests updated.
- Frontend: build + browser check (default order matches ESPN column, nulls
  at bottom, clicking Rank restores our order).
