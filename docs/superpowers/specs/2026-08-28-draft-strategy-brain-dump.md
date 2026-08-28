# Draft strategy brain dump (owner, 2026-08-28)

Raw notes captured during the scale-out day. Not a design yet; each bullet
needs its own brainstorm before it becomes a spec.

## Onboarding: players the user likes

- When a user signs in, an onboarding flow asks them to pick at least 5 and
  up to 25 players they really like.
- That list is calibration for what the user actually wants, not a ranking
  input.

## Strategy from the pick slot

- Given the user's draft slot, build a strategy that targets as many of the
  liked players as possible.
- For each candidate, show pros and cons relative to who else is available.

## No re-ranking

- Do not re-rank players and do not build our own ranking at all.
- Default to ESPN's ranks; keep showing ADP.
- Ideally show a consensus across all platforms.

## Per-round recommendation

- For every pick, estimate who the user should select in each round (close to
  what exists today).
- Bug: with pick 6 in round 1, the room recommends Jahmyr Gibbs, who is
  clearly the first overall pick and will not be there. The recommendation
  has to be the best player expected to be available at the user's pick, not
  the best player on the board right now.
- Points differential must be computed against who we estimate the user can
  get at their NEXT pick. If the rail is showing running backs when it is the
  user's turn, the differential is against the running backs projected to be
  available at the next-round pick.
- The only genuinely dynamic number is the probability that a player is still
  around at the user's next pick.

## Copy

- Rename "steady" to "reliable" wherever it appears.

## Owner's note on cost

- Dropping our own ranking in favour of ESPN ranks plus an availability
  probability should also cut CPU use a lot (today the live recompute runs
  400 survival rollouts per pick per session).
