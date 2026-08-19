# Draft room: pick from the helper, rank in points

Three problems, reported together after a live draft:

1. Opening Draft Helper disconnects the user from the ESPN draft room, so no
   picks can be made at all.
2. Recommendations push real NFL starters several rounds before their ADP.
3. The live UI is not usable under a pick clock.

They are one piece of work because the fix for (1) determines the shape of the
UI in (3), and the numbers in (2) are what that UI displays.

## 1. Why the ESPN session breaks

`pipeline/draft_socket.socket_url` mints a JOIN URL that identifies the
caller as a specific team:

```
wss://fantasydraft.espn.com/game-1/league-<id>/JOIN
  ?1=1&2=<leagueId>&3=<teamId>&4=<swid>
  &5=1:<leagueId>:<teamId>:<swid>:<token>&6=false&7=false&8=KONA
```

ESPN permits one draft session per team. When `api/live.py` opens that socket,
ESPN treats it as the team's session and drops the browser's. This is a
regression introduced with the browserless socket (`bda628e`); the previous
listener piggybacked on the window the human was drafting in (`d0e659b`).

The session cannot be shared, so the tool has to own it and give the user a
way to pick. The recorded capture already shows the outbound frame that does
this — `tests/fixtures/espn_draft_socket.jsonl`:

```
ws-send | SELECT 4429795
ws-recv | SELECTED 2 4429795 2 {8491403C-A53F-4257-8D52-F8AE32CED897}
```

`SELECT <espnPlayerId>` makes the pick; ESPN answers with `SELECTED <pick>
<espnPlayerId> <teamId> <swid>`. The same socket also carries `CLOCK`,
`SELECTING`, and `AUTODRAFT`, which is everything a draft room needs to
render.

So: the ESPN draft room is used once, to launch the bookmarklet. Everything
after that happens in Draft Helper.

## 2. Why the recommendations reach

`scoring/factors.py:103` normalizes every scoring factor to a within-position
percentile:

```python
out[out_col] = (out.groupby("position")[raw_col].rank(pct=True) * 100)
```

`composite` is a weighted blend of those percentiles, so it is a 0–100
within-position quality score. `scoring/composite.py:31` then builds VOR from
it:

```python
out.loc[grp.index, "vor"] = grp["composite"] - replacement
```

That is a difference of two within-position percentiles. `scoring/board.py:340`
sorts the whole board across positions by that number, which compares
quantities that have no common unit. Being twelve percentile points above TE10
scores the same as being twelve percentile points above RB22.

The effect is visible in the current board: Mark Andrews ranks 13th overall at
ADP 149, Travis Kelce 19th at ADP 104, and three defenses rank 29th–31st at ADP
176–258.

Every downstream consumer inherits it. `scoring/draft_sim._candidate_indices`
draws half its candidate slots by VOR, so the simulator's candidate set is
seeded with reaches before a single rollout runs.

There is a second, smaller problem on top. `search_pick` ranks its twelve
candidates by expected end-of-draft roster value, and that estimate is noisy
relative to the spread it is trying to resolve — the module's own docstring
records `ev=1332.449153, se=7.406298`. When several candidates sit within one
standard error of each other, the top slot is decided by sampling noise.

## 3. The value model

### Projected points as the base unit

`scoring/draft_sim.projections()` already computes projected season points per
`player_id`. Promote it out of the simulator into a place the board can use,
and add `proj_points` as a board column.

`scoring/draft_sim._replacement_points()` already computes the projected points
of the last starter-caliber player at each position, derived from
`LeagueSettings`. Reuse it.

Redefine VOR in points:

```
vor_points(p) = proj_points(p) - replacement_points[position(p)]
```

Keep `composite` and the five factors exactly as they are. They are legitimate
within-position quality signals and the player profile presents them that way.
What changes is that VOR is no longer built from them, and the board no longer
ranks across positions on a percentile difference.

`apply_vor` keeps its current behavior under a new name for the profile's
within-position use, or gains a parameter naming the column it differences.
Either way `assign_tiers` continues to run per position, where percentile
differences are meaningful.

### Gain now: value over the next man up

Raw VOR still over-values quarterbacks, because a starting QB scores more
points than a starting running back and the gap to QB9 is large in absolute
terms even when the gap to QB10 is not. The number that matters on the clock is
what you lose by waiting.

For each available player `p` at position `q`:

```
gain_now(p) = need_weight(q) * (vor_points(p) - expected_best_at_next_turn(q))
```

`expected_best_at_next_turn(q)` is the survival-weighted best `vor_points`
among players at position `q` who are still available at the user's next pick.
`scoring/draft_sim.survival()` already produces per-player availability
probabilities for exactly that pick, and it is a far lower-variance quantity
than end-of-draft roster value: it counts occurrences of a single event rather
than averaging a sum over fifteen simulated picks.

```
expected_best_at_next_turn(q) = sum over players r at q, in descending
    vor_points order, of vor_points(r) * P(r survives) * P(no better r
    survives)
```

`need_weight(q)` reflects what the roster can still use:

| Roster state for `q`                    | Weight |
| --------------------------------------- | ------ |
| A starting slot at `q` is open           | 1.0    |
| Only a FLEX slot can take `q`            | 0.75   |
| Starters and FLEX full, bench open       | 0.35   |
| Position at its roster cap               | 0.0    |

The weights are a starting point to calibrate against replayed drafts, not a
derived constant. They are one dict in `scoring/config.py`.

The available list sorts by `gain_now` descending. The top three rows are the
recommendation.

### What happens to the simulator

`search_pick` stops driving the live recommendation. `survival()` stays and
becomes the only simulator call on the live path, which also makes the
recompute cheap enough to run on every pick without a rollout budget that
shrinks as the clock runs down. `search_pick` and `run_sim` remain for offline
analysis; nothing about them changes.

This is a deliberate trade. The simulator models opponent behavior with fitted
manager coefficients, and `gain_now` uses that model only through the survival
probabilities. In exchange the ranking becomes deterministic, sub-100ms, and
explainable in one sentence per row.

## 4. Sending a pick

### Reaching the socket

`run_socket_listener` owns its websocket in a local variable and replaces it on
every reconnect. To send on it, introduce a small handle:

```python
class SocketHandle:
    """The live connection, as much of it as a sender needs.

    `run_socket_listener` swaps the underlying socket on reconnect; this
    holds the current one behind a lock so a send never races a reconnect
    into a closed socket.
    """
    def send(self, text: str) -> None: ...
    def alive(self) -> bool: ...
```

`run_socket_listener` gains an `on_socket` callback, matching the shape of its
existing `on_change` and `on_activity` seams, and publishes the handle through
it on each successful connect. `api/live.py` records it in `state` alongside
`listener` and clears it on stop.

### The endpoint

`POST /api/live/select` with `{"player_id": "..."}`:

1. Reject unless the session is active and `session.my_slot` is on the clock.
   Sending `SELECT` off-turn is at best ignored and at worst a misdraft.
2. Resolve `player_id` to an ESPN id. Board rows carry `espn_id`; D/ST rows
   carry `None` and are reconstructed with
   `pipeline.espn_live._dst_espn_id(pro_team_id)`, the same negative-id rule
   `build_crosswalk` already relies on. A player that resolves to nothing is a
   400 naming the player, never a guessed id.
3. Reject if the player is already in `drafted`.
4. `handle.send(f"SELECT {espn_id}\n")`.
5. Wait for the listener to observe a `SELECTED` frame for that ESPN id, up to
   a timeout of about eight seconds. Return the confirmed pick number.

The endpoint returns only what ESPN confirmed. On timeout it returns 504 with
the player still marked undrafted locally, because the pick may or may not have
landed and the board must not claim otherwise. The UI tells the user to check
ESPN in that case.

Nothing is written to `drafted` by this endpoint. The existing listener path
writes it when the `SELECTED` frame arrives, exactly as it does for every other
team's pick, so there is one writer and no divergence between what the tool
believes and what ESPN did.

## 5. The draft room

A new route, `/draft`. `Landing` and `/players/<slug>` are untouched.

Layout, no page scroll:

```
┌──────────────────────────────────┬──────────────┐
│ [ AVAILABLE ] [ SNAKE BOARD ]    │  CLOCK       │
│                                  │  pick / next │
│  AVAILABLE:                      │  gap         │
│  ┌────────────────────────────┐  ├──────────────┤
│  │ TAKE ONE OF THESE (top 3)  │  │  MY ROSTER   │
│  └────────────────────────────┘  │  open slots  │
│  ranked list, scrolls            │  highlighted │
│  search · position filters       │              │
└──────────────────────────────────┴──────────────┘
```

The right rail is pinned and never moves. The main area toggles between the two
views, and switches itself to Available the moment the user goes on the clock.

Available list columns: rank, position, player, projected points, points over
replacement, gain now, chance he lasts to your next pick, ADP, which slot he
fills, draft button. The list sorts on gain now.

Clicking a draft button opens a confirmation dialog naming the player, the slot
he fills, and the gain. Confirming calls `POST /api/live/select`. The dialog
shows sending, confirmed, or failed; there is no optimistic state, because the
board must only ever show what ESPN confirmed.

The visual system is the existing one in `web/src/index.css` — the dark
three-elevation surface set, the position hues, the mono numerics. The problem
was hierarchy and density, not tokens.

A published mock of the room is at
https://claude.ai/code/artifact/ae269a33-666c-429c-b7ce-40018ba6d691

## 6. Error handling

- **Socket dropped while on the clock.** The clock panel says so and the draft
  buttons disable. `run_socket_listener` already reconnects; the buttons
  re-enable when the handle reports alive again.
- **Token expired.** Already surfaced through `listener_error`. The draft room
  shows it in place of the clock, with the instruction to click the bookmarklet
  again.
- **`SELECT` sent, no `SELECTED` back.** 504, board unchanged, user told to
  check ESPN. This is the one case the tool cannot resolve on its own.
- **Player has no ESPN id.** 400 naming the player. This is a crosswalk gap and
  should also appear in the existing unmapped-picks surface.
- **Clicked off-turn.** 409. The button should already be disabled, so this is
  a guard against a stale page, not an expected path.
- **Projections missing for a player.** `draft_sim` already floors these per
  position rather than producing NaN. That floor moves with `projections()` and
  keeps its current behavior.

## 7. Testing

The value model is pure and gets unit tests:

- `vor_points` is in points, and a top TE does not outrank a top RB when the
  RB's points over replacement is larger. This is the regression test for the
  Mark Andrews case, written against a fixture board.
- `expected_best_at_next_turn` falls to zero when no player at a position
  survives, and equals the best player's `vor_points` when survival is
  certain.
- `gain_now` is zero for a position at its roster cap, and ranks a scarce
  position above an abundant one at equal `vor_points`.
- A quarterback whose next man up is nearly as good ranks below a receiver with
  a smaller raw VOR but a real cliff behind him. This is the Josh Allen case.

The socket send gets tests against the existing fake-socket seam
(`pipeline.draft_socket._connect` is already monkeypatched in tests):

- `SocketHandle.send` after a reconnect goes to the new socket.
- `POST /api/live/select` off-turn is a 409 and sends nothing.
- A D/ST resolves to its negative ESPN id.
- A `SELECTED` frame for the requested id resolves the request; a different id
  does not.
- Timeout returns 504 and leaves `drafted` untouched.

The room itself gets the same treatment the existing live board has in
`tests/test_live_api.py`: state shapes and transitions, not pixel assertions.

## 8. Order of work

1. Projected points on the board; VOR in points; board ranks on it.
2. `gain_now` and the survival-weighted next-man-up, served from
   `/api/live/state`.
3. `SocketHandle`, `on_socket`, `POST /api/live/select`.
4. The `/draft` room.

The UI is a shell around the numbers, so the numbers come first. Step 3 is the
one that unblocks drafting at all, and can be pulled forward if a draft is
imminent.

## Out of scope

- Changing how `composite` or the five factors are computed. They stay
  within-position percentiles.
- Auto-draft, queueing, or trade logic.
- Replacing `search_pick` or `run_sim` for offline analysis.
- Any change to `Landing` or the player profile.
