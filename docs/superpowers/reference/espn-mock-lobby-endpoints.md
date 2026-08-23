# ESPN mock lobby: verified endpoints (probed live 2026-08-23 05:23)

All of this was confirmed against the live ESPN API with the cookies now in
`data/espn_state.json`. Do not re-derive it; do verify responses at runtime.

## 1. List open mock rooms — CONFIRMED WORKING

```
GET https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}/leaguedirectory/MOCKDRAFT_LOBBY
```

Authenticated with the `espn_s2` + `SWID` cookies via
`pipeline.draft_socket.load_cookies()` / `http_fetch()`. Returned a JSON
array of 234 rooms.

The subType path segment is the NAME, not the numeric id. For the record,
`MOCKDRAFT_LOBBY` is subType id 4; `DRAFT_LOBBY` is 3, `CUSTOM_MOCK` is 5.

Each row's useful fields:

| field | meaning |
|---|---|
| `leagueId` | int, what you join and what the socket needs |
| `leagueSize` | seats in the room |
| `teamsJoined` | seats taken |
| `full` | bool |
| `draftInProgress` | bool |
| `draftDate` | epoch MILLISECONDS, when picking starts |
| `draftAvailableDate` | epoch ms, when the room opens (~90s before draftDate) |
| `draftType` | `SNAKE` in every row observed |
| `scoringType` | `H2H_POINTS` in every row observed |

Live measurements from that one call:

- 234 rooms; 186 not full; 94 already drafting; 140 with a future draftDate
- **137 joinable right now** (not full, draft still ahead)
- Rooms start in batches roughly every 3 minutes, all night
- `leagueSize` distribution: 10 (91), 12 (76), 8 (29), 14 (12), 18 (13),
  16 (7), 20 (6)

## 2. Join a room — CONFIRMED WORKING (probed live, HTTP 201)

```
POST https://lm-api-writes.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}/segments/0/leagues/{leagueId}/invites?memberId={SWID}
```

**The exact call, verified against the live API:**

```
POST https://lm-api-writes.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}/segments/0/leagues/{leagueId}/invites
  ?memberId={SWID}&join=true
  content-type: application/json
  body: [{"teamId": -1}]
```

`join=true` is REQUIRED. Without it the call returns HTTP 400 with an HTML
error body and no explanation — that was the first thing tried and it cost a
round of guessing. `teamId: -1` means "any open seat"; the body is a JSON
ARRAY, not an object.

Response is `201` with the assigned seat:

```json
[{"isDeleted": false, "teamId": 7}]
```

Read `teamId` from `response[0]["teamId"]`. Do NOT assume a seat number and
do not fall back to a follow-up `?view=mTeam` read — the response carries it.

Headers: the same `DRAFT_SECURITY_HEADERS` already in `draft_socket.py`, plus
`content-type: application/json`.

**Verified continuation:** with that `teamId`,
`draft_security_token(fetch, leagueId, teamId, 2026)` returned a real nonce
and `socket_url(...)` minted a well-formed `wss://fantasydraft.espn.com/...`
JOIN url. The whole chain — directory, join, draftSecurity, socket url —
works today with the cookies on disk. Nothing about the transport is
unknown any more; what remains is the loop around it.

## 3. Then the existing code takes over

```
GET .../seasons/{season}/segments/0/leagues/{leagueId}/teams/{teamId}/draftSecurity
```
`pipeline/draft_socket.draft_security_token()` — already implemented.
Then `socket_url()` and `run_socket_listener()` — already implemented.

## CONSEQUENCES FOR THE PLAN

**League size is NOT 8.** Plan Task 1 correctly assumes 8x16 for the drafts
already on disk (verified from the trace). But the live lobby serves rooms
from 8 to 20 seats. The farm loop MUST read each room's real settings —
`leagueSize` from the directory row, roster/rounds from the league's own
`?view=mSettings` — and must never hardcode 8. A draft recorded with the
wrong team count has every slot attributed to the wrong seat, which silently
poisons the corpus rather than failing.

**Timing is the loop's real constraint.** A room is enterable at
`draftAvailableDate` and picks start at `draftDate`. The loop should select a
room whose `draftDate` is a little ahead, join before it starts, and hold the
socket through the draft.

**Joining a room takes a seat from a real person.** Prefer rooms that already
have several `teamsJoined` over empty ones, so the bot fills out a draft
people are waiting on rather than squatting an empty room.

## ROOM SELECTION POLICY (owner's explicit instruction, 2026-08-23)

**Farm 8-person PPR snake mocks only.** `scoring/config.py` sets
`LEAGUE_TEAMS = 8`, the drafts already on disk are 8x16, and the owner asked
for this directly. Holding team count and scoring format fixed means the
fitted prior is not averaging over league shapes it will never be used on.

Filter a directory row in, requiring ALL of:

```python
row["leagueSize"] == 8
row["draftType"] == "SNAKE"
row["rankType"] == "PPR"
53 in row["scoringItemStatIds"]     # receptions; STANDARD rooms omit it
not row["full"]
row["draftDate"] > now_ms
```

`rankType` and stat id 53 are independent signals that agree in every row
observed. Require both: a room whose name says PPR but whose scoring lacks
receptions would poison the corpus with standard-scoring behavior, and a
single field could change meaning under us.

Rank the survivors by, in order:

1. **`teamsJoined` descending.** A room at 6/8 is six humans waiting for a
   draft; a room at 0/8 will be filled by ESPN autodraft and teaches us
   nothing but ADP. This is the single most important ordering key — it is
   the corpus-quality question the spec's phase 0 was built to answer.
2. **`experienceType`**: PRO and EXPERT before BEGINNER.
3. **soonest `draftDate`**, to keep the loop busy.

### Measured supply (snapshot at 2026-08-23 05:23)

- 14 of 234 rooms were 8-team PPR snake; 12 joinable
- experienceType among them: PRO 8, EXPERT 2, BEGINNER 4
- **new 8-team PPR starts every ~5 minutes**, all night
- most were 0/8 or 1/8 joined — so the `teamsJoined` ordering above is doing
  real work, and the corpus report in Task 2 must be read before trusting
  any prior fitted on these

A draft is 128 picks. One session at a time is the constraint, not room
supply. If the loop turns out to be the bottleneck, running 2-3 concurrent
sessions is the lever — but only after one has been observed working
end to end.
