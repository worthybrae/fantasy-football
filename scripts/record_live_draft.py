"""Watch a live ESPN draft and record the first payload that carries picks.

Answers the one question the live-draft design rests on: does ESPN populate
`draftDetail.picks` as a draft runs, or only when it ends?

Start this before or during a draft and leave it running. It polls every few
seconds and prints the pick count whenever it changes, so the answer is
observed directly rather than inferred from a single well-timed snapshot.
Once enough real picks are on record it writes the fixture and exits.

    .venv/bin/python scripts/record_live_draft.py "<espn draft url>"

Ctrl-C stops it. If the count never leaves zero while picks are visibly
happening in the browser, that is the negative result -- ESPN is not serving
picks live -- and it is just as useful as the positive one.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.espn_league import (EspnClient, _is_real_pick, parse_league_id,
                                  season_url)
from scoring.config import CURRENT_SEASON

OUT = Path("tests/fixtures/espn_live_draft.json")
POLL_SECONDS = 5
# Enough picks that the fixture exercises ordering and multiple teams, and
# enough that "some picks appeared" cannot be a placeholder artifact.
ENOUGH_PICKS = 8


def _describe(detail: dict) -> tuple:
    picks = [p for p in (detail.get("picks") or []) if _is_real_pick(p)]
    return len(picks), detail.get("drafted")


def main(league_url: str) -> int:
    league_id = parse_league_id(league_url)
    # Printed because parse_league_id falls back to "any run of 3+ digits"
    # when there is no leagueId= parameter, which can silently pick up the
    # wrong number from an unfamiliar URL shape.
    print(f"league id parsed as: {league_id}")
    print(f"polling every {POLL_SECONDS}s -- Ctrl-C to stop\n")

    last = None
    with EspnClient() as client:
        while True:
            try:
                payload = client.get_json(season_url(league_id, CURRENT_SEASON))
            except FileNotFoundError:
                print(f"404 -- ESPN does not serve league {league_id} at this "
                      "path. If this is a lobby mock, it likely has no "
                      "addressable league id; try your own league instead.")
                return 1
            except Exception as e:                       # noqa: BLE001
                print(f"  fetch failed ({e}); retrying")
                time.sleep(POLL_SECONDS)
                continue

            detail = payload.get("draftDetail") or {}
            count, drafted = _describe(detail)
            if count != last:
                print(f"  picks={count:<4} draftDetail.drafted={drafted}")
                last = count

            if count >= ENOUGH_PICKS:
                OUT.parent.mkdir(parents=True, exist_ok=True)
                OUT.write_text(json.dumps(payload, indent=2))
                print(f"\nESPN DOES serve picks live. Wrote {OUT} "
                      f"({count} picks, drafted={drafted}).")
                return 0

            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: record_live_draft.py <espn draft url or league id>")
        raise SystemExit(2)
    try:
        raise SystemExit(main(sys.argv[1]))
    except KeyboardInterrupt:
        print("\nstopped")
        raise SystemExit(130)
