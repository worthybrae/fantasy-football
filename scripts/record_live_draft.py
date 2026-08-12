"""Capture a live ESPN draft payload for use as a test fixture.

Run this while sitting in an ESPN mock draft, after several picks have been
made. It answers the one question the live-draft design rests on: does ESPN
populate draftDetail.picks as the draft runs, or only when it ends?
"""
import json
import sys
from pathlib import Path

# Run directly (not via `-m`), so only this file's own directory lands on
# sys.path -- not the repo root, where `pipeline` and `scoring` live. Add it
# before importing either.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scoring.config import CURRENT_SEASON
from pipeline.espn_league import EspnClient, parse_league_id, season_url, _is_real_pick

OUT = Path("tests/fixtures/espn_live_draft.json")


def main(league_url: str) -> int:
    league_id = parse_league_id(league_url)
    with EspnClient() as client:
        payload = client.get_json(season_url(league_id, CURRENT_SEASON))
    detail = payload.get("draftDetail") or {}
    picks = [p for p in (detail.get("picks") or []) if _is_real_pick(p)]
    print(f"draftDetail.drafted = {detail.get('drafted')}")
    print(f"real picks in payload = {len(picks)}")
    if not picks:
        print("\nNo real picks. Either the draft has not started, or ESPN "
              "does not serve picks until completion -- see the plan's Task 1 "
              "fallback before continuing.")
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {OUT} ({len(picks)} picks)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: record_live_draft.py <espn league url or id>")
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
