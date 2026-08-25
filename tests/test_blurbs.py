import json

import pytest


class _Block:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Response:
    def __init__(self, text, stop_reason="end_turn"):
        self.content = [_Block(text)]
        self.stop_reason = stop_reason
        self._request_id = "req_test"


class FakeClient:
    """Answers each `messages.create` from a queue of responses or exceptions."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []
        outer = self

        class _Messages:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                answer = outer.answers.pop(0)
                if isinstance(answer, Exception):
                    raise answer
                return answer
        self.messages = _Messages()


def _facts():
    return {
        "league_id": "1", "season": 2026, "league_name": "The League", "teams": 2,
        "status": "ready", "intro": None,
        "power_rankings": [
            {"rank": 1, "manager": "ann", "team_name": "Ann's", "score": 1.0, "first_year": False, "line": None},
            {"rank": 2, "manager": "bob", "team_name": "Bob's", "score": -1.0, "first_year": True, "line": None}],
        "report_cards": [
            {"manager": "ann", "team_name": "Ann's", "team_id": 1, "grade": "A",
             "value_total": 30.0, "value_per_pick": 3.0, "graded_picks": 10, "steals": 2, "reaches": 0,
             "best_pick": {"player_name": "X", "position": "RB", "round": 3, "overall_pick": 5,
                           "market_rank": 20.0, "value": 15.0, "verdict": "steal"},
             "worst_pick": None, "shape": {"positions": {"RB": 5}, "first_round": {"QB": 4}},
             "nickname": None, "blurb": None},
            {"manager": "bob", "team_name": "Bob's", "team_id": 2, "grade": "F",
             "value_total": -30.0, "value_per_pick": -3.0, "graded_picks": 10, "steals": 0, "reaches": 3,
             "best_pick": None, "worst_pick": None, "shape": {"positions": {}, "first_round": {}},
             "nickname": None, "blurb": None}],
        "profiles": [
            {"manager": "ann", "team_name": "Ann's", "seasons": [], "titles": [2024], "playoffs": [2024],
             "completed": 1, "win_pct": 0.7, "avg_finish": 1.0, "ppg": 120.0, "drafts": 2,
             "mean_value": 2.0, "steal_rate": 0.2, "reach_rate": 0.0,
             "first_pick_positions": {"RB": 2}, "career_best": None, "career_worst": None,
             "model_notes": ["reaches for tight ends"]},
            {"manager": "bob", "team_name": "Bob's", "seasons": [], "titles": [], "playoffs": [],
             "completed": 0, "win_pct": None, "avg_finish": None, "ppg": None, "drafts": 0,
             "mean_value": None, "steal_rate": None, "reach_rate": None,
             "first_pick_positions": {}, "career_best": None, "career_worst": None,
             "model_notes": []}],
    }


def _good():
    return json.dumps({"intro": "What a night.",
                       "cards": [{"manager": "ann", "nickname": "The Bandit", "blurb": "Stole X."},
                                 {"manager": "bob", "nickname": "Rookie", "blurb": "First draft."}],
                       "rankings": [{"manager": "ann", "line": "Top."},
                                    {"manager": "bob", "line": "Bottom."}]})


def test_write_blurbs_makes_one_structured_call():
    from scoring import blurbs
    client = FakeClient([_Response(_good())])
    out = blurbs.write_blurbs(client, _facts())
    assert out["intro"] == "What a night."
    assert [c["manager"] for c in out["cards"]] == ["ann", "bob"]
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["model"] == "claude-haiku-4-5"
    assert call["max_tokens"] == 4096
    assert "thinking" not in call
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["output_config"]["format"]["schema"]["additionalProperties"] is False
    assert call["system"] == blurbs.SYSTEM
    body = call["messages"][0]["content"]
    assert "ann" in body and "bob" in body and "The Bandit" not in body


def test_compact_facts_drops_what_the_model_does_not_need():
    from scoring.blurbs import compact_facts
    compact = compact_facts(_facts())
    assert set(compact) == {"league_name", "season", "teams", "power_rankings", "report_cards", "profiles"}
    card = compact["report_cards"][0]
    assert "team_id" not in card and "nickname" not in card and "blurb" not in card
    assert card["grade"] == "A" and card["best_pick"]["player_name"] == "X"
    prof = compact["profiles"][0]
    assert prof["model_notes"] == ["reaches for tight ends"] and "seasons" in prof


def test_missing_manager_is_retried_once_with_the_failure_named():
    from scoring import blurbs
    short = json.dumps({"intro": "x", "cards": [{"manager": "ann", "nickname": "n", "blurb": "b"}],
                        "rankings": [{"manager": "ann", "line": "l"}]})
    client = FakeClient([_Response(short), _Response(_good())])
    out = blurbs.write_blurbs(client, _facts())
    assert out is not None and len(out["cards"]) == 2
    assert len(client.calls) == 2
    retry = client.calls[1]["messages"]
    assert retry[0]["role"] == "user" and retry[1]["role"] == "assistant"
    assert retry[2]["role"] == "user" and "bob" in retry[2]["content"]


def test_two_bad_answers_yield_none():
    from scoring import blurbs
    client = FakeClient([_Response("not json"), _Response("{}")])
    assert blurbs.write_blurbs(client, _facts()) is None
    assert len(client.calls) == 2


def test_unknown_manager_fails_validation():
    from scoring import blurbs
    extra = json.loads(_good())
    extra["cards"].append({"manager": "zed", "nickname": "n", "blurb": "b"})
    client = FakeClient([_Response(json.dumps(extra)), _Response(json.dumps(extra))])
    assert blurbs.write_blurbs(client, _facts()) is None


def test_api_errors_yield_none_and_do_not_raise():
    import anthropic
    from scoring import blurbs
    err = anthropic.APIConnectionError(request=None)
    client = FakeClient([err])
    assert blurbs.write_blurbs(client, _facts()) is None


def test_refusal_yields_none():
    from scoring import blurbs
    client = FakeClient([_Response("", stop_reason="refusal")])
    assert blurbs.write_blurbs(client, _facts()) is None


def test_none_client_is_skipped():
    from scoring import blurbs
    assert blurbs.write_blurbs(None, _facts()) is None


def test_default_client_off_without_a_key(monkeypatch):
    from scoring import blurbs
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert blurbs.default_client() is None and not blurbs.enabled()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert blurbs.enabled() and blurbs.default_client() is not None
