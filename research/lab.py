"""The research lab: run an experiment, keep its numbers, write it up.

Every claim this project makes about its own model should be traceable to a
measurement someone can re-run. That is not a general principle, it is a
reaction to how this codebase's own analysis has actually gone wrong: over
one afternoon the schedule factor was called "essentially noise" (it was
not), the value of preseason betting lines was projected at 0.272 (measured:
0.337), and an O-line rating was declared to have no effect on production
(it has a real one, on efficiency -- the first test just measured the wrong
outcome). Each of those was a number stated from memory or from a model
rather than read off a result.

So the rule here is mechanical: a post renders from an experiment's returned
`findings` dict, by key. `{r_blend}` in prose resolves against that dict or
`render` raises. Prose cannot drift from data, because prose cannot restate
data -- it can only reference it.

Results are written to `research/results/<id>.json` with the git revision
they were produced at, so a number in a published post can always be tied to
the code that produced it.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
POSTS = ROOT / "posts"


def _git_rev() -> str | None:
    """The revision the numbers were produced at, or None outside a repo."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


@dataclass(frozen=True)
class Experiment:
    """One question, one measurement, one write-up.

    `question` is the thing being asked, in a sentence, before any answer is
    known -- written first on purpose, so an experiment cannot be quietly
    reframed around whatever it happened to find.

    `run` returns a flat dict of measured values. Flat because the post
    template resolves `{key}` against it directly, and a nested structure
    would invite formatting logic to creep into prose.
    """
    id: str
    title: str
    question: str
    run: Callable[..., dict]
    tags: tuple[str, ...] = ()
    # Set when the answer contradicts what the project currently does or
    # believed. These are the posts worth reading, and worth flagging.
    overturns: str | None = None


@dataclass
class Result:
    experiment: Experiment
    findings: dict
    ran_at: str
    git_rev: str | None = None
    notes: list[str] = field(default_factory=list)

    def save(self) -> Path:
        RESULTS.mkdir(parents=True, exist_ok=True)
        path = RESULTS / f"{self.experiment.id}.json"
        path.write_text(json.dumps({
            "id": self.experiment.id,
            "title": self.experiment.title,
            "question": self.experiment.question,
            "tags": list(self.experiment.tags),
            "overturns": self.experiment.overturns,
            "ran_at": self.ran_at,
            "git_rev": self.git_rev,
            "findings": self.findings,
            "notes": self.notes,
        }, indent=2, sort_keys=True, default=str))
        return path

    @staticmethod
    def load(experiment_id: str) -> dict:
        return json.loads((RESULTS / f"{experiment_id}.json").read_text())


def run(experiment: Experiment, *args, **kwargs) -> Result:
    """Run an experiment and persist what it measured."""
    findings = experiment.run(*args, **kwargs)
    if not isinstance(findings, dict):
        raise TypeError(
            f"{experiment.id}.run must return a dict of measured values, "
            f"got {type(findings).__name__} -- the post template resolves "
            f"its placeholders against that dict")
    notes = list(findings.pop("_notes", []))
    result = Result(experiment=experiment, findings=findings, notes=notes,
                    ran_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    git_rev=_git_rev())
    result.save()
    return result


class MissingFinding(KeyError):
    """A post referenced a number the experiment did not measure.

    Loud on purpose. The whole point of rendering from findings is that a
    number in prose is a number that was measured; silently rendering an
    empty span would defeat it entirely, and quietly rendering the literal
    `{key}` would ship a broken sentence.
    """


def resolve(text: str, findings: dict) -> str:
    """Substitute `{key}` in prose from measured findings.

    Deliberately not str.format: that would also interpret `{}`, `{0}` and
    format specs, and would fail confusingly on any literal brace in copy.
    This only ever replaces whole `{identifier}` tokens.
    """
    import re

    def sub(match):
        key = match.group(1)
        if key not in findings:
            raise MissingFinding(
                f"post references {{{key}}} but the experiment measured "
                f"{sorted(findings)}")
        value = findings[key]
        return f"{value:g}" if isinstance(value, float) else str(value)

    return re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", sub, text)
