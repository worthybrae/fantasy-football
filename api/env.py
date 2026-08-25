"""`.env`, for a local checkout only.

WHY THIS EXISTS. Every secret this project takes -- the custody key, the
Stripe key, the webhook secret -- is read from the environment, because that
is the one place a deployment can hold one without it reaching a git history
(`pipeline/credentials.py` says so at length, and railway.toml repeats it).
That is right for the server and awkward on a laptop, where the alternative
is a shell that has to be `export`ed by hand before every `make api` and
loses the lot the moment somebody opens a new terminal.

WHY NOT python-dotenv. This file is thirty lines and the format is
`KEY=VALUE`. The project has taken exactly two dependencies that are not data
sources, each with a paragraph explaining why nothing else would do; "reads a
text file into a dict" is not that.

THE TWO RULES THAT MAKE IT SAFE:

  * THE ENVIRONMENT ALWAYS WINS. A variable already set is never overwritten.
    On Railway the platform sets them, and a stray `.env` that got baked into
    an image must not be able to quietly replace a production key with a test
    one -- or, far worse, the other way round.
  * IT IS NEVER REQUIRED. No file, unreadable file, malformed line: nothing
    happens and nothing is said. A missing `.env` is the ordinary state of
    every deployment.

`.env` is in `.gitignore` and must stay there. Nothing in this module checks
that, because a check that runs after the commit is not a protection.
"""
from __future__ import annotations

import os
from pathlib import Path

# The repo root, which is where a person puts a `.env` -- this file is
# `api/env.py`, so up two.
DEFAULT_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_env_file(path=None) -> list:
    """Read `KEY=VALUE` lines into the environment. Returns the names set.

    The NAMES, never the values, so a caller that wants to say what it loaded
    can do it without printing a secret key to a terminal or a log.
    """
    path = Path(path) if path is not None else DEFAULT_ENV_FILE
    try:
        text = path.read_text()
    except OSError:
        return []
    loaded = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # `export FOO=bar` is what a person pastes out of a shell session, and
        # refusing it would be refusing the most likely input.
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, sep, value = line.partition("=")
        if not sep:
            continue
        name = name.strip()
        if not name or name in os.environ:
            continue
        value = value.strip()
        # Quotes are a shell habit rather than part of the value: a key pasted
        # as "sk_test_..." is the same key.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ[name] = value
        loaded.append(name)
    return loaded
