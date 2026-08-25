"""The farm's ESPN login, delivered as an environment variable.

WHY NOT AN UPLOAD. `pipeline/espn_league.py` reads the farm account's login
from `data/espn_state.json` -- a Playwright storage state, written by an
actual browser doing an actual sign-in. A server has no browser, so it cannot
make one; and a hosting platform has no file manager, so putting one on a
volume by hand means a shell session and a heredoc, repeated every time the
session expires or the volume is recreated.

An environment variable is the same channel every other secret in this
deployment already uses, it is versioned by the platform, and rotating an
expired login becomes "paste a new value, redeploy" rather than "find the
CLI, remember the flags, hope the shell is up".

WHY IT IS NOT SIMPLY THE JSON. Two reasons. The file is 13 KB of JSON full
of quotes, braces and newlines, which is exactly the shape that a web form,
a shell, or a CI runner will mangle in some small way that is very annoying
to find. And gzipped it is 11 KB of base64 instead of 18 KB, which matters
against the size limits variable stores tend to have.
"""
from __future__ import annotations

import base64
import binascii
import gzip
import json
import os
from pathlib import Path

# The variable itself. Named after what it holds rather than what it is for,
# so it reads correctly in a settings list next to ESPN_CUSTODY_KEYS.
STATE_ENV = "FARM_ESPN_STATE_B64"


def _decode(raw: str) -> dict | None:
    """The value, as a storage state, or None if it is not one.

    ACCEPTS GZIPPED OR PLAIN base64. Gzip is a size optimisation, not a
    format: somebody who pastes the output of `base64 < data/espn_state.json`
    should get a working deployment rather than a puzzling one, so the
    content decides and both work.
    """
    try:
        blob = base64.b64decode(raw.strip(), validate=True)
    except (binascii.Error, ValueError):
        return None
    # The gzip magic number. Cheaper and more honest than decompressing
    # inside a try and reading the exception.
    if blob[:2] == b"\x1f\x8b":
        try:
            blob = gzip.decompress(blob)
        except OSError:
            return None
    try:
        state = json.loads(blob)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(state, dict):
        return None
    # DECODING IS NOT THE SAME AS BEING RIGHT. A state file with no cookies
    # in it parses perfectly and lets the farm start, then fail one room at a
    # time with an authentication error -- a much worse signal than refusing
    # here, where the reason is a single line in the boot log.
    if not state.get("cookies"):
        return None
    return state


def write_state(target: Path | str | None = None) -> bool:
    """Write the login from the environment, if there is one. Returns whether.

    THE VARIABLE WINS OVER AN EXISTING FILE, on purpose. An expired login
    sitting on the volume is precisely the case this exists to fix, so "there
    is already a file there" must not mean "leave the dead session in place".
    With no variable set, nothing is touched at all -- which is what makes
    this safe to call on a laptop, where the real file was made by the
    developer's own browser and must not be overwritten by anything.

    IT NEVER RAISES. This runs during app construction, and these values are
    eleven thousand characters pasted through a web form: a truncated paste
    is the likely mistake, and it must cost the farm rather than the whole
    container. A deployment with a bad value comes up serving the site, with
    the farm refusing to start and saying why.
    """
    raw = os.environ.get(STATE_ENV)
    if not raw or not raw.strip():
        return False

    from pipeline.espn_league import STATE_PATH
    path = Path(target) if target is not None else Path(STATE_PATH)

    state = _decode(raw)
    if state is None:
        print(f"{STATE_ENV} is set but is not a usable ESPN storage state "
              f"(expected base64, optionally gzipped, of a JSON object with "
              f"cookies in it). The farm will not start.", flush=True)
        return False

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # 0600 BEFORE THE CONTENT. `touch` then `chmod` then `write` leaves no
        # window in which a live ESPN session is on disk at the process
        # umask, which is 0644 on a default machine -- the same reasoning
        # `pipeline/credentials.py` applies to the custody database, and for
        # the same kind of secret.
        path.touch(mode=0o600, exist_ok=True)
        path.chmod(0o600)
        path.write_text(json.dumps(state))
    except OSError as exc:
        print(f"Could not write {path}: {exc}. The farm will not start.",
              flush=True)
        return False

    print(f"Wrote the farm's ESPN login to {path} from {STATE_ENV} "
          f"({len(state.get('cookies') or [])} cookies).", flush=True)
    return True
