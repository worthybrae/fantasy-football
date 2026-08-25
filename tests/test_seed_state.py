"""Getting the farm's ESPN login onto a deployment.

THE PROBLEM THIS SOLVES. `pipeline/espn_league.py` reads the farm account's
login from `data/espn_state.json`, a Playwright storage state written by a
browser. A server has no browser and cannot generate one, and a hosting
platform has no file manager to put one there by hand -- so the file arrives
as an environment variable, gzipped and base64'd, and is written to disk at
boot. Rotating an expired login is then editing a variable, not an upload.
"""
import base64
import gzip
import json
import stat

from api import seed_state


STATE = {"cookies": [{"name": "espn_s2", "value": "secret"}], "origins": []}


def _encoded(payload=None) -> str:
    raw = json.dumps(payload if payload is not None else STATE).encode()
    return base64.b64encode(gzip.compress(raw)).decode()


def test_the_variable_becomes_the_file(tmp_path, monkeypatch):
    target = tmp_path / "espn_state.json"
    monkeypatch.setenv(seed_state.STATE_ENV, _encoded())

    assert seed_state.write_state(target) is True
    assert json.loads(target.read_text()) == STATE


def test_plain_base64_works_too(tmp_path, monkeypatch):
    """Gzip is a size optimisation, not a format. Somebody pasting the output
    of `base64 < data/espn_state.json` should get a working deployment rather
    than a puzzling one, so both are accepted and the content decides."""
    target = tmp_path / "espn_state.json"
    monkeypatch.setenv(seed_state.STATE_ENV,
                       base64.b64encode(json.dumps(STATE).encode()).decode())

    assert seed_state.write_state(target) is True
    assert json.loads(target.read_text()) == STATE


def test_the_file_is_not_world_readable(tmp_path, monkeypatch):
    """It is a live ESPN session. The same reasoning as the custody store's
    0600: a credential written at the process umask is 0644 by default, and
    'nobody else is on this container' is an assumption rather than a fact."""
    target = tmp_path / "espn_state.json"
    monkeypatch.setenv(seed_state.STATE_ENV, _encoded())
    seed_state.write_state(target)

    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_no_variable_writes_nothing(tmp_path, monkeypatch):
    """A local checkout has the real file already and no variable set.
    Writing anything here would be this code overwriting the login a person
    generated with their own browser."""
    target = tmp_path / "espn_state.json"
    target.write_text('{"cookies": [{"name": "mine"}], "origins": []}')
    monkeypatch.delenv(seed_state.STATE_ENV, raising=False)

    assert seed_state.write_state(target) is False
    assert "mine" in target.read_text()


def test_the_variable_wins_over_an_existing_file(tmp_path, monkeypatch):
    """Rotation. An expired login on the volume is exactly the case the
    variable exists to fix, so 'the file is already there' must not mean
    'leave the dead session in place'. Set the variable, redeploy, farm
    works -- with no upload and no shell."""
    target = tmp_path / "espn_state.json"
    target.write_text('{"cookies": [{"name": "expired"}], "origins": []}')
    monkeypatch.setenv(seed_state.STATE_ENV, _encoded())

    assert seed_state.write_state(target) is True
    assert "expired" not in target.read_text()


def test_rubbish_does_not_take_the_server_down(tmp_path, monkeypatch):
    """A truncated paste is the likely failure -- these are eleven thousand
    characters through a web form. It must cost the farm and nothing else:
    this runs during app construction, and raising here would turn a mistyped
    variable into a container that will not boot."""
    target = tmp_path / "espn_state.json"
    monkeypatch.setenv(seed_state.STATE_ENV, "not base64 at all!!")

    assert seed_state.write_state(target) is False
    assert not target.exists()


def test_valid_base64_that_is_not_a_login_is_refused(tmp_path, monkeypatch):
    """Decoding is not the same as being right. A state file with no cookies
    in it would let the farm start and fail one room at a time, which is a
    much worse signal than refusing at boot."""
    target = tmp_path / "espn_state.json"
    monkeypatch.setenv(seed_state.STATE_ENV, _encoded({"cookies": [],
                                                       "origins": []}))

    assert seed_state.write_state(target) is False


def test_the_directory_is_made_if_missing(tmp_path, monkeypatch):
    """First boot on an empty volume: `data/` does not exist yet."""
    target = tmp_path / "fresh" / "espn_state.json"
    monkeypatch.setenv(seed_state.STATE_ENV, _encoded())

    assert seed_state.write_state(target) is True
    assert target.is_file()


def test_the_farm_runs_as_its_own_process():
    """Not a function call. In one process DuckDB requires every connection
    to a file to share a configuration, and the farm wants `nfl.duckdb`
    read-only while the app holds it read-write -- which is a refusal, not a
    wait. `mock_farm.open_board_db` already handles the cross-process form of
    this by snapshotting, so a child process is the arrangement that works.

    Measured, in the built image, before this was a child process:
        ConnectionException: Can't open a connection to same database file
        with a different configuration than existing connections
    """
    import inspect

    from api import jobs

    source = inspect.getsource(jobs._run_farm)
    assert "subprocess.run" in source
    assert "pipeline.mock_farm" in source
