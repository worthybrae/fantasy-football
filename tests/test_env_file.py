"""api.env: the `.env` a laptop uses and a deployment never has.

The two properties worth pinning are the two that could hurt: a file must
never override a variable the platform set, and a missing or malformed one
must be silent. Everything else here is `KEY=VALUE`.
"""
import os

from api.env import load_env_file


def _write(tmp_path, text):
    path = tmp_path / ".env"
    path.write_text(text)
    return path


def test_names_are_returned_and_values_are_set(tmp_path, monkeypatch):
    monkeypatch.delenv("SOME_KEY", raising=False)
    names = load_env_file(_write(tmp_path, "SOME_KEY=some_value\n"))

    # The NAMES, never the values: a caller that reports what it loaded must
    # be able to do it without printing a secret to a terminal.
    assert names == ["SOME_KEY"]
    assert os.environ["SOME_KEY"] == "some_value"


def test_the_environment_always_wins(tmp_path, monkeypatch):
    """The one that matters on a server. A `.env` that got baked into an
    image must not be able to replace a production key with a test one."""
    monkeypatch.setenv("SOME_KEY", "from_the_platform")
    assert load_env_file(_write(tmp_path, "SOME_KEY=from_the_file\n")) == []
    assert os.environ["SOME_KEY"] == "from_the_platform"


def test_comments_blanks_and_junk_are_skipped(tmp_path, monkeypatch):
    monkeypatch.delenv("REAL", raising=False)
    names = load_env_file(_write(tmp_path, """
# a comment

a line with no equals sign
REAL=yes
"""))
    assert names == ["REAL"]


def test_a_shell_paste_works(tmp_path, monkeypatch):
    """`export FOO="bar"` is what a person copies out of a terminal, and
    refusing it would be refusing the most likely input."""
    monkeypatch.delenv("FOO", raising=False)
    load_env_file(_write(tmp_path, 'export FOO="bar baz"\n'))
    assert os.environ["FOO"] == "bar baz"


def test_no_file_is_not_an_error(tmp_path):
    """The ordinary state of every deployment."""
    assert load_env_file(tmp_path / "nothing-here") == []
