"""api/main.py must not load `.env` when it is imported under pytest.

Both cases run in a subprocess against a `.env` in tmp_path that holds a
MARKER variable only -- never a key -- with `api.env.DEFAULT_ENV_FILE`
pointed at it before `api.main` is imported (the loader resolves that path
from the repository, not from the working directory). The subprocess prints
"set" or "unset", never a value.
"""
import os
import subprocess
import sys
import textwrap

MARKER = "LOAD_ENV_GUARD_MARKER"


def _probe(tmp_path, import_pytest_first: bool) -> str:
    (tmp_path / ".env").write_text(f"{MARKER}=marker\n")
    script = textwrap.dedent(f"""
        import os, sys
        {"import pytest" if import_pytest_first else "assert 'pytest' not in sys.modules"}
        os.environ.pop("PYTEST_CURRENT_TEST", None)
        import pathlib
        import api.env
        api.env.DEFAULT_ENV_FILE = pathlib.Path({str(tmp_path / ".env")!r})
        import api.main
        print("set" if {MARKER!r} in os.environ else "unset")
    """)
    env = {**os.environ, "PYTHONPATH": os.getcwd(),
           "DRAFT_DB_PATH": str(tmp_path / "scratch.duckdb"),
           "WARM_ON_BOOT": "0", "SEO_WARM": "0", "DEMO_WARM": "0"}
    env.pop(MARKER, None)
    env.pop("PYTEST_CURRENT_TEST", None)
    out = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip().splitlines()[-1]


def test_importing_main_under_pytest_leaves_the_dotenv_alone(tmp_path):
    assert _probe(tmp_path, import_pytest_first=True) == "unset"


def test_importing_main_outside_pytest_loads_the_dotenv(tmp_path):
    assert _probe(tmp_path, import_pytest_first=False) == "set"


def test_the_suite_itself_never_sees_a_shared_dsn():
    assert "SUPABASE_DB_URL" not in os.environ
