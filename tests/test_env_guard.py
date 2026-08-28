"""api/main.py must not load `.env` when it is imported under pytest."""
import os
import subprocess
import sys
import textwrap


def test_importing_main_under_pytest_leaves_the_dotenv_alone(tmp_path):
    (tmp_path / ".env").write_text("SUPABASE_DB_URL=postgresql://nobody@nowhere/x\n"
                                   "STRIPE_SECRET_KEY=sk_test_should_not_load\n")
    script = textwrap.dedent("""
        import os, sys
        import pytest                      # what the runner has imported
        os.environ.pop("PYTEST_CURRENT_TEST", None)
        import api.main
        print(os.environ.get("SUPABASE_DB_URL", "<unset>"))
        print(os.environ.get("STRIPE_SECRET_KEY", "<unset>"))
    """)
    env = {**os.environ, "PYTHONPATH": os.getcwd(),
           "DRAFT_DB_PATH": str(tmp_path / "scratch.duckdb"),
           "WARM_ON_BOOT": "0", "SEO_WARM": "0", "DEMO_WARM": "0"}
    env.pop("SUPABASE_DB_URL", None); env.pop("STRIPE_SECRET_KEY", None)
    out = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.split() == ["<unset>", "<unset>"]


def test_the_suite_itself_never_sees_a_shared_dsn():
    assert "SUPABASE_DB_URL" not in os.environ
