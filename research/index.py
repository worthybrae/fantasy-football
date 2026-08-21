"""The lab's front page: every experiment, its question, and its answer.

Built from stored results rather than a hand-kept list, so an experiment
that has never been run cannot appear here claiming a finding, and one whose
numbers changed cannot show stale ones. Same rule as the posts.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research.lab import POSTS, RESULTS                          # noqa: E402

SHELL = """<title>Research</title>
<style>
  :root {{
    --bg-0:#0a0b0e; --bg-1:#121319; --bg-2:#1b1d25; --border:#262932;
    --text-1:#e8eaef; --text-2:#98a0b0; --text-3:#565b69;
    --accent:#ffb454; --fail:#d03b3b;
    --font-ui:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif;
    --font-mono:'JetBrains Mono',ui-monospace,'SFMono-Regular',Menlo,monospace;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg-0); color:var(--text-2);
         font:15px/1.6 var(--font-ui); -webkit-font-smoothing:antialiased; }}
  .wrap {{ max-width:780px; margin:0 auto; padding:56px 24px 90px; }}
  h1 {{ font-size:30px; color:var(--text-1); margin:0 0 10px; font-weight:700;
        letter-spacing:-.02em; }}
  .sub {{ color:var(--text-2); margin:0 0 40px; max-width:60ch; }}
  .row {{ display:block; text-decoration:none; border:1px solid var(--border);
          border-radius:6px; background:var(--bg-1); padding:18px 20px;
          margin:0 0 12px; transition:border-color .12s; }}
  .row:hover {{ border-color:#3a4050; }}
  .id {{ font-family:var(--font-mono); font-size:11px; letter-spacing:.09em;
         color:var(--text-3); }}
  .t {{ font-size:17px; color:var(--text-1); font-weight:600; margin:5px 0 6px;
        letter-spacing:-.01em; }}
  .q {{ font-size:13.5px; color:var(--text-2); margin:0; }}
  .flag {{ display:inline-block; font-family:var(--font-mono); font-size:10px;
           letter-spacing:.07em; text-transform:uppercase; color:var(--fail);
           border:1px solid var(--fail); border-radius:3px; padding:1px 6px;
           margin-left:8px; vertical-align:2px; }}
  .tags {{ font-family:var(--font-mono); font-size:11px; color:var(--text-3);
           margin-top:9px; }}
  .none {{ color:var(--text-3); font-style:italic; }}
</style>
<div class="wrap">
  <h1>Research</h1>
  <p class="sub">Experiments testing what this draft tool believes about its own
  model. Every number in a post is read off the code that produced it &mdash; the
  prose is a template, and a figure that was never measured raises rather than
  renders.</p>
{rows}
</div>
"""


def build() -> Path:
    rows = []
    for path in sorted(RESULTS.glob("*.json")):
        r = json.loads(path.read_text())
        flag = '<span class="flag">overturns</span>' if r.get("overturns") else ""
        tags = " &middot; ".join(r.get("tags", []))
        rows.append(
            f'  <a class="row" href="{r["id"]}.html">\n'
            f'    <div class="id">{r["id"].upper()} &middot; {r["ran_at"][:10]}</div>\n'
            f'    <div class="t">{html.escape(r["title"])}{flag}</div>\n'
            f'    <p class="q">{html.escape(r["question"])}</p>\n'
            f'    <div class="tags">{tags}</div>\n'
            f'  </a>')
    body = "\n".join(rows) or '  <p class="none">No experiments have been run yet.</p>'
    out = POSTS / "index.html"
    out.write_text(SHELL.format(rows=body))
    return out


if __name__ == "__main__":
    print("wrote", build())
