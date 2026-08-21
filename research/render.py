"""Turn a stored result plus a prose template into a publishable post.

The template is prose with `{key}` placeholders that resolve against the
experiment's measured findings (see lab.resolve). A placeholder with no
matching finding raises rather than rendering blank -- the whole point is
that every number in a post was measured by the code that produced it.
"""
from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research.lab import (MissingFinding, POSTS, Result,          # noqa: E402
                          resolve)

SHELL = """<title>{title}</title>
<style>
  :root {{
    --bg-0:#0a0b0e; --bg-1:#121319; --bg-2:#1b1d25; --border:#262932;
    --text-1:#e8eaef; --text-2:#98a0b0; --text-3:#565b69;
    --accent:#ffb454; --ok:#0ca30c; --fail:#d03b3b;
    --font-ui:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif;
    --font-mono:'JetBrains Mono',ui-monospace,'SFMono-Regular',Menlo,monospace;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg-0); color:var(--text-2);
         font:16px/1.65 var(--font-ui); -webkit-font-smoothing:antialiased; }}
  .wrap {{ max-width:760px; margin:0 auto; padding:56px 24px 96px; }}
  .eyebrow {{ font-family:var(--font-mono); font-size:12px; letter-spacing:.1em;
              text-transform:uppercase; color:var(--text-3); }}
  h1 {{ font-size:34px; line-height:1.2; letter-spacing:-.02em; color:var(--text-1);
        margin:14px 0 10px; font-weight:700; }}
  .question {{ font-size:18px; line-height:1.55; color:var(--text-2);
               border-left:2px solid var(--accent); padding-left:16px; margin:22px 0 32px; }}
  h2 {{ font-size:21px; color:var(--text-1); margin:44px 0 12px; font-weight:600;
        letter-spacing:-.01em; }}
  p {{ margin:0 0 18px; }}
  strong {{ color:var(--text-1); font-weight:600; }}
  code {{ font-family:var(--font-mono); font-size:.92em; color:var(--accent);
          background:var(--bg-2); padding:1px 5px; border-radius:3px; }}
  pre {{ background:var(--bg-1); border:1px solid var(--border); border-radius:5px;
         padding:16px 18px; overflow-x:auto; font-family:var(--font-mono);
         font-size:13px; line-height:1.6; color:var(--text-1); margin:0 0 22px; }}
  .callout {{ background:var(--bg-1); border:1px solid var(--border);
              border-left:2px solid var(--fail); border-radius:4px;
              padding:16px 18px; margin:0 0 22px; font-size:15px; }}
  .callout .lbl {{ font-family:var(--font-mono); font-size:11px; letter-spacing:.08em;
                   text-transform:uppercase; color:var(--fail); display:block;
                   margin-bottom:6px; }}
  .meta {{ margin-top:56px; padding-top:20px; border-top:1px solid var(--border);
           font-family:var(--font-mono); font-size:12px; color:var(--text-3); }}
  .meta li {{ margin:5px 0; }}
  ul {{ padding-left:22px; }} li {{ margin:7px 0; }}
</style>
<div class="wrap">
  <div class="eyebrow">{eyebrow}</div>
  <h1>{title}</h1>
  <div class="question">{question}</div>
{body}
  <div class="meta">
    <ul>
      <li>experiment {eid} &middot; run {ran_at} &middot; revision {rev}</li>
{notes}
    </ul>
  </div>
</div>
"""


# ---------------------------------------------------------------- charts --
# Inline SVG, no library: a published post has no network egress beyond its
# own origin, and a chart that needs a CDN is a chart that does not render.
# Values come from findings exactly as prose does, so a bar cannot show a
# number the experiment did not measure.

BAR_W, ROW_H, LABEL_W = 300, 26, 132
SERIES = ("var(--accent)", "var(--text-2)")


def _bars(rows, series, vmax, caption):
    """Grouped horizontal bars. `rows` is [(label, [v, ...]), ...]."""
    h = len(rows) * ROW_H + (26 if caption else 6)
    out = [f'<svg viewBox="0 0 {LABEL_W + BAR_W + 52} {h}" width="100%" '
           f'style="max-width:520px;display:block;margin:0 0 22px">']
    if caption:
        out.append(f'<text x="0" y="12" fill="#565b69" font-size="11" '
                   f'font-family="var(--font-mono)">{html.escape(caption)}</text>')
    top = 26 if caption else 6
    n = max(len(v) for _, v in rows)
    for i, (label, vals) in enumerate(rows):
        y = top + i * ROW_H
        out.append(f'<text x="0" y="{y + 12}" fill="#98a0b0" font-size="12">'
                   f'{html.escape(label)}</text>')
        bh = max(4, int(14 / n))
        for j, v in enumerate(vals):
            w = max(1, int(abs(v) / vmax * BAR_W))
            by = y + 2 + j * (bh + 2)
            out.append(f'<rect x="{LABEL_W}" y="{by}" width="{w}" height="{bh}" '
                       f'rx="1" fill="{SERIES[j % len(SERIES)]}"/>')
            out.append(f'<text x="{LABEL_W + w + 6}" y="{by + bh - 1}" fill="#565b69" '
                       f'font-size="10" font-family="var(--font-mono)">{v:g}</text>')
    if n > 1 and series:
        out.append(f'<text x="{LABEL_W}" y="{h - 1}" font-size="10" '
                   f'font-family="var(--font-mono)">')
        for j, name in enumerate(series):
            out.append(f'<tspan fill="{SERIES[j % len(SERIES)]}">{html.escape(name)}'
                       f'</tspan><tspan fill="#262932">  </tspan>')
        out.append('</text>')
    out.append("</svg>")
    return "".join(out)


def _curve(series, caption, xlabel):
    """Small multiple line chart. `series` is [(name, [[x, y], ...]), ...]."""
    W, H, PAD = 440, 150, 26
    ys = [y for _, pts in series for _, y in pts]
    lo, hi = min(ys), max(ys)
    span = (hi - lo) or 1
    out = [f'<svg viewBox="0 0 {W} {H + 30}" width="100%" '
           f'style="max-width:520px;display:block;margin:0 0 22px">']
    if caption:
        out.append(f'<text x="0" y="11" fill="#565b69" font-size="11" '
                   f'font-family="var(--font-mono)">{html.escape(caption)}</text>')
    out.append(f'<line x1="{PAD}" y1="{H}" x2="{W - 6}" y2="{H}" stroke="#262932"/>')
    for j, (name, pts) in enumerate(series):
        col = SERIES[j % len(SERIES)]
        d = []
        best = max(pts, key=lambda t: t[1])
        for x, y in pts:
            px = PAD + x * (W - PAD - 10)
            py = H - (y - lo) / span * (H - 24)
            d.append(f"{'M' if not d else 'L'}{px:.1f},{py:.1f}")
        out.append(f'<path d="{"".join(d)}" fill="none" stroke="{col}" stroke-width="2"/>')
        bx = PAD + best[0] * (W - PAD - 10)
        by = H - (best[1] - lo) / span * (H - 24)
        out.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="3.5" fill="{col}"/>')
        out.append(f'<text x="{bx:.1f}" y="{by - 8:.1f}" fill="{col}" font-size="10" '
                   f'text-anchor="middle" font-family="var(--font-mono)">'
                   f'{name} {best[1]:g}</text>')
    out.append(f'<text x="{PAD}" y="{H + 14}" fill="#565b69" font-size="10" '
               f'font-family="var(--font-mono)">{html.escape(xlabel)}</text>')
    out.append("</svg>")
    return "".join(out)


def _chart_block(spec: str, findings: dict) -> str:
    """`::bars` / `::curve` blocks. Rows are `label|value|value`, and any
    value may be a {finding} -- resolved before it ever reaches a bar."""
    head, *lines = spec.strip().splitlines()
    kind = head[2:].split()[0]
    # Attribute text is prose too -- a caption naming a sample size must
    # resolve from findings like everything else, or it ships a literal brace.
    attrs = {k: resolve(v, findings)
             for k, v in re.findall(r'(\w+)="([^"]*)"', head)}
    if kind == "curve":
        names = [n.strip() for n in attrs.get("series", "").split(",") if n.strip()]
        keys = [k.strip() for k in attrs["keys"].split(",")]
        for k in keys:
            if k not in findings:
                raise MissingFinding(f"chart references {{{k}}}, measured: {sorted(findings)}")
        return _curve(list(zip(names or keys, (findings[k] for k in keys))),
                      attrs.get("caption", ""), attrs.get("xlabel", ""))
    rows = []
    for ln in lines:
        if not ln.strip() or ln.strip() == "::":
            continue
        label, *vals = [c.strip() for c in ln.split("|")]
        # The LABEL is prose too. Only the values were resolved at first, so
        # a row named "weeks {weeks_label}" drew the literal brace into the
        # chart -- the one place a leaked placeholder is easy to miss,
        # because it renders inside an SVG rather than in the body text.
        rows.append((resolve(label, findings),
                     [float(resolve(v, findings)) for v in vals]))
    vmax = float(attrs["max"]) if "max" in attrs else max(
        v for _, vs in rows for v in vs) * 1.08
    series = [s.strip() for s in attrs.get("series", "").split(",") if s.strip()]
    return _bars(rows, series, vmax, attrs.get("caption", ""))


def _blocks(text: str, findings: dict) -> str:
    """A deliberately small markup: blank-line paragraphs, ## headings,
    ``` code fences, and > callouts. Anything richer belongs in the template
    as literal HTML rather than in a parser nobody asked for."""
    out, i = [], 0
    parts = re.split(r"\n\s*\n", text.strip())
    while i < len(parts):
        p = parts[i].strip()
        if p.startswith("::"):
            out.append(f"  {_chart_block(p, findings)}")
        elif p.startswith("```"):
            code = p.strip("`").lstrip("\n")
            out.append(f"  <pre>{html.escape(resolve(code, findings))}</pre>")
        elif p.startswith("## "):
            out.append(f"  <h2>{resolve(p[3:].strip(), findings)}</h2>")
        elif p.startswith("> "):
            # Every line of a callout carries the marker, so strip it from
            # all of them -- partitioning the raw block left "> " sitting at
            # the front of the second line and rendered it as literal text.
            lines = [ln[2:] if ln.startswith("> ") else ln
                     for ln in p.splitlines()]
            label, rest = lines[0], "\n".join(lines[1:])
            out.append(f'  <div class="callout"><span class="lbl">{resolve(label, findings)}</span>'
                       f'{resolve(rest.strip(), findings)}</div>')
        elif p.startswith("- "):
            items = "".join(f"<li>{resolve(ln[2:].strip(), findings)}</li>" for ln in p.splitlines())
            out.append(f"  <ul>{items}</ul>")
        else:
            out.append(f"  <p>{resolve(p, findings)}</p>")
        i += 1
    return "\n".join(out)


def render(experiment_id: str) -> Path:
    stored = Result.load(experiment_id)
    findings = stored["findings"]
    tmpl = (POSTS / f"{experiment_id}.md").read_text()
    body = _blocks(tmpl, findings)
    # Notes are prose too: one referencing {weeks_label} shipped the literal
    # brace into the metadata list, because only the body was ever resolved.
    notes = "\n".join(f"      <li>{html.escape(resolve(n, findings))}</li>"
                      for n in stored.get("notes", []))
    page = SHELL.format(
        title=html.escape(stored["title"]),
        eyebrow=f"Research &middot; {experiment_id.upper()}",
        question=html.escape(stored["question"]),
        body=body, eid=experiment_id.upper(),
        ran_at=stored["ran_at"][:10], rev=stored.get("git_rev") or "unknown",
        notes=notes)
    out = POSTS / f"{experiment_id}.html"
    out.write_text(page)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ids", nargs="+")
    args = ap.parse_args()
    for eid in args.ids:
        print("wrote", render(eid))
