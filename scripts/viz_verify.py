#!/usr/bin/env python3
"""Render verifier runs as a single self-contained HTML page.

Reads a results JSONL written by ``verify_trajectory.py --save_trajectory`` and lays each
run out as a timeline: what the verifier searched for, what the tool returned, and how it
scored the answer. Long tool output is collapsed by default and the searched term is
highlighted inside it, which is the thing you actually want to see -- whether the value
the agent claimed really appears in the raw observation.

Usage::

    python scripts/viz_verify.py output/verify_bcp/results.jsonl -o /tmp/verify.html
    python scripts/viz_verify.py output/verify_bcp/results.jsonl --only FN -o /tmp/fn.html
"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

# support_score is continuous; colour by band (see FinishTool.SUPPORT_BANDS)
SCORE_BANDS = [
    (1.0, "#1a7f37", "#dafbe1"),   # fully established
    (0.7, "#2da44e", "#e6ffec"),   # strongly supported
    (0.4, "#9a6700", "#fff8c5"),   # partially supported
    (0.1, "#bc4c00", "#fff1e5"),   # weak
    (0.0, "#cf222e", "#ffebe9"),   # contradicted / none
]
# an answer counts as holding up at or above this score; mirrors verify_trajectory.py
PASS_THRESHOLD = 1.0


def band(score: float | None) -> tuple[str, str]:
    if score is None:
        return "#656d76", "#eaeef2"
    for lo, fg, bg in SCORE_BANDS:
        if score >= lo:
            return fg, bg
    return "#cf222e", "#ffebe9"

CSS = """
*{box-sizing:border-box}
body{margin:0;padding:24px;background:#f6f8fa;color:#1f2328;
     font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto}
h1{font-size:19px;margin:0 0 4px}
.sub{color:#656d76;font-size:13px;margin-bottom:20px}
.run{background:#fff;border:1px solid #d1d9e0;border-radius:8px;margin-bottom:20px;overflow:hidden}
.run>summary{cursor:pointer;padding:14px 16px;list-style:none;display:flex;gap:10px;
             align-items:center;flex-wrap:wrap;background:#fff}
.run>summary::-webkit-details-marker{display:none}
.run[open]>summary{border-bottom:1px solid #d1d9e0}
.pill{font:600 11px/1 ui-monospace,SFMono-Regular,monospace;padding:5px 8px;border-radius:999px;
      white-space:nowrap}
.q{flex:1;min-width:240px;color:#1f2328;font-size:13px}
.body{padding:16px}
.kv{display:grid;grid-template-columns:132px 1fr;gap:6px 12px;margin-bottom:16px;font-size:13px}
.kv dt{color:#656d76}
.kv dd{margin:0;word-break:break-word}
.mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12px}
.sec{font:600 11px/1 ui-monospace,monospace;color:#656d76;letter-spacing:.07em;
     text-transform:uppercase;margin:18px 0 8px}
.step{border-left:2px solid #d1d9e0;padding:0 0 14px 14px;position:relative}
.step:last-child{padding-bottom:0}
.step::before{content:"";position:absolute;left:-5px;top:4px;width:8px;height:8px;
              border-radius:50%;background:#8c959f}
.step.call::before{background:#0969da}
.step.fin::before{background:#1a7f37}
.tool{font:600 12px/1 ui-monospace,monospace;color:#0969da}
.args{background:#f6f8fa;border:1px solid #d1d9e0;border-radius:6px;padding:7px 9px;
      margin-top:5px;font-family:ui-monospace,monospace;font-size:12px;overflow-x:auto}
.think{color:#656d76;font-size:12.5px;margin-top:6px;white-space:pre-wrap;
       border-left:2px solid #eaeef2;padding-left:9px}
details.obs{margin-top:6px}
details.obs>summary{cursor:pointer;color:#656d76;font-size:12px;
                    font-family:ui-monospace,monospace}
pre.obs{background:#f6f8fa;border:1px solid #d1d9e0;border-radius:6px;padding:9px;
        margin:6px 0 0;white-space:pre-wrap;word-break:break-word;font-size:12px;
        max-height:420px;overflow:auto}
mark{background:#fff8c5;padding:0 2px;border-radius:2px}
mark.hit{background:#dafbe1;font-weight:600}
.box{border:1px solid #d1d9e0;border-radius:6px;padding:11px;margin-top:8px;font-size:13px;
     white-space:pre-wrap;word-break:break-word}
.box.ev{background:#f6f8fa}
.agree-no{color:#cf222e;font-weight:600}
.agree-yes{color:#1a7f37;font-weight:600}
"""


def esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def highlight(text: str, terms: list[str]) -> str:
    """Escape text, then mark each searched term where it appears."""
    out = esc(text)
    for t in sorted({t for t in terms if t and len(t) >= 2}, key=len, reverse=True):
        out = re.sub(f"({re.escape(esc(t))})", r'<mark class="hit">\1</mark>', out,
                     flags=re.IGNORECASE)
    return out


def quadrant(rec: dict) -> str | None:
    s, jc = rec.get("support_score"), rec.get("judged_correct")
    if s is None or jc is None:
        return None
    passed = s >= PASS_THRESHOLD
    return "TP" if (passed and jc) else "FP" if passed else "FN" if jc else "TN"


def render_run(rec: dict, idx: int) -> str:
    score = rec.get("support_score")
    fg, bg = band(score)
    label = rec.get("label") or ("no score" if score is None else "")
    quad = quadrant(rec)
    jc = rec.get("judged_correct")
    agree = None if quad is None else quad in ("TP", "TN")

    parts = [f'<details class="run"{" open" if idx == 0 else ""}><summary>']
    parts.append(f'<span class="pill" style="color:{fg};background:{bg}">'
                 f'{"-" if score is None else f"{score:.2f}"} {esc(label)}</span>')
    parts.append(f'<span class="pill" style="background:#eaeef2;color:#656d76">'
                 f'judge: {"correct" if jc else "wrong"}</span>')
    if quad:
        cls = "agree-yes" if agree else "agree-no"
        parts.append(f'<span class="pill {cls}" style="background:#f6f8fa">{quad}</span>')
    parts.append(f'<span class="q">{esc(rec.get("question", "")[:150])}</span></summary>')
    parts.append('<div class="body">')

    parts.append('<dl class="kv">')
    for k, v in (("question", rec.get("question")),
                 ("gold", rec.get("gold")),
                 ("agent answer", rec.get("candidate_answer")),
                 ("verifier re-derived", rec.get("verified_answer") or "(empty)"),
                 ("support_score", "-" if score is None else f"{score:.2f}  ({label})"),
                 ("source file", Path(str(rec.get("file", ""))).name)):
        parts.append(f"<dt>{esc(k)}</dt><dd{' class=mono' if k=='source file' else ''}>"
                     f"{esc(v)}</dd>")
    parts.append("</dl>")

    if rec.get("evidence"):
        parts.append('<div class="sec">evidence cited</div>'
                     f'<div class="box ev">{esc(rec["evidence"])}</div>')
    if rec.get("reason"):
        parts.append('<div class="sec">reason</div>'
                     f'<div class="box">{esc(rec["reason"])}</div>')

    msgs = rec.get("verifier_messages") or []
    if msgs:
        st = rec.get("verifier_stats") or {}
        parts.append(f'<div class="sec">verifier trajectory &mdash; '
                     f'{st.get("iterations", "?")} iterations, '
                     f'{len(msgs)} messages, tools {esc(st.get("tool_calls"))}</div>')
        pending: list[str] = []          # terms searched by the preceding call
        for m in msgs:
            role = m.get("role")
            if role == "system":
                continue
            if role == "user":
                continue
            if role == "tool":
                content = str(m.get("content") or "")
                name = esc(m.get("name"))
                head = content[:150].replace("\n", " ")
                parts.append('<div class="step">'
                             f'<details class="obs"><summary>&#9656; {name} returned '
                             f'{len(content)} chars &mdash; {esc(head)}&hellip;</summary>'
                             f'<pre class="obs">{highlight(content, pending)}</pre>'
                             "</details></div>")
                pending = []
                continue
            # assistant
            calls = m.get("tool_calls") or []
            is_fin = any(c.get("function", {}).get("name") == "finish" for c in calls)
            parts.append(f'<div class="step {"fin" if is_fin else "call" if calls else ""}">')
            reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
            if reasoning:
                parts.append(f'<div class="think">{esc(reasoning)}</div>')
            for c in calls:
                fn = c.get("function", {})
                raw = fn.get("arguments") or "{}"
                parts.append(f'<div class="tool">{esc(fn.get("name"))}</div>')
                try:
                    a = json.loads(raw)
                    pending = [str(a[k]) for k in ("query",) if a.get(k)]
                    parts.append(f'<div class="args">{esc(json.dumps(a, ensure_ascii=False))}</div>')
                except json.JSONDecodeError:
                    parts.append(f'<div class="args">{esc(raw)}</div>')
            if m.get("content") and not calls:
                parts.append(f'<div class="think">{esc(m["content"])}</div>')
            parts.append("</div>")
    else:
        parts.append('<div class="sec">verifier trajectory</div>'
                     '<div class="box">Not recorded. Re-run verify_trajectory.py with '
                     '--save_trajectory to capture it.</div>')

    parts.append("</div></details>")
    return "".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path, help="results JSONL from verify_trajectory.py")
    ap.add_argument("-o", "--out", type=Path, default=Path("verify.html"))
    ap.add_argument("--only", choices=["TP", "FP", "TN", "FN"], default=None,
                    help="Keep only runs in this outcome quadrant")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.results) if l.strip()]
    if args.only:
        recs = [r for r in recs if quadrant(r) == args.only]
    if args.limit:
        recs = recs[: args.limit]

    counts: dict[str, int] = {}
    for r in recs:
        q = quadrant(r) or "n/a"
        counts[q] = counts.get(q, 0) + 1
    summary = "  ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    with_traj = sum(1 for r in recs if r.get("verifier_messages"))

    body = "".join(render_run(r, i) for i, r in enumerate(recs))
    doc = (f"<!doctype html><meta charset=utf-8><title>verifier runs</title>"
           f"<style>{CSS}</style><div class=wrap>"
           f"<h1>Verifier runs &mdash; {len(recs)} shown</h1>"
           f"<div class=sub>{esc(summary)} &nbsp;&middot;&nbsp; "
           f"{with_traj} with recorded trajectory &nbsp;&middot;&nbsp; "
           f"source: {esc(args.results)}</div>{body}</div>")
    args.out.write_text(doc)
    print(f"wrote {args.out} ({len(doc)/1024:.0f} KB, {len(recs)} runs, {with_traj} with trajectory)")


if __name__ == "__main__":
    main()
