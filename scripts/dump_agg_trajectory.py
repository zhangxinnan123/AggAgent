#!/usr/bin/env python3
"""Render one aggregator trajectory from an ``<strategy>_logs_k{k}.jsonl`` into readable Markdown.

The aggregation harness logs the AggAgent tool-calling loop under the
``aggagent_messages`` key of each JSONL entry (one entry per sampled combo).
This prints a single entry as Markdown so the trajectory can be read directly.

Usage::

    python scripts/dump_agg_trajectory.py output/agg_test/aggagent_logs_k4.jsonl --index 0
    python scripts/dump_agg_trajectory.py output/agg_test/aggagent_logs_k4.jsonl --list
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FENCE = "```"


def load(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def render(entry: dict, model: str | None) -> str:
    out: list[str] = ["# AggAgent trajectory", ""]
    stats = entry.get("aggagent_stats") or {}
    if model:
        out.append(f"- aggregator model: {model}")
    out += [
        f"- combo: {entry.get('combo')}  (base-traj correctness: {entry.get('combo_correct')})",
        f"- messages: {len(entry.get('aggagent_messages') or [])}   iterations: {stats.get('iterations')}",
        f"- tool_calls: {stats.get('tool_calls')}",
        f"- is_correct: {entry.get('is_correct')}   aggregation_cost: {entry.get('aggregation_cost')}",
        "",
        "## Question",
        "",
        str(entry.get("question", "")),
        "",
        "## Gold answer",
        "",
        str((entry.get("instance") or {}).get("answer")),
        "",
        "## Aggregated prediction",
        "",
        str(entry.get("prediction") or ""),
        "",
        "## Full message trace",
    ]

    def block(label: str, body: str, lang: str = "") -> None:
        out.extend(["", f"**{label}:**", "", FENCE + lang, body, FENCE])

    for i, m in enumerate(entry.get("aggagent_messages") or []):
        out += ["", f"### [{i}] {str(m.get('role')).upper()}"]
        reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
        if reasoning:
            block("reasoning", reasoning)
        for call in m.get("tool_calls") or []:
            fn = call.get("function", {})
            block(f"tool_call `{fn.get('name')}`", fn.get("arguments", ""), "json")
        content = m.get("content")
        if content:
            block("content", content if isinstance(content, str) else json.dumps(content, indent=2))
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", type=Path, help="Path to <strategy>_logs_k{k}.jsonl")
    ap.add_argument("--index", type=int, default=0, help="Which entry to render (default: 0)")
    ap.add_argument("--model", default=None, help="Aggregator model name, for the header")
    ap.add_argument("--out", type=Path, default=None, help="Write to this file instead of stdout")
    ap.add_argument("--list", action="store_true", help="List entries and exit")
    args = ap.parse_args()

    rows = load(args.logs)
    if args.list:
        for i, r in enumerate(rows):
            n = len(r.get("aggagent_messages") or [])
            print(f"[{i}] combo={r.get('combo')} msgs={n} is_correct={r.get('is_correct')} "
                  f"q={str(r.get('question'))[:70]!r}")
        return

    text = render(rows[args.index], args.model)
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out} ({len(text)} chars)")
    else:
        print(text)


if __name__ == "__main__":
    main()
