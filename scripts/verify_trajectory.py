#!/usr/bin/env python3
"""Run AggAgent in verify mode over individual rollout trajectories.

Unlike aggregation, this treats each trajectory on its own (K=1 by default) and asks
whether its final answer is actually supported by the tool observations it recorded.
The agent cannot search the web, so the 0-3 score measures *evidence support*, not
ground truth -- see ``AggAgent.verify``.

Because each rollout JSON already carries ``auto_judge``, this doubles as a way to
measure the verifier: compare its score against the judged correctness it never saw.

Usage::

    python scripts/verify_trajectory.py output/rollout/<M>/<ds> \
        --model Qwen3.5-35B-A3B --api_base http://node:6000/v1 --task deepsearchqa \
        --limit 4 --out output/verify/results.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aggagent import AggAgent  # noqa: E402

# support_score is continuous in [0, 1] (see FinishTool.SUPPORT_BANDS), so the pass/fail
# cut-off is a reporting choice rather than a property of the verifier. This default says
# "only a fully established answer passes"; --threshold overrides it, and the report sweeps
# a range so the trade-off is visible without re-running anything.
DEFAULT_THRESHOLD = 1.0


def holds_up(score: float | None, threshold: float = DEFAULT_THRESHOLD) -> bool | None:
    """Fold a continuous support_score into a binary judgement. None if absent."""
    if score is None:
        return None
    return float(score) >= threshold


def judged_correct(data: dict, task: str) -> bool | None:
    """Correctness per the rollout's own inline judge, mirroring Strategy.is_correct."""
    aj = data.get("auto_judge") or {}
    if task == "deepsearchqa":
        val = aj.get("all_correct")
        return bool(val) if val is not None else None
    val = aj.get("correctness")
    if val is None:
        return None
    if isinstance(val, str):
        return val.strip().lower() in ("correct", "true", "yes")
    return bool(val)


def collect(root: Path, limit: int | None,
            total_splits: int = 1, worker_split: int = 1) -> list[tuple[Path, dict]]:
    """Gather individual trajectory JSONs from the iter*/ layout.

    With total_splits > 1, keeps only this shard's slice so several processes can run
    against separate model servers. The split is round-robin over the sorted file list,
    which keeps each shard's mix of questions and iterations comparable.
    """
    paths = sorted(root.rglob("*.json"))
    if total_splits > 1:
        paths = [p for i, p in enumerate(paths) if i % total_splits == (worker_split - 1)]
    out: list[tuple[Path, dict]] = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        if "error" in data or not data.get("messages"):
            continue
        out.append((path, data))
        if limit and len(out) >= limit:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rollout_dir", nargs="?", type=Path, default=None)
    ap.add_argument("--summarize", type=Path, nargs="+", default=None,
                    help="Re-print the table for existing results JSONL file(s) and exit; "
                         "pass several to merge shards")
    ap.add_argument("--model", default=None)
    ap.add_argument("--api_base", default=None)
    ap.add_argument("--task", default="deepsearchqa")
    ap.add_argument("--limit", type=int, default=None, help="Max trajectories to verify")
    ap.add_argument("--max_workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None, help="Write JSONL results here")
    ap.add_argument("--total_splits", type=int, default=1,
                    help="Shard the work across this many processes (each needs its own --api_base)")
    ap.add_argument("--worker_split", type=int, default=1,
                    help="Which shard this process handles, 1-based")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"support_score at or above which an answer counts as holding up "
                         f"(default {DEFAULT_THRESHOLD}); the report sweeps a range regardless")
    ap.add_argument("--save_trajectory", action="store_true",
                    help="Also record the verifier's own messages/stats (for scripts/viz_verify.py)")
    ap.add_argument("--no_resume", action="store_true",
                    help="Re-verify everything instead of skipping entries already in --out")
    args = ap.parse_args()

    if args.summarize:
        summarize(args.summarize, args.threshold)
        return
    if args.rollout_dir is None or args.model is None:
        ap.error("rollout_dir and --model are required unless --summarize is given")

    if not 1 <= args.worker_split <= args.total_splits:
        ap.error(f"--worker_split must be between 1 and --total_splits ({args.total_splits})")
    items = collect(args.rollout_dir, args.limit, args.total_splits, args.worker_split)
    if args.total_splits > 1:
        print(f"shard {args.worker_split}/{args.total_splits}")

    def one(item: tuple[Path, dict]) -> dict:
        path, data = item
        agent = AggAgent(model=args.model, api_base=args.api_base,
                         task=args.task, mode="verify")
        # Call _run directly (as aggregation/_strategy/aggagent.py does) so the
        # verifier's OWN trajectory is kept, not just its verdict -- verify() returns
        # only the result dict and would discard the messages we want to inspect.
        out = agent._run(data.get("question", ""), [{"messages": data["messages"]}])
        verified = out.get("result") or {"support_score": None, "error": "no valid support_score"}
        rec = {
            "file": str(path),
            "question": data.get("question", ""),
            "candidate_answer": (data.get("auto_judge") or {}).get("extracted_final_answer"),
            "gold": (data.get("instance") or {}).get("answer"),
            "judged_correct": judged_correct(data, args.task),
            **verified,
        }
        if args.save_trajectory:
            rec["verifier_messages"] = out.get("messages")
            rec["verifier_stats"] = out.get("stats")
        return rec

    # Resume: each result is appended as soon as it lands, so a run cut short by the
    # serving job timing out can be continued by re-invoking with the same --out.
    done: dict[str, dict] = {}
    if args.out and args.out.exists() and not args.no_resume:
        with open(args.out) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[r["file"]] = r
        if done:
            print(f"resume: {len(done)} already in {args.out}")

    todo = [it for it in items if str(it[0]) not in done]
    print(f"verifying {len(todo)} trajectories with {args.model} "
          f"({len(items) - len(todo)} skipped)")

    results = list(done.values())
    if todo:
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
        lock = Lock()
        sink = open(args.out, "a") if args.out else None
        try:
            with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
                for i, r in enumerate(ex.map(one, todo), 1):
                    results.append(r)
                    with lock:
                        if sink:
                            sink.write(json.dumps(r) + "\n")
                            sink.flush()
                    print(f"  [{i}/{len(todo)}] support_score={r.get('support_score')} "
                          f"judged={r.get('judged_correct')} {Path(r['file']).name}",
                          flush=True)
        finally:
            if sink:
                sink.close()
        if args.out:
            print(f"wrote {args.out}")

    report(results, args.threshold)


def confusion(results: list[dict], threshold: float) -> dict:
    """TP/FP/TN/FN of the verifier against the inline judge at a given threshold."""
    c = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "n": 0}
    for r in results:
        hu, jc = holds_up(r.get("support_score"), threshold), r.get("judged_correct")
        if hu is None or jc is None:
            continue
        c["n"] += 1
        if hu and jc:
            c["tp"] += 1
        elif hu and not jc:
            c["fp"] += 1       # passed an answer the judge marked wrong
        elif not hu and jc:
            c["fn"] += 1       # rejected an answer the judge marked right
        else:
            c["tn"] += 1
    return c


def report(results: list[dict], threshold: float = DEFAULT_THRESHOLD) -> None:
    """Print the per-run table, then sweep the threshold so the trade-off is visible."""
    print(f"\n{'score':>6}  {'label':<22}{'pass':>6}  {'judged':>7}  agree  file")
    for r in results:
        s = r.get("support_score")
        hu, jc = holds_up(s, threshold), r.get("judged_correct")
        ok = None if (hu is None or jc is None) else hu == bool(jc)
        print(f"{'-' if s is None else f'{s:.2f}':>6}  {str(r.get('label')):<22}"
              f"{str(hu):>6}  {str(jc):>7}  "
              f"{'-' if ok is None else ('yes' if ok else 'NO'):>5}  {Path(r['file']).name}")

    scored = [r for r in results if r.get("support_score") is not None]
    if not scored:
        return
    vals = sorted(r["support_score"] for r in scored)
    print(f"\nsupport_score over {len(scored)} runs: "
          f"min {vals[0]:.2f}  median {vals[len(vals)//2]:.2f}  max {vals[-1]:.2f}  "
          f"distinct {len(set(vals))}")
    corr = [r["support_score"] for r in scored if r.get("judged_correct")]
    wrong = [r["support_score"] for r in scored if r.get("judged_correct") is False]
    if corr and wrong:
        print(f"  mean score: judged-correct {sum(corr)/len(corr):.2f}  "
              f"vs judged-wrong {sum(wrong)/len(wrong):.2f}")

    print(f"\n{'threshold':>10}{'acc':>8}{'prec':>8}{'rec':>8}{'TP':>5}{'FP':>5}{'TN':>5}{'FN':>5}")
    for t in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4):
        c = confusion(scored, t)
        if not c["n"]:
            continue
        tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
        acc = (tp + tn) / c["n"]
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        mark = " <- default" if abs(t - threshold) < 1e-9 else ""
        print(f"{t:>10.2f}{acc:>8.1%}{prec:>8.1%}{rec:>8.1%}"
              f"{tp:>5}{fp:>5}{tn:>5}{fn:>5}{mark}")


def summarize(paths: list[Path], threshold: float = DEFAULT_THRESHOLD) -> None:
    """Re-print the table for existing results file(s), without re-running the agent.

    Several paths are merged and de-duplicated by source file, so shards written by
    separate --worker_split processes can be reported together.
    """
    seen: dict[str, dict] = {}
    for p in paths:
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                seen[r["file"]] = r
    if len(paths) > 1:
        print(f"merged {len(paths)} files -> {len(seen)} unique trajectories")
    report(list(seen.values()), threshold)


if __name__ == "__main__":
    main()
