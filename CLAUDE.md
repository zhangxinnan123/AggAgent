# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AggAgent is a research framework for **parallel test-time scaling of long-horizon agentic tasks**: sample K independent agent trajectories, then aggregate them into a single better answer. The repo has three cooperating stages plus a shippable package:

- `rollout/` — generates K independent ReAct trajectories per benchmark question (needs the `rollout` extra).
- `aggregation/` — evaluates aggregation strategies (heuristic + LLM-based) over rollout outputs and scores them.
- `aggagent/` — the published PyPI package (`pip install aggagent`); a self-contained tool-calling agent that synthesizes one answer from N trajectories. Imported by the `aggagent` aggregation strategy.
- `evaluation/` — per-task LLM judges (`compute_score`) shared by rollout and aggregation.

The paper's `aggagent` strategy is just the `aggagent` package wrapped in the aggregation harness — the core aggregation logic lives in `aggagent/agent.py`, not in `aggregation/`.

Only `aggagent/` is packaged (`[tool.hatch.build.targets.wheel] packages = ["aggagent"]`); `rollout/`, `aggregation/`, `evaluation/` are repo-root scripts/modules imported via sys.path, so **run everything from the repo root**.

## Setup & commands

Uses [uv](https://docs.astral.sh/uv/); requires Python ≥ 3.12.

```bash
uv sync --extra rollout          # full dev env (rollout deps: vllm, crawl4ai, faiss, tevatron, ...)
pip install -e ".[rollout]"      # pip alternative
cp .env.example .env             # then fill OPENAI_API_KEY / GEMINI_API_KEY / SERPER_KEY_ID
uv pip install --reinstall-package litellm 'litellm==1.83.0'   # REQUIRED after every `uv sync --extra rollout`
```

**The litellm collision (run the third command or nothing works).** `crawl4ai` depends on `unclecode-litellm`, a fork that unpacks into the same `site-packages/litellm/` directory as the real `litellm`. Whichever installs second partially overwrites the other, and `import litellm` then dies with e.g. `ImportError: cannot import name 'REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD' from 'litellm.constants'`. Since litellm is the backbone of `aggagent/agent.py` and every judge in `evaluation/`, a fresh `--extra rollout` env is broken on arrival. Reinstalling `litellm` last repairs the directory; `crawl4ai` still imports fine against upstream 1.83.0. `uv sync` re-breaks it, so this is not a one-time fix.

There is **no test suite, linter config, or CI** in this repo — validation is done by running the pipeline scripts. Run Python via `uv run python ...` so the venv and `.env` are picked up. Quickest post-setup check: `python -c "import litellm, crawl4ai, vllm; from aggagent import AggAgent; from evaluation import get_evaluator"` from the repo root.

`uv sync` needs network access (`transformers` and `tevatron` are git dependencies). The SFM-P5 cluster has none, so sync the env where there is internet; **poe-skywalker does reach pypi/github**, so `uv sync` works directly there.

**Cluster note:** per the user's global HyperPod workflow, GPU/vLLM jobs and training run on a cluster over SSH, not locally. Local machine is for editing + light aggregation against a remote `--api_base`. Current env lives at `poe-skywalker:/fsx/xinnanzh/AggAgent` (`.venv`, Python 3.13, torch 2.10+cu128, vllm 0.19.0; `uv` at `~/.local/bin/uv`) — an A100 `ml-p4d-24xlarge` Slurm cluster, which the user refers to as "p4".

Other env vars beyond `.env.example`: `EVAL_API_BASE` (judge endpoint for Qwen judges, default `http://localhost:7000/v1`), `SEARCH_SERVER_URL` (rollout search server), and the `SEARCHER_TYPE`/`INDEX_PATH`/... set for `browsecomp-plus` in `scripts/rollout.sh`.

### End-to-end flow

```bash
# 1. Rollout (needs a vLLM server + search server running first — see rollout/README.md)
bash scripts/rollout.sh
# default layout: output/rollout/<MODEL>/<DATASET>/iter{k}/<question>.json
# note scripts/rollout.sh passes --output_dir output/<MODEL_BASENAME>/<DATASET>, overriding the default

# 2. Aggregate + score
uv run python aggregation/aggregate.py --strategy aggagent --model GLM-4.7-Flash \
  --api_base http://localhost:6000/v1 --task browsecomp --k 4 \
  output/rollout/GLM-4.7-Flash/browsecomp
# writes output/aggregation/<MODEL>/<DIR_TAG>/aggagent_{logs,stats}_k{k}.{jsonl,json}
# scripts/aggregation.sh is the same call with the settings hoisted to top-of-file vars
```

`--strategy` accepts any of `pass mv wmv bon fewtool solagg summagg aggagent`, plus `heuristic` (all no-LLM strategies) and `all`. `--model` is required for any LLM-based strategy. `--k` defaults to `1,2,4,...,N`. `--max_workers` (default 10) is the per-problem thread pool.

`aggregate.py` must be invoked as a script path from the repo root (`uv run python aggregation/aggregate.py`), **not** `python -m`: it does a bare `from _strategy import ...` (resolved via the script's own directory) while `_strategy/base.py` does `from evaluation import get_evaluator` and `_strategy/aggagent.py` does `from aggagent import AggAgent` (resolved via repo root / the installed package).

### Using published rollouts instead of generating them

```bash
uv run python scripts/download_dataset.py            # datasets → data/<name>.jsonl
python scripts/hf_to_rollout.py --repo <hf-repo> --model <M> --out output/rollout/<M>/<ds>  # flat parquet → the iter{k}/ layout aggregation expects
python scripts/collect_trajs.py <rollout_dir> "<question substring>"  # gather one question's trajectories into a flat dir for the Claude Code skill
```

## Key architecture details

### On-disk contract between stages
Aggregation discovers **leaf directories** (dirs with no subdirs) under the paths passed on the CLI (`find_leaf_directories` in `aggregate.py`), treating each leaf as one rollout iteration. It groups result JSONs across leaves by `question` / `instance_id` key (falling back to filename). So the `iter{1..N}/` layout is load-bearing — N = number of leaf dirs, and every question is expected to appear once per leaf; questions with fewer than N results are warned about and skipped by LLM strategies. `hf_to_rollout.py` exists specifically to reconstruct this layout from flat parquet (question id = `instance_id`/`prompt_id` when present, else a sha1 of the question).

**Multi-node rollouts break this**: with `--total_splits > 1`, `run_multi_react.py` writes `iter{k}_split{w}of{t}/`, so N inflates to `roll_out_count × total_splits` and every question is missing from other splits' leaves. Merge split dirs into plain `iter{k}/` before aggregating.

### Result JSON shape (read by strategies via `Strategy` helpers in `_strategy/base.py`)
Correctness, answers, confidence, and cost are all read from fields on the result dict — not recomputed unless a strategy synthesizes a new prediction. Rollout runs the judge inline, so `auto_judge` is already present in rollout outputs.
- `data["auto_judge"]` → `correctness`, `all_correct`, `extracted_final_answer`, `confidence` (0–100, normalized to 0–1), rubric grades. `is_correct()` branches by task (binary for browsecomp/hle, `all_correct` for deepsearchqa, `metrics.overall_score` / rubric-weighted score for healthbench/researchrubrics).
- `data["cost"]` → `{rollout, tool}` (precomputed at rollout time by `rollout/utils.compute_rollout_cost`); LLM strategies add `aggregation` cost on top (`CostBreakdown`). Per-model $/1M-token rates are duplicated in **two** tables — `MODEL_COSTS` in `rollout/utils.py` (plus `SEARCH_COST_PER_K` / `SCRAPE_COST_PER_K` for tool cost) and `MODEL_COSTS` in `_strategy/base.py` (aggregation cost). **Add a model to both** before costing new results; an unlisted model silently costs $0.
- `data["debug_data"]["tool_usage"]` → tool counts (used by `fewtool`).
- `data["messages"]` → the full trajectory, consumed by `summagg` and by the aggagent strategy.

### How @k is measured (matters for reading any number this repo prints)
Heuristic strategies (`pass/mv/wmv/bon/fewtool`) enumerate **all** C(N,k) combinations and average. LLM strategies (`solagg/summagg/aggagent`) do **not**: `calculate_at_k` randomly samples at most `max_combos = 3` combos per problem with a fixed seed (42, `Strategy.rng`) and averages over those — so LLM-strategy numbers are noisier than heuristic ones at the same k. The `± X%` printed next to a metric is the std **across problems**, not across combos or seeds.

### Resume is always on
`aggregate.py` hardcodes `resume: True` in `strategy_kwargs`. LLM strategies reload `output/.../<strategy>_logs_k{k}.jsonl` and reuse every entry whose `question` matches, so a re-run with different settings will silently return the old numbers — **delete the log file** (or pass a different `--output_dir`) to force recomputation. Rollout has the same behavior at the JSON level: it skips questions whose result file already exists and has no `"error"` key.

`--skip_score` logs the aggregated prediction with `is_correct: None` (counted as 0 in the printed metric) for scoring later; there is no posthoc-scoring script in the repo, so that scoring pass has to be written.

Adding a new strategy: subclass `Strategy`, implement `calculate_at_k`, and register it in both `STRATEGIES` and (if no LLM) `HEURISTIC_STRATEGIES` in `_strategy/__init__.py`.

### The `aggagent` package (`aggagent/agent.py`)
Single class `AggAgent`. It is a tool-calling loop (`_run`) with four tools defined in `aggagent/tools.py`: `get_solution`, `search_trajectory` (ROUGE-L ranked), `get_segment`, `finish`. Notable behaviors when editing:
- **Two entry points**: `run(question, trajectories)` is the public API and returns `{"solution", "reason"}` (or `{..., "error"}`); the aggregation strategy calls the private `_run(question, run_results)` instead, which takes full result dicts (uses `r["messages"]`) and returns `{"result", "messages", "stats"}`. Changing `_run`'s return shape or `stats` keys breaks `_strategy/aggagent.py` (cost accounting reads `stats["token_usage_each_step"]`, stats file reads `iterations`/`server_errors`/`tool_call_errors`/`tool_calls`/`context_limit_reached`).
- **Provider routing lives in `call_server`**: model-name substrings (`gemini`, `gpt`, `oss`, `minimax`, `qwen`) select the litellm model prefix, keys, reasoning-effort, and body tweaks. Local vLLM is the default (`hosted_vllm/<model>` + `api_base`). Passing `llm_kwargs` bypasses all of this — only `messages` and `tools` are then injected. Only the first tool call per response is honored (`tool_calls[:1]`, `parallel_tool_calls=False`).
- **Prompt & output format switch on task**: `LONG_FORM_TASKS = {healthbench, researchrubrics}` and a `qwen` model branch pick different system prompts in `_run`; `FinishTool(variant=...)` controls whether `finish` emits `<explanation>/<answer>` (short-answer) or a full cited report (long-form).
- **Context management**: approximate token count (`count_tokens_approx`, 4 chars/token, plus the tool schemas) is checked each iteration against `max_context_tokens` (default 100 KiB tokens); on overflow the agent is forced to a `finish`-only call. `MAX_ITERATIONS = 100`. A provider context-length error returns `"ContextLengthError"` and yields a `None` result (scored 0).
- Trajectory messages use OpenAI format; reasoning goes under `reasoning_content` (assistant), remapped to `reasoning` / `thinking_blocks` per provider in `_message_to_dict`.

### Evaluation judges (`evaluation/`)
`get_evaluator(task)` returns an `Evaluator` (subclass of `evaluation/base.py`). Each implements `build_prompt` / `parse_response` / `compute_score` (async, via `litellm.acompletion`). There is no HLE-specific judge — `hle`, `browsecomp-plus`, and any unknown task fall through to `BrowseCompEvaluator`. Provider routing for judges is separate, in `_build_litellm_body` — Qwen judges hit `EVAL_API_BASE` (default `http://localhost:7000/v1`). `_compute_score` in `_strategy/base.py` is the sync bridge the strategies call, and it defaults the judge to `gpt-4.1` regardless of `--model` (confidence extraction uses `gpt-4.1-mini`).

### Rollout (`rollout/`)
Adopted from Tongyi DeepResearch. `run_multi_react.py` fans out `MultiTurnReactAgent` (`react_agent.py`) across `roll_out_count` × instances with `max_workers`, judges each result inline, and supports `--total_splits`/`--worker_split` for multi-node distribution. Per-dataset question-key extraction happens here (`problem` for browsecomp/deepsearchqa, `prompt_id` for healthbench, `prompt` for researchrubrics) and defines the `question` field aggregation later groups on. Search/visit tools talk to a local FastAPI server (`tools/serve_search.py`, Serper + crawl4ai); `browsecomp-plus` instead uses local FAISS/BM25 retrieval (`searchers/`, `tool_*_bcp.py`) configured entirely via env vars (`SEARCHER_TYPE`, `INDEX_PATH`, `SEARCH_MODEL_NAME`, ... — see `scripts/rollout.sh`). Model-specific vLLM `--tool-call-parser` / `--reasoning-parser` flags are tabulated in `rollout/README.md`.

## Supported tasks (must match across stages)
`browsecomp`, `browsecomp-plus`, `hle`, `deepsearchqa`, `healthbench`, `researchrubrics`. The `--task`/`task=` string drives judge selection, `is_correct` logic, and the aggagent output format simultaneously — keep it consistent between rollout, aggregation, and the package.

## Claude Code skill
`.claude/skills/aggagent/` mirrors the package logic interactively. It operates on a **flat directory of per-trajectory JSONs for one question** (produced by `scripts/collect_trajs.py`, which also has `--list` and `--all` modes), not the `iter{k}/` layout. Invoke with `/aggagent <dir>`.
