#!/bin/bash
#SBATCH -J qwen35_serve
#SBATCH -N 1
#SBATCH --gres=gpu:a100:8
#SBATCH -t 8:00:00
#SBATCH -p dev
#SBATCH -o /fsx/xinnanzh/AggAgent/logs/serve_qwen35_%j.log
#SBATCH -e /fsx/xinnanzh/AggAgent/logs/serve_qwen35_%j.log

# Serves Qwen/Qwen3.5-35B-A3B for rollout. Point --api_base at the URL printed below.
# NOTE: p4d A100s are the 40 GB variant, so 66 GB of bf16 weights need tp>=4;
# tp=2 OOMs during cudagraph memory profiling.
#
# --enable-prefix-caching must be passed EXPLICITLY for this model, and it matters a lot.
# Qwen3.5-35B-A3B is a hybrid (every 4th layer full attention, the rest linear
# attention/GDN). vllm/config/model.py:1747 `is_prefix_caching_supported` returns False
# for attn_type == "hybrid", but that value only feeds `default_prefix_caching` in
# arg_utils.py:2107 `_set_default_chunked_prefill_and_prefix_caching_args` -- it is a
# DEFAULT, not a hard block. Passing the flag makes enable_prefix_caching non-None, which
# skips the default branch entirely (the accompanying warning is gated on
# runner_type == "pooling", which does not apply here).
#
# Do not also pass --mamba-cache-mode: models/config.py:437 sets it automatically to
# "align" once prefix caching is on, since this model does not declare
# supports_mamba_prefix_caching. Passing "all" explicitly makes qwen3_5.py:471 abort.
#
# Why it matters: the agentic loops here (rollout, aggregation, verify) resend the whole
# growing message list every turn, so without caching total prefill is O(n^2) in turns.
# Measured with it off: ~16000 tok/s prefill vs ~250 tok/s generation, prefix cache hit
# rate 0.0%, KV cache only 16% used and no requests queued -- i.e. compute going almost
# entirely into recomputing identical prefixes. The chat template renders history
# append-only (verified: turn-N render is a strict prefix of turn-N+1), so the prefix is
# stable and cacheable.
cd /fsx/xinnanzh/AggAgent

echo "NODE: $(hostname)  IP: $(hostname -i)"
echo "API_BASE: http://$(hostname):6000/v1"

# On TIMEOUT or scancel, Slurm signals this script but vLLM's TP workers are NOT
# reaped with their parent -- they survive as orphans holding ~40 GB per GPU on a
# node Slurm already reports as idle. Reap them explicitly.
cleanup() {
    kill -TERM "$VLLM_PID" 2>/dev/null
    sleep 10
    pkill -9 -u "$(id -u)" -f 'VLLM::Worker' 2>/dev/null
    echo "cleanup: remaining GPU compute apps:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
}
trap cleanup EXIT TERM INT

.venv/bin/vllm serve Qwen/Qwen3.5-35B-A3B \
    --served-model-name Qwen3.5-35B-A3B \
    --host 0.0.0.0 --port 6000 \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.92 \
    --enable-prefix-caching \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --max-model-len 128000 \
    --trust-remote-code &
VLLM_PID=$!
wait "$VLLM_PID"
