#!/bin/bash
#SBATCH --job-name=maxgpt-data
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#
# Build every dataset the run needs, in order, on CPU cores (the GPU is requested only because the
# NAIRR H100 partitions are GPU allocations; ask the site whether a zero-GPU job is allowed).
#     sbatch -A <allocation> slurm/data_build.sbatch            # everything
#     STAGE=proxy sbatch -A <allocation> slurm/data_build.sbatch   # only the small proxy shards (about 30 min)
# Every step skips itself when its output exists and resumes when interrupted, so resubmitting is safe.
# Needs outbound internet for the downloads; if compute nodes have none, run the fetch_sources.py
# lines on a login node first (they only download) and resubmit this.

set -uo pipefail
ROOT="${MAXGPT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ROOT" || exit 1
[ -f slurm/env.sh ] && source slurm/env.sh
DATA="${MAXGPT_DATA:-data}"
SRC="${MAXGPT_SOURCES_ROOT:-$DATA/sources}"
CONFIG="${MAXGPT_CONFIG:-configs/ultra_h100.yaml}"
STAGE="${STAGE:-all}"
export MAXGPT_SOURCES_ROOT="$SRC" TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
mkdir -p "$DATA" "$SRC" logs runs/data
say() { echo "[data] $(date '+%F %T') $*"; }
run() { say "$*"; "$@" || { say "FAILED: $*"; exit 1; }; }
[ -n "${HF_TOKEN:-}" ] || say "WARNING: HF_TOKEN is not set; anonymous downloads are rate-limited per IP"

# 1) Sources for the pretraining mix (about 500 GB), then a 2-minute check that each one opens.
run python scripts/fetch_sources.py --mix pretrain --root "$SRC" --tokens 1.0e11
run python scripts/probe_sources.py --mix pretrain --sources-root "$SRC"

# 2) Proxy shards for the A/B: the first 3.3B tokens of the same stream (also trains the tokenizer
#    on the new mix if tokenizer/maxgpt-ultra.tokenizer.json does not exist yet).
if [ ! -f "$DATA/shards_proxy_val/meta.json" ]; then
    run python scripts/prepare_data.py --config "$CONFIG" --mix pretrain --shards-out "$DATA/shards_proxy" \
        --max-tokens 3.3e9 --skip-posttrain
    run python scripts/holdout_shard.py --shards "$DATA/shards_proxy" --train "$DATA/shards_proxy_train" --val "$DATA/shards_proxy_val"
fi
[ "$STAGE" = proxy ] && { say "proxy stage done"; exit 0; }

# 3) The 100B-token pretraining shards (about 11 hours on 96 cores), with the SFT and DPO data,
#    then one shard held out for validation.
run python scripts/prepare_data.py --config "$CONFIG" --mix pretrain --shards-out "$DATA/shards" \
    --metrics-out runs/data/metrics.jsonl
if [ ! -f "$DATA/shards_val_ultra/meta.json" ]; then
    run python scripts/holdout_shard.py --shards "$DATA/shards" --train "$DATA/shards_train" --val "$DATA/shards_val_ultra"
fi

# 4) The decay-phase mix (8B tokens), continuing the code and Wikipedia streams after the rows the
#    pretraining build consumed. Needs data/sft.jsonl from step 3 for its chat share.
run python scripts/fetch_sources.py --mix anneal --root "$SRC" --tokens 8.0e9
run python scripts/prepare_data.py --config "$CONFIG" --mix anneal --shards-out "$DATA/shards_anneal" \
    --max-tokens 8.0e9 --skip-posttrain --continue-from "$DATA/shards/meta.json"
say "all data built. Next: sbatch slurm/pretrain_chain.sbatch (see NAIRR.md)"
