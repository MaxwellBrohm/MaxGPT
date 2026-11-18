#!/bin/bash
#SBATCH --job-name=maxgpt-proxy
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=08:00:00
#SBATCH --array=0-7
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/%x-%A_%a.out
#
# The proxy A/B (docs/research_2026-11-18_restart.md section 7): eight single-GPU runs of the 176M
# proxy, about 3 GPU-hours each. Three baseline seeds give the noise floor; every other run changes
# one thing. Afterwards: python scripts/proxy_verdict.py --runs runs/proxy
#     sbatch -A <allocation> slurm/proxy_ab.sbatch
#     sbatch -A <allocation> --array=8 slurm/proxy_ab.sbatch    # the old-data-mix arm, once its shards exist
# A run that hits the time limit checkpoints and can be resubmitted; a finished run is skipped.

set -uo pipefail
ROOT="${MAXGPT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ROOT" || exit 1
[ -f slurm/env.sh ] && source slurm/env.sh
DATA="${MAXGPT_DATA:-data}"
ARMS=(base:0 base:1 base:2 lr_2e-4:0 lr_4.5e-4:0 peri_ln:0 peri_ln_noscale:0 relu2:0 oldmix:0)
PICK="${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}"
ARM="${PICK%%:*}"; SEED="${PICK##*:}"
OUT="runs/proxy/${ARM}_s${SEED}"
TRAIN="$DATA/shards_proxy_train"; VAL="$DATA/shards_proxy_val"
[ "$ARM" = oldmix ] && { TRAIN="$DATA/shards_proxy_old_train"; VAL="$DATA/shards_proxy_old_val"; }
mkdir -p "$OUT" logs
say() { echo "[proxy $ARM s$SEED] $(date '+%F %T') $*"; }
[ -f "$TRAIN/meta.json" ] || { say "no shards at $TRAIN (build them first, see NAIRR.md)"; exit 1; }

if [ ! -f "$OUT/DONE" ]; then
    STOP="$OUT/STOP"; rm -f "$STOP"
    trap 'say "time limit near: asking the trainer to stop"; touch "$STOP"' USR1 TERM
    python scripts/train.py --config "configs/proxy/$ARM.yaml" --data "$TRAIN" --eval-data "$VAL" --eval-every 250 \
        --tokenizer tokenizer/maxgpt-ultra.tokenizer.json --seed "$SEED" --out "$OUT" --stop-file "$STOP" &
    PID=$!
    while :; do wait "$PID"; RC=$?; kill -0 "$PID" 2>/dev/null || break; done
    [ -f "$OUT/DONE" ] || { say "not finished (exit $RC); resubmit this array index to continue"; exit "${RC:-1}"; }
fi
if [ ! -f "$OUT/suite.json" ]; then
    CK="$OUT/checkpoints/$(python3 -c "import json;print(json.load(open('$OUT/checkpoints/latest.json'))['path'])")"
    python scripts/eval_suite.py --config "configs/proxy/$ARM.yaml" --checkpoint "$CK" \
        --tokenizer tokenizer/maxgpt-ultra.tokenizer.json --suite-dir data/bench --out "$OUT/suite.json" || exit 1
fi
say "done: $(python3 -c "import json;print('benchmark average', round(json.load(open('$OUT/suite.json'))['avg'], 4))")"
