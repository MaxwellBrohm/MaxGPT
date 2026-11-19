#!/bin/bash
#SBATCH --job-name=maxgpt-ultra
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=47:30:00
#SBATCH --signal=B:USR1@900
#SBATCH --dependency=singleton
#SBATCH --output=logs/%x-%j.out
#
# One link of the MaxGPT-Ultra pretraining chain. Both NAIRR H100 systems cap a job at 48 hours and the
# run needs about six days, so submit several copies of this script at once:
#
#     mkdir -p logs
#     for i in 1 2 3 4 5; do sbatch -A <allocation> slurm/pretrain_chain.sbatch; done
#
# `--dependency=singleton` lets only one job with this name run at a time; each link resumes from the
# latest checkpoint, and a link that finds the run finished exits in seconds. 15 minutes before the time
# limit Slurm sends USR1 to this script, which asks the trainer to checkpoint and exit cleanly.
# SDSC Expanse AI: add `-p nairr-gpu` (and its GPU request form) on the sbatch command line.
# Everything below runs the same under plain bash, which is how scripts/test_slurm_chain.sh tests it.
#
# Settings come from the environment (slurm/env.sh, or in front of sbatch): MAXGPT_RUN, MAXGPT_CONFIG,
# MAXGPT_DATA (the folder holding shards_train/, shards_val_ultra/ and shards_anneal/), or each of
# those three on its own (MAXGPT_TRAIN_SHARDS, MAXGPT_VAL_SHARDS, MAXGPT_ANNEAL_SHARDS), and
# MAXGPT_EVAL_EVERY / MAXGPT_SUITE_EVERY. The launch test in NAIRR.md uses these to run this same
# script for an hour on the proxy shards before the real chain is queued.

set -uo pipefail
ROOT="${MAXGPT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ROOT" || { echo "[chain] cannot cd to $ROOT"; exit 1; }
[ -f slurm/env.sh ] && source slurm/env.sh
RUN="${MAXGPT_RUN:-runs/pretrain}"
CONFIG="${MAXGPT_CONFIG:-configs/ultra_h100.yaml}"
DATA="${MAXGPT_DATA:-data}"
TRAIN="${MAXGPT_TRAIN_SHARDS:-$DATA/shards_train}"
VAL="${MAXGPT_VAL_SHARDS:-$DATA/shards_val_ultra}"
ANNEAL="${MAXGPT_ANNEAL_SHARDS:-$DATA/shards_anneal}"   # handed to the trainer, so it follows MAXGPT_DATA like the others
EVAL_EVERY="${MAXGPT_EVAL_EVERY:-500}"
SUITE_EVERY="${MAXGPT_SUITE_EVERY:-10000}"
GPUS="${MAXGPT_GPUS:-4}"
LAUNCH="${MAXGPT_LAUNCH:-python -m torch.distributed.run --nnodes=1 --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:0 --nproc_per_node=$GPUS}"
mkdir -p "$RUN" logs
say() { echo "[chain] $(date '+%F %T') $*"; }

if [ -f "$RUN/DONE" ]; then
    say "training already finished ($(cat "$RUN/DONE")): nothing to do"
    exit 0
fi
if [ -f "$RUN/GRAD_HALT" ]; then
    say "GRAD_HALT is present ($(cat "$RUN/GRAD_HALT")): the gradient watch stopped the run. Not training."
    say "Read NAIRR.md, 'If the gradient watch speaks', then delete $RUN/GRAD_HALT to continue."
    exit 3
fi

if [ -f "$RUN/ANNEAL_MISSING" ]; then
    say "ANNEAL_MISSING is present ($(cat "$RUN/ANNEAL_MISSING")): the run reached the decay phase without its data."
    say "Build $ANNEAL (slurm/data_build.sbatch), then delete $RUN/ANNEAL_MISSING to continue."
    exit 3
fi

STOP="$RUN/STOP"
rm -f "$STOP"                                   # a stop request left over from the previous link
trap 'say "time limit near: asking the trainer to checkpoint and stop"; touch "$STOP"' USR1
trap 'say "TERM received: asking the trainer to checkpoint and stop"; touch "$STOP"' TERM

say "starting on $(hostname) with $GPUS GPU(s), config $CONFIG, commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
say "shards: train $TRAIN, validation $VAL, decay phase $ANNEAL; eval every $EVAL_EVERY steps, benchmarks every $SUITE_EVERY"
$LAUNCH scripts/train.py --config "$CONFIG" \
    --data "$TRAIN" --eval-data "$VAL" --anneal-data "$ANNEAL" --eval-every "$EVAL_EVERY" \
    --tokenizer tokenizer/maxgpt-ultra.tokenizer.json --suite-dir data/bench --suite-every "$SUITE_EVERY" \
    --out "$RUN" --stop-file "$STOP" &
PID=$!
RC=0
while :; do                                     # `wait` returns early when a trapped signal arrives
    wait "$PID"; RC=$?
    kill -0 "$PID" 2>/dev/null || break
done

if [ -f "$RUN/DONE" ]; then
    say "pretraining finished ($(cat "$RUN/DONE")). Queued links will exit on their own."
    exit 0
fi
if [ -f "$RUN/GRAD_HALT" ]; then
    say "stopped by the gradient watch; see $RUN/GRAD_HALT and NAIRR.md"
    exit 3
fi
if [ -f "$RUN/ANNEAL_MISSING" ]; then
    say "stopped at the decay start: the anneal shards are missing; see $RUN/ANNEAL_MISSING and NAIRR.md"
    exit 3
fi
if [ "$RC" -ne 0 ]; then
    say "the trainer exited with code $RC before finishing. Check the log above; the next link will try to resume."
    exit "$RC"
fi
say "link ended cleanly before the end of training; the next queued link resumes from the checkpoint"
exit 0
