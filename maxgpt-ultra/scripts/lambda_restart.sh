#!/usr/bin/env bash
# Restart MaxGPT work on the Lambda box after a reboot / outage, on at most 8 GPUs:
#   bash ~/MaxGPT/maxgpt-ultra/scripts/lambda_restart.sh            # data build + the 4 A/B runs
#   bash ~/MaxGPT/maxgpt-ultra/scripts/lambda_restart.sh data       # only the data build
# Idempotent: anything already running in tmux is left alone. The data build resumes from its
# last shard checkpoint; an A/B run resumes from its checkpoint if it saved one, else restarts.
# Also starts a 5-minute GPU temperature/power log (~/MaxGPT/gpu_watch.log).
WHAT="${1:-all}"
cd "$HOME/MaxGPT/maxgpt-ultra" || exit 1
git pull -q 2>/dev/null && echo "repo at $(git log --oneline -1 | cut -c1-60)"
source "$HOME/venv/bin/activate"
export PYTHONUNBUFFERED=1

if tmux has-session -t data 2>/dev/null; then
    echo "data build: already running"
elif [ -f data/shards/meta.json ]; then
    echo "data build: DONE (meta.json exists)"
else
    tmux new-session -d -s data "cd $HOME/MaxGPT/maxgpt-ultra && source $HOME/venv/bin/activate && nice -n 10 python -u scripts/prepare_data.py --config configs/ultra.yaml --metrics-out runs/data/metrics.jsonl 2>&1 | tee -a $HOME/MaxGPT/data_build.log"
    echo "data build: started (resumes from data/shards/progress.json)"
fi

if ! tmux has-session -t gpuwatch 2>/dev/null; then
    tmux new-session -d -s gpuwatch "while true; do echo \"\$(date '+%F %T') \$(nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu --format=csv,noheader,nounits | tr '\n' '|')\" >> $HOME/MaxGPT/gpu_watch.log; sleep 300; done"
    echo "gpu watch: started (5-min temp/power log)"
fi

[ "$WHAT" = data ] && exit 0

i=0
for v in adamw normuon adamw_arch normuon_arch; do
    g="$((2*i)),$((2*i+1))"; i=$((i+1))
    if tmux has-session -t "ab_$v" 2>/dev/null; then
        echo "ab_$v: already running"; continue
    fi
    if grep -q "done at step" "$HOME/MaxGPT/ab_$v.log" 2>/dev/null; then
        echo "ab_$v: finished"; continue
    fi
    if [ -f "runs/ab/$v/checkpoints/latest.json" ]; then
        echo "ab_$v: resuming from its checkpoint on GPUs $g"
    else
        rm -rf "runs/ab/$v"; : > "$HOME/MaxGPT/ab_$v.log"
        echo "ab_$v: starting fresh on GPUs $g"
    fi
    tmux new-session -d -s "ab_$v" "cd $HOME/MaxGPT/maxgpt-ultra && source $HOME/venv/bin/activate && export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$g && python -m torch.distributed.run --nnodes=1 --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:0 --nproc_per_node=2 scripts/train.py --config configs/ab/$v.yaml --data data/shards_ab --eval-data data/shards_val --eval-every 250 --out runs/ab/$v >> $HOME/MaxGPT/ab_$v.log 2>&1; echo AB-EXIT=\$? >> $HOME/MaxGPT/ab_$v.log"
done
echo "check in a few minutes with: bash scripts/lambda_status.sh"
