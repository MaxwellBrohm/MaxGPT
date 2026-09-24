#!/usr/bin/env bash
# Build the decay-phase annealing shards on the Lambda box (CPU only; the training run is untouched).
#   cd ~/MaxGPT/maxgpt-ultra && git pull && bash scripts/anneal_build.sh
# Resumable: re-running continues from data/shards_anneal/progress.json. Done when meta.json exists.
cd "$HOME/MaxGPT/maxgpt-ultra" || exit 1
if [ -f data/shards_anneal/meta.json ]; then echo "anneal shards already built (data/shards_anneal/meta.json)"; exit 0; fi
if tmux has-session -t anneal 2>/dev/null; then echo "anneal build already running (tmux attach -t anneal)"; exit 0; fi
mkdir -p runs/anneal
tmux new-session -d -s anneal "cd $HOME/MaxGPT/maxgpt-ultra && source $HOME/venv/bin/activate && export RAYON_NUM_THREADS=40 TOKENIZERS_PARALLELISM=true && nice -n 15 python -u scripts/prepare_data.py --config configs/ultra.yaml --mix anneal --shards-out data/shards_anneal --max-tokens 6.0e9 --skip-posttrain --metrics-out runs/anneal/metrics.jsonl 2>&1 | tee -a $HOME/MaxGPT/anneal_build.log"
echo "anneal build started $(date +%T) in tmux 'anneal'; progress: grep '^\[data\]' ~/MaxGPT/anneal_build.log | tail -1"
