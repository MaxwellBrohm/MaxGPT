#!/usr/bin/env bash
# Build the decay-phase annealing shards on the Lambda box (CPU only; the training run is untouched).
#   cd ~/MaxGPT/maxgpt-ultra && git pull && bash scripts/anneal_build.sh
# Resumable: re-running continues from $OUT/progress.json. Done when $OUT/meta.json exists.
# OUT=data/shards_anneal_v2 bash scripts/anneal_build.sh   builds into another directory (tmux
# session and log are named after it), so a rebuild never touches the shards the config points at
# until you swap the directories yourself.
cd "$HOME/MaxGPT/maxgpt-ultra" || exit 1
OUT="${OUT:-data/shards_anneal}"
TAG="$(basename "$OUT")"                       # shards_anneal -> tmux 'anneal', log anneal_build.log
SESSION="${TAG#shards_}"
LOG="$HOME/MaxGPT/${SESSION}_build.log"
if [ -f "$OUT/meta.json" ]; then echo "anneal shards already built ($OUT/meta.json)"; exit 0; fi
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "build already running (tmux attach -t $SESSION)"; exit 0; fi
mkdir -p "runs/$SESSION"
tmux new-session -d -s "$SESSION" "cd $HOME/MaxGPT/maxgpt-ultra && source $HOME/venv/bin/activate && export RAYON_NUM_THREADS=40 TOKENIZERS_PARALLELISM=true && nice -n 15 python -u scripts/prepare_data.py --config configs/ultra.yaml --mix anneal --shards-out $OUT --max-tokens 6.0e9 --skip-posttrain --metrics-out runs/$SESSION/metrics.jsonl 2>&1 | tee -a $LOG"
echo "anneal build -> $OUT started $(date +%T) in tmux '$SESSION'; progress: grep '^\[data\]' $LOG | tail -1"
