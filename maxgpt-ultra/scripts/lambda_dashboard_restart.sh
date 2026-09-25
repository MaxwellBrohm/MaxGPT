#!/usr/bin/env bash
# Restart the dashboard server on the Lambda box so it runs the pulled gui/server.py (the server keeps
# its pipeline code in memory: new flags on the train command never reach a running server). Pauses
# the run first (checkpoint + exit), relaunches the server with the same cards and arguments the
# keeper uses, then presses play. Run ON the box:
#   bash scripts/lambda_dashboard_restart.sh            # cards default to 0,4,5,9
#   CARDS=0,4,5,9 bash scripts/lambda_dashboard_restart.sh
set -u
CARDS="${CARDS:-0,4,5,9}"
API="${API:-http://127.0.0.1:8800}"
ARGS="python gui/server.py --config configs/ultra_lambda_final.yaml --shards data/shards_train --eval-shards data/shards_val_ultra"
cd "$HOME/MaxGPT/maxgpt-ultra" || exit 1
say() { echo "$(date '+%F %T') dashboard-restart: $*"; }
pgrep -f "gui/server.py" >/dev/null || say "no dashboard process is running; will just start one"
P=$(curl -s -m 10 "$API/api/pipeline")
if echo "$P" | grep -q '"running": *true'; then
    say "pausing the run"
    curl -s -m 10 -X POST "$API/api/pause"; echo
    for i in $(seq 1 120); do
        sleep 5
        P=$(curl -s -m 5 "$API/api/pipeline")
        echo "$P" | grep -q '"running": *false' && break
    done
    echo "$P" | grep -q '"running": *false' || { say "still running after $((i*5))s: leaving everything as it is"; exit 1; }
    say "paused after $((i*5))s"
fi
tmux kill-session -t ultra 2>/dev/null; sleep 3
pgrep -f "gui/server.py" >/dev/null && { pkill -f "gui/server.py"; sleep 3; }
pgrep -f "scripts/train.py" >/dev/null && say "WARNING: a train.py process is still alive: $(pgrep -f scripts/train.py | tr '\n' ' ')"
tmux new-session -d -s ultra "cd $HOME/MaxGPT/maxgpt-ultra && source $HOME/venv/bin/activate && export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$CARDS && $ARGS 2>&1 | tee -a $HOME/MaxGPT/ultra_server.log"
for i in $(seq 1 30); do sleep 2; curl -s -m 3 "$API/api/pipeline" >/dev/null 2>&1 && break; done
say "dashboard up after $((i*2))s on cards $CARDS"
WS=$(cut -d' ' -f1 "$HOME/MaxGPT/thermal_state" 2>/dev/null || echo armed)
[ "$WS" = armed ] || { say "watchdog state '$WS': not pressing play (the watchdog resumes)"; exit 0; }
say "pressing play"
curl -s -m 10 -X POST "$API/api/start"; echo
sleep 90
grep -rhE "benchmark suite|annealing|resumed=" $(find runs/pretrain/ranks -newermt "-3 minutes" -type f -name stdout.log 2>/dev/null) 2>/dev/null | sort -u | cut -c1-170
say "train command: $(ps -o args= -p $(pgrep -f 'scripts/train.py' | head -1) 2>/dev/null | grep -o '\-\-suite-dir [^ ]* --suite-every [0-9]*' || echo '(no --suite-dir)')"
