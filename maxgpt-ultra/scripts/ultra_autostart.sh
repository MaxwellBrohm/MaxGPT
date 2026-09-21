#!/usr/bin/env bash
# Unattended hand-off from the shakedown A/B to the real Ultra run on the Lambda box.
#
#   tmux new -d -s autostart "bash ~/MaxGPT/maxgpt-ultra/scripts/ultra_autostart.sh 1,2,3,4,5,7,8,9 6 0"
#                                                             ^ training cards   ^ thermometer  ^ buffer/idle
#
# 1. waits until every A/B run has exited (or --max-wait hours pass),
# 2. decides the recipe (scripts/ab_verdict.py -> configs/ultra_lambda_final.yaml),
# 3. carves the held-out val shard (scripts/holdout_shard.py) if not done,
# 4. starts the dashboard on the training cards (tmux 'ultra'), the thermal watchdog in
#    pause/resume mode (tmux 'thermal'), and presses play.
# Everything is logged to ~/MaxGPT/ultra_autostart.log. Re-running is safe: it never starts a
# second dashboard or watchdog if they are already up.
CARDS="${1:?training cards, e.g. 1,2,3,4,5,7,8,9}"
THERMO="${2:?thermometer card}"
MAX_WAIT_H="${3:-14}"
AB_MAP="${4:-}"     # optional 'variant:card,...': relaunch an A/B run the watchdog killed, once its card has cooled
LOG="$HOME/MaxGPT/ultra_autostart.log"
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
cd "$HOME/MaxGPT/maxgpt-ultra" || exit 1
source "$HOME/venv/bin/activate"
export PYTHONUNBUFFERED=1

say "autostart: cards=$CARDS thermometer=$THERMO; waiting for the A/B runs (max ${MAX_WAIT_H}h)"
deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
declare -A relaunches
while :; do
    n=0
    for v in adamw normuon adamw_arch normuon_arch; do
        if grep -qE "AB-EXIT|done at step" "$HOME/MaxGPT/ab_$v.log" 2>/dev/null; then
            n=$((n + 1))                                  # finished or crashed on its own
        elif ! tmux has-session -t "ab_$v" 2>/dev/null; then
            # killed by the watchdog (no exit line): relaunch on its card once that card is cool, up to 5 times
            card=$(echo "$AB_MAP" | tr ',' '\n' | grep "^$v:" | cut -d: -f2)
            ws=$(cut -d' ' -f1 "$HOME/MaxGPT/thermal_state" 2>/dev/null || echo armed)
            t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits -i "${card:-0}" 2>/dev/null || echo 99)
            if [ -n "$card" ] && [ "${relaunches[$v]:-0}" -lt 5 ] && [ "$ws" = armed ] && [ "$t" -le 60 ]; then
                relaunches[$v]=$(( ${relaunches[$v]:-0} + 1 ))
                echo "[relaunched by autostart on GPU $card $(date +%T)]" >> "$HOME/MaxGPT/ab_$v.log"
                tmux new-session -d -s "ab_$v" "cd $HOME/MaxGPT/maxgpt-ultra && source $HOME/venv/bin/activate && export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$card && python scripts/train.py --config configs/ab/$v.yaml --data data/shards_ab --eval-data data/shards_val --eval-every 250 --out runs/ab/$v >> $HOME/MaxGPT/ab_$v.log 2>&1; echo AB-EXIT=\$? >> $HOME/MaxGPT/ab_$v.log"
                say "A/B $v was stopped by the watchdog: relaunched on GPU $card (card at ${t}C), attempt ${relaunches[$v]}"
            elif [ -z "$card" ] || [ "${relaunches[$v]:-0}" -ge 5 ]; then
                n=$((n + 1))                              # cannot or will not relaunch: treat as over
            fi
        fi
    done
    [ "$n" -ge 4 ] && { say "all four A/B runs are over"; break; }
    [ "$(date +%s)" -ge "$deadline" ] && { say "A/B still running after ${MAX_WAIT_H}h: deciding on what has finished"; break; }
    sleep 300
done

say "verdict:"
python scripts/ab_verdict.py --runs runs/ab --base configs/ultra_lambda.yaml --out configs/ultra_lambda_final.yaml 2>&1 | tee -a "$LOG"

if [ ! -f data/shards_val_ultra/meta.json ]; then
    python scripts/holdout_shard.py --shards data/shards --train data/shards_train --val data/shards_val_ultra 2>&1 | tee -a "$LOG"
fi

# stop the A/B runs' tmux sessions if any are still alive (they hold GPUs we are about to use)
for v in adamw normuon adamw_arch normuon_arch; do tmux kill-session -t "ab_$v" 2>/dev/null; done
pkill -f "[c]onfigs/ab/" 2>/dev/null; sleep 5

if ! tmux has-session -t ultra 2>/dev/null; then
    tmux new-session -d -s ultra "cd $HOME/MaxGPT/maxgpt-ultra && source $HOME/venv/bin/activate && export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$CARDS && python gui/server.py --config configs/ultra_lambda_final.yaml --shards data/shards_train --eval-shards data/shards_val_ultra 2>&1 | tee -a $HOME/MaxGPT/ultra_server.log"
    say "dashboard started on cards $CARDS (tmux ultra)"
    sleep 10
fi
# the A/B phase ran a kill-mode watchdog; Ultra needs the pause/resume one, so always start fresh
tmux kill-session -t thermal 2>/dev/null; sleep 2
if ! tmux has-session -t thermal 2>/dev/null; then
    tmux new-session -d -s thermal "cd $HOME/MaxGPT/maxgpt-ultra && source $HOME/venv/bin/activate && python -u scripts/thermal_watch.py --thermometer $THERMO --max-temp 92 --baseline-after-load 900 --idle-rise 8 --cpu-rise 8 --pause-cmd 'curl -s -m 5 -X POST http://127.0.0.1:8800/api/pause' --resume-cmd 'curl -s -m 5 -X POST http://127.0.0.1:8800/api/start' --kill-sessions ultra 2>&1 | tee -a $HOME/MaxGPT/thermal_watch.out"
    say "watchdog started (pause/resume mode, thermometer $THERMO, baseline 15 min into load, trip at +8C)"
fi
R=$(curl -s -m 10 -X POST http://127.0.0.1:8800/api/start)
say "pressed play: $R"
sleep 90
say "pipeline: $(curl -s -m 5 http://127.0.0.1:8800/api/pipeline | cut -c1-200)"
say "dashboard http://localhost:8800 via: ssh -L 8800:localhost:8800 lambda"

# keeper: stay alive and press play again whenever the pipeline is idle for a reason that is not the
# watchdog (a crashed stage, a stray OOM from someone else's job on one of our cards, a reboot of
# the dashboard). Never overrides a thermal pause: it checks the watchdog's state file first.
say "keeper: checking every 10 min"
while :; do
    sleep 600
    if ! tmux has-session -t ultra 2>/dev/null; then
        tmux new-session -d -s ultra "cd $HOME/MaxGPT/maxgpt-ultra && source $HOME/venv/bin/activate && export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$CARDS && python gui/server.py --config configs/ultra_lambda_final.yaml --shards data/shards_train --eval-shards data/shards_val_ultra 2>&1 | tee -a $HOME/MaxGPT/ultra_server.log"
        say "keeper: dashboard was down, restarted it"; sleep 15
    fi
    if ! tmux has-session -t thermal 2>/dev/null; then
        tmux new-session -d -s thermal "cd $HOME/MaxGPT/maxgpt-ultra && source $HOME/venv/bin/activate && python -u scripts/thermal_watch.py --thermometer $THERMO --max-temp 92 --baseline-after-load 900 --idle-rise 8 --cpu-rise 8 --pause-cmd 'curl -s -m 5 -X POST http://127.0.0.1:8800/api/pause' --resume-cmd 'curl -s -m 5 -X POST http://127.0.0.1:8800/api/start' --kill-sessions ultra 2>&1 | tee -a $HOME/MaxGPT/thermal_watch.out"
        say "keeper: watchdog was down, restarted it"; sleep 5
    fi
    P=$(curl -s -m 5 http://127.0.0.1:8800/api/pipeline)
    WS=$(cut -d' ' -f1 "$HOME/MaxGPT/thermal_state" 2>/dev/null || echo armed)
    if echo "$P" | grep -q '"running": *false' && [ "$WS" = "armed" ] && ! echo "$P" | grep -q '"status": *"ready"'; then
        R=$(curl -s -m 10 -X POST http://127.0.0.1:8800/api/start)
        say "keeper: pipeline idle (watchdog $WS): pressed play -> $R"
    fi
done
