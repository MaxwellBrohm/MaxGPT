#!/usr/bin/env bash
# MaxGPT keeper: keep the Ultra run alive without stepping on other GPU work.
#
# Runs every 30 min from Windows Task Scheduler:
#   wsl.exe -u maxwell -e bash -lc ~/MaxGPT/maxgpt-ultra/scripts/keeper.sh
#
# Rules:
#   1. If any MaxGPT stage process is already running, do nothing.
#   2. If anything else holds >2GB VRAM (another training run, a game), stand down.
#   3. Otherwise start the GUI server (if down) and press play; the pipeline
#      resumes from its checkpoints / progress.json on its own.
# Actions are appended to ~/MaxGPT/keeper.log.

LOG="$HOME/MaxGPT/keeper.log"
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

cd "$HOME/MaxGPT/maxgpt-ultra" || { say "ERROR: repo not found"; exit 1; }

# 1) our stack already busy?
if pgrep -f "scripts/(train|sft|dpo|prepare_data)\.py" > /dev/null; then
    exit 0
fi

# 2) someone else on the card? (idle desktop measures ~0.5GB; training/games are GBs)
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
USED=${USED:-0}
if [ "$USED" -gt 2000 ]; then
    say "stand down: GPU busy (${USED}MiB used by something else)"
    exit 0
fi

# 3) server up? if not, start it
if ! curl -s -m 3 http://localhost:8800/api/pipeline > /dev/null; then
    say "starting GUI server"
    nohup "$HOME/venv/bin/python" gui/server.py --config configs/ultra.yaml \
        >> "$HOME/MaxGPT/server.log" 2>&1 &
    sleep 8
fi

# 4) press play (harmless if already complete; resumes the current stage otherwise)
R=$(curl -s -m 5 -X POST http://localhost:8800/api/start)
say "pressed play (gpu ${USED}MiB): $R"
