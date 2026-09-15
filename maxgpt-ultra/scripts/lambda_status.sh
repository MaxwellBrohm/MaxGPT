#!/usr/bin/env bash
# One-screen status of everything MaxGPT on the Lambda box. Safe to run any time:
#   bash ~/MaxGPT/maxgpt-ultra/scripts/lambda_status.sh
cd "$HOME/MaxGPT/maxgpt-ultra" || exit 1
source "$HOME/venv/bin/activate" 2>/dev/null
echo "== host =="
echo "$(date '+%F %T')  up $(uptime -p | sed 's/up //')  load $(cut -d' ' -f1-3 /proc/loadavg)"
echo "last boots: $(last -x reboot 2>/dev/null | head -2 | awk '{print $5,$6,$7,$8}' | paste -sd'|')"
if journalctl -q -b -1 -n 1 >/dev/null 2>&1; then
    echo "previous boot ended with:"; journalctl -q -b -1 -n 4 --no-pager 2>/dev/null | cut -c1-150 | sed 's/^/   /'
fi
echo "== tmux sessions =="; tmux ls 2>/dev/null || echo "   (none)"
echo "== gpus (index, used MiB, util %, temp C, power W) =="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader | sed 's/^/   /'
echo "== data build =="
grep -E "^\[data\]" "$HOME/MaxGPT/data_build.log" 2>/dev/null | tail -1 | sed 's/^/   /'
echo "   shards on disk: $(ls data/shards 2>/dev/null | grep -c shard_)   meta.json: $([ -f data/shards/meta.json ] && echo DONE || echo not-yet)"
echo "== A/B (124M) =="
python scripts/ab_report.py 2>/dev/null | sed 's/^/   /'
for v in adamw normuon adamw_arch normuon_arch; do
    f="$HOME/MaxGPT/ab_$v.log"
    [ -f "$f" ] && grep -E "AB-EXIT|compiled forward failed|Root Cause" "$f" | tail -1 | cut -c1-140 | sed "s/^/   [$v] /"
done
echo "== gpu watch (last 3) =="; tail -3 "$HOME/MaxGPT/gpu_watch.log" 2>/dev/null | sed 's/^/   /'
