#!/usr/bin/env bash
# Unattended swap of a finished anneal build into the run (Lambda box). Waits for the build to write
# its meta.json, sanity-checks the shares (scripts/anneal_check.py), pauses the run through the
# dashboard (checkpoint + exit), moves the new shards to data/shards_anneal, and presses play so the
# new trainer process loads the new meta. Respects a thermal pause exactly like the keeper does: if
# the watchdog holds the run, the shards are swapped but play is left to the watchdog.
#   tmux new-session -d -s anneal_swap "bash scripts/anneal_autoswap.sh data/shards_anneal_v3 2>&1 | tee -a ~/MaxGPT/anneal_swap.log"
# Refuses to touch anything unless the build finished, the check passes, and the pause completed.
set -u
NEW="${1:?new shard dir, e.g. data/shards_anneal_v3}"
API="${API:-http://127.0.0.1:8800}"
cd "$HOME/MaxGPT/maxgpt-ultra" || exit 1
say() { echo "$(date '+%F %T') autoswap: $*"; }
build_session="${NEW##*/}"; build_session="${build_session#shards_}"      # data/shards_anneal_v3 -> anneal_v3

say "waiting for $NEW/meta.json (build session '$build_session')"
while [ ! -f "$NEW/meta.json" ]; do
    if ! tmux has-session -t "$build_session" 2>/dev/null; then
        say "build session '$build_session' is gone and there is no meta.json: NOT swapping"; exit 1
    fi
    sleep 120
done
say "build finished"
if ! CHECK=$(python3 scripts/anneal_check.py "$NEW"); then
    say "check failed: $CHECK"; say "NOT swapping"; exit 1
fi
say "check passed: $CHECK"

PREV="data/shards_anneal.prev_$(date +%Y%m%d_%H%M)"
[ -e "$PREV" ] && { say "$PREV already exists: NOT swapping"; exit 1; }
P=$(curl -s -m 10 "$API/api/pipeline")
if echo "$P" | grep -q '"running": *true'; then
    say "pausing the run"
    curl -s -m 10 -X POST "$API/api/pause"; echo
    for i in $(seq 1 120); do
        sleep 5
        P=$(curl -s -m 5 "$API/api/pipeline")
        echo "$P" | grep -q '"running": *false' && break
    done
    if ! echo "$P" | grep -q '"running": *false'; then
        say "still running after $((i*5))s: NOT swapping (run left as it was)"; exit 1
    fi
    say "paused after $((i*5))s"
else
    say "run is not running (paused or idle): swapping without a pause"
fi
mv data/shards_anneal "$PREV" && mv "$NEW" data/shards_anneal || { say "mv failed; check data/ by hand"; exit 1; }
say "swapped: $NEW -> data/shards_anneal (previous build kept at $PREV)"
WS=$(cut -d' ' -f1 "$HOME/MaxGPT/thermal_state" 2>/dev/null || echo armed)
if [ "$WS" != armed ]; then
    say "watchdog state is '$WS' (thermal pause): leaving the run paused; the watchdog's resume loads the new shards"
    exit 0
fi
say "pressing play"
curl -s -m 10 -X POST "$API/api/start"; echo
sleep 120
LINE=$(grep -rh "annealing" $(find runs/pretrain/ranks -newermt "-4 minutes" -type f -name stdout.log 2>/dev/null) 2>/dev/null | sort -u | tail -1)
say "new trainer: ${LINE:-(no annealing line yet; check runs/pretrain/ranks/<newest>/attempt_0/0/stdout.log)}"
say "done; delete $PREV once the run is confirmed healthy"
