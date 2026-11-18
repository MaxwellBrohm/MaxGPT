#!/bin/bash
# Tests slurm/pretrain_chain.sbatch under plain bash with a stand-in for the training launcher.
# Run from maxgpt-ultra/:  bash scripts/test_slurm_chain.sh
set -u
cd "$(dirname "$0")/.." || exit 1
SCRIPT="$PWD/slurm/pretrain_chain.sbatch"
T="$(mktemp -d /tmp/mgu_chain_XXXXXX)"
fail() { echo "FAIL: $*"; exit 1; }

# The stand-in launcher: finds --out and --stop-file in its arguments and acts out FAKE_MODE.
cat > "$T/fake_launch.sh" <<'FAKE'
#!/bin/bash
out=""; stop=""
while [ $# -gt 0 ]; do
    case "$1" in --out) out="$2"; shift;; --stop-file) stop="$2"; shift;; esac
    shift
done
echo "invoked" >> "$out/LAUNCHED"
[ -e "$stop" ] && echo "stale" >> "$out/STALE_STOP_SEEN"
case "${FAKE_MODE:-finish}" in
    finish) echo '{"step": 5, "total_steps": 5}' > "$out/DONE"; exit 0;;
    halt)   echo '{"event": "grad_halt"}' > "$out/GRAD_HALT"; exit 0;;
    crash)  exit 7;;
    pause)  for _ in $(seq 1 200); do [ -e "$stop" ] && exit 0; sleep 0.1; done; exit 9;;
esac
FAKE
chmod +x "$T/fake_launch.sh"
run() { ( cd "$T" && MAXGPT_ROOT="$T" MAXGPT_RUN="$1" MAXGPT_LAUNCH="$T/fake_launch.sh" FAKE_MODE="$2" bash "$SCRIPT" ) > "$T/$1.log" 2>&1; }

echo "[1] a link that reaches the end reports it and exits 0"
mkdir -p "$T/r1"; run r1 finish; rc=$?
[ $rc -eq 0 ] && [ -f "$T/r1/DONE" ] && grep -q "pretraining finished" "$T/r1.log" || fail "finish: rc=$rc $(cat "$T/r1.log")"

echo "[2] a link that finds DONE does not start the trainer"
mkdir -p "$T/r2"; echo '{"step": 5}' > "$T/r2/DONE"; run r2 finish; rc=$?
[ $rc -eq 0 ] && [ ! -f "$T/r2/LAUNCHED" ] && grep -q "nothing to do" "$T/r2.log" || fail "done: rc=$rc launched=$(ls "$T/r2")"

echo "[3] a link that finds GRAD_HALT refuses to train and exits 3"
mkdir -p "$T/r3"; echo '{"event": "grad_halt"}' > "$T/r3/GRAD_HALT"; run r3 finish; rc=$?
[ $rc -eq 3 ] && [ ! -f "$T/r3/LAUNCHED" ] || fail "grad halt: rc=$rc launched=$(ls "$T/r3")"

echo "[4] a stale STOP is cleared, and USR1 makes the trainer stop cleanly (exit 0, no DONE)"
mkdir -p "$T/r4"; touch "$T/r4/STOP"
( cd "$T" && MAXGPT_ROOT="$T" MAXGPT_RUN=r4 MAXGPT_LAUNCH="$T/fake_launch.sh" FAKE_MODE=pause bash "$SCRIPT" ) > "$T/r4.log" 2>&1 &
pid=$!
for _ in $(seq 1 100); do [ -f "$T/r4/LAUNCHED" ] && break; sleep 0.1; done
[ -f "$T/r4/LAUNCHED" ] || fail "the trainer never started: $(cat "$T/r4.log")"
sleep 0.5
[ ! -f "$T/r4/STALE_STOP_SEEN" ] || fail "the trainer saw a STOP file left over from the previous link"
kill -USR1 "$pid"; wait "$pid"; rc=$?
[ $rc -eq 0 ] && [ -f "$T/r4/STOP" ] && [ ! -f "$T/r4/DONE" ] && grep -q "next queued link resumes" "$T/r4.log" || fail "usr1: rc=$rc $(cat "$T/r4.log")"

echo "[5] a trainer crash is passed on as a non-zero exit"
mkdir -p "$T/r5"; run r5 crash; rc=$?
[ $rc -eq 7 ] && grep -q "exited with code 7" "$T/r5.log" || fail "crash: rc=$rc $(cat "$T/r5.log")"

echo "[6] a halt raised during the link exits 3"
mkdir -p "$T/r6"; run r6 halt; rc=$?
[ $rc -eq 3 ] && grep -q "stopped by the gradient watch" "$T/r6.log" || fail "halt: rc=$rc"

rm -rf "$T"
echo "ALL CHECKS PASSED"
