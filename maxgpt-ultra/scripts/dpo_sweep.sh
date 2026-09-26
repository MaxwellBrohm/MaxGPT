#!/usr/bin/env bash
# The DPO beta sweep the research report asks for (docs/research_2026-09-22.md 3.2): 0.1 is the
# value Zephyr tuned at 7B, SmolLM2-1.7B used 0.5, and small models degrade faster under DPO, so
# beta is the lever. Runs the three betas one after another on the same SFT checkpoint with ONE
# shared reference-logprob cache (computed once), then scores each result on the fixed suite.
# Keep the SFT checkpoint as the shippable fallback; pick by the suite, never by win-rate.
#   bash scripts/dpo_sweep.sh runs/sft/checkpoints/ckpt_XXXXXXXX.pt [data/prefs.jsonl] [0.1,0.3,0.5]
set -u
INIT="${1:?SFT checkpoint}"
DATA="${2:-data/prefs.jsonl}"
BETAS="${3:-0.1,0.3,0.5}"
CFG="${CFG:-configs/ultra_lambda_final.yaml}"
TOK="${TOK:-tokenizer/maxgpt-ultra.tokenizer.json}"
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs/dpo_sweep
for b in ${BETAS//,/ }; do
    out="runs/dpo_sweep/beta_$b"
    echo "=== beta $b -> $out ($(date +%T)) ==="
    python scripts/dpo.py --config "$CFG" --init "$INIT" --tokenizer "$TOK" --data "$DATA" \
        --out "$out" --beta "$b" --ref-cache runs/dpo_sweep/ref_logps.npz "${@:4}" || { echo "beta $b failed"; continue; }
    ck=$(python3 -c "import json;print(json.load(open('$out/checkpoints/latest.json'))['path'])" 2>/dev/null)
    [ -n "$ck" ] && python scripts/eval_suite.py --config "$CFG" --checkpoint "$out/checkpoints/$ck" \
        --tokenizer "$TOK" --suite-dir data/bench --out "$out/suite.json"
done
echo "=== SFT fallback on the same suite ==="
python scripts/eval_suite.py --config "$CFG" --checkpoint "$INIT" --tokenizer "$TOK" --suite-dir data/bench --out runs/dpo_sweep/sft_suite.json
echo "compare: python3 -c \"import json,glob; [print(f, round(100*json.load(open(f))['avg'],1)) for f in sorted(glob.glob('runs/dpo_sweep/**/suite.json', recursive=True))]\""
