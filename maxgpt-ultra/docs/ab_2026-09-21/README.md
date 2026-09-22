# Shakedown A/B, Sep 21-22 2026 (Lambda box, one Titan RTX per variant)

Four 124M variants of the shakedown recipe, same 1.1B training tokens (shards 0-10 of the 100B
build), held-out shard 11, same batch (131k tokens/step), same LR schedule; configs in
`configs/ab/`. Each `*.metrics.jsonl` is the trainer's log (train loss every 10 steps, held-out
`val_loss` every 250 steps).

| variant | recipe | val loss (mean of last 3) | min val |
|---|---|---|---|
| adamw | original: AdamW, plain blocks | 2.993 | 2.880 |
| adamw_arch | + gated attention, normalized value residual, 1/sqrt(depth) norm scaling | 2.803 | 2.746 |
| normuon | NorMuon + cautious weight decay (AdamW for embeddings / 1D) | 2.840 | 2.715 |
| **normuon_arch** | both | **2.786** | **2.663** |

Verdict (`scripts/ab_verdict.py`, 0.5% margin): normuon_arch, 6.9% lower held-out loss than the
original recipe. The NorMuon variants trailed at step 4000 (3.43 vs 3.13) and overtook during the
LR decay. The 1.1B Ultra run uses this recipe (`configs/ultra_lambda_final.yaml`).
