# MaxGPT-Ultra: what to change for the restart on NAIRR H100s

**Date: 18 Nov 2026. Scope: a from-scratch restart of the 1.1B run on one 4x H100 80GB node (NAIRR Pilot Start-Up, decision pending).**

Method: five scoped research lanes (architecture, data, optimizer and recipe, NAIRR and H100 engineering, reference scores), then the claims that drive a recommendation were re-read at the primary source. Full lane evidence, with every source marked read or snippet, is in `research_2026-11-18_restart_lanes/`. Items marked (verified) below were re-read by the orchestrator at the paper, repo or official page. Nothing here has been run on hardware.

---

## 1. Verdict

1. **Restart from scratch: yes.** It is forced (the checkpoint, the 187 GB of shards and the tokenizer file exist only on the powered-down CCM server) and cheap (the old run reached about 4% of the token budget, roughly 6 hours of one H100 node). It also puts Ultra in bf16, the same number format as the Planck curve.
2. **Keep the envelope and the core design.** The submitted request describes a 1.1B dense decoder on about 100B tokens, 540 GPU-hours of pretraining. Nothing found justifies changing the size, the architecture family or the optimizer.
3. **The data mix is the change with the strongest evidence.** Cut Cosmopedia from 22% to about 5%, blend DCLM into the web share, replace the code and math sources. Section 3.
4. **Three recipe changes have medium support:** 1M tokens per step instead of 0.5M, no weight decay on the embedding matrix, a 20% decay to zero. Section 4.
5. **No architecture change is ready to adopt untested.** Two cheap ones deserve a place in a small seeded A/B: Peri-LN and a ReLU-squared MLP. Section 5.
6. **September's headline architecture candidate was a misattribution.** The "Softpick" result came from a different function, and published Softpick loses at 1.8B in its own paper. Section 8.
7. **The schedule risk is the queue, not the compute.** Jobs are capped at 48 hours, and Anvil's H100 partition averaged a 119.5-hour wait over the last 30 days. Section 6.

---

## 2. What is fixed, and what the restart frees

| Fixed by the submitted request | Free to change in a restart |
|---|---|
| About 1.1B parameters, dense decoder (RoPE, RMSNorm, SwiGLU, GQA, QK-norm named) | Data mix, staging, packing |
| About 100B tokens, WSD schedule, then SFT and DPO | Tokenizer (must be retrained anyway) |
| 540 GPU-hours pretraining, 60 post-training, one 4-GPU node | Batch size, learning rate, weight decay, decay length |
| All data public, results open | Norm placement, MLP activation (if an A/B supports it) |

**Size versus tokens.** At this compute a loss-optimal model would be somewhat larger (about 1.5B to 2.5B depending on which Chinchilla fit is used). The predicted cost of staying at 1.1B is 0.004 to 0.02 nats of pretraining loss, under one benchmark point. That does not justify contradicting the request and everything published about Ultra. Stay at 1.1B.

---

## 3. Data mix (highest-confidence change)

### Evidence

| Finding | Scale | Who measured | Status |
|---|---|---|---|
| FineWeb-Edu and DCLM are complementary. FW-Edu wins MMLU, ARC, OpenBookQA; DCLM wins HellaSwag, CommonsenseQA, PIQA. A 60/40 blend keeps most of both (MMLU 37.0 vs 37.5, HellaSwag 62.2 vs 60.1) | 1.7B, 350B tokens per arm | Hugging Face, SmolLM2 paper (arXiv 2502.02737) | verified |
| Same pattern with seeds: DCLM vs FW-Edu, HellaSwag 0.632 vs 0.594, ARC-C 0.377 vs 0.421, CSQA 0.638 vs 0.610 | 1B, 100B tokens, 3 seeds | AI2 DataDecide (arXiv 2504.11393), from its released results | setup verified, per-task numbers read by the lane |
| Cosmopedia alone scores 10.33 against DCLM 13.77 on a 12-task average; as the mix-in partner 12.99 against 14.32 | 1.2B, 21B tokens | Hugging Face (the makers of Cosmopedia), arXiv 2604.13977 | verified |
| SmolLM2-1.7B used Cosmopedia at 4%, in the final stage only | 1.7B, 11T | SmolLM2 paper | verified |
| Nemotron-CC high-quality: average 0.66 vs DCLM 0.57 and FW-Edu 0.56. Most of the gap is MMLU (0.50 vs 0.24 and 0.26, 5-shot) and CSQA, which the authors attribute to its instruction-like synthetic text | 1.7B, 1T tokens | LAION and Juelich, open-sci-ref (arXiv 2509.09009) | verified |
| The same Nemotron synthetic data scored 13.54 against DCLM 13.77 | 1.2B, 21B tokens | Hugging Face, arXiv 2604.13977 | verified |
| nanochat moved from FineWeb-Edu to ClimbMix (built from Nemotron-CC): time to its GPT-2 target fell 27%. Five earlier dataset attempts had failed, including a FinePDFs/DCLM/FW-Edu blend | 0.5B to 0.9B, under 10B tokens | Karpathy, dev/LOG.md | verified |
| Education-filtered code raises MultiPL-E (Python 20.7 to 25.6, C++ 16.7 to 24.8) | 1.7B | SmolLM2 paper | read by the lane |
| `HuggingFaceTB/stack-edu` ships no file contents. `common-pile/stackv2_edu_filtered` does: many languages, open licenses, ungated, 67.8B tokens | n/a | Hugging Face API | verified (gating and files) |
| FineMath 4+ gives about 2x GSM8K over InfiMM-WebMath; OpenWebMath is weaker | 1.7B | SmolLM2 paper | read by the lane |

### Recommended mix (no gated data)

| Source | Now | Recommended, stable phase (first 80%) |
|---|---|---|
| FineWeb-Edu-dedup | 55% | 42% |
| DCLM-baseline (`mlfoundations/dclm-baseline-1.0-parquet`) | 0 | 28% |
| Cosmopedia v2 | 22% | 5% |
| Code | 10% codeparrot (Python only) | 12% `common-pile/stackv2_edu_filtered` |
| Math | 8% OpenWebMath | 8% FineMath 3+ and InfiWebMath 3+, half each |
| Wikipedia | 5% (2023 dump) | 5% `HuggingFaceFW/finewiki` (en) |

Decay phase (last 20%): keep the annealing design already built (40% of each batch from FineMath 4+, InfiWebMath 4+, the best code, chat in ChatML, FineWiki; the rest the stable mix), with the per-source repeat fix from September so chat and the small code sets reach their shares.

Confidence: high that Cosmopedia at 22% is too much and that the code and math swaps help; medium on the exact FW-Edu to DCLM ratio (60/40 is SmolLM2's choice; nanochat's one DCLM-containing blend was slightly worse on its own metric).

### Optional upgrades, each with a catch

- **Nemotron-CC-v2** (high-quality plus synthetic and Diverse-QA) for up to half the web share. Best independent result found, but the repository is gated with manual approval under NVIDIA's data agreement (verified: `gated: manual`). The request told NAIRR the data is public with nothing licensed, and approval could miss the two-week start window. Use only if access is granted in time and the PI is comfortable.
- **FinePhrase** (Hugging Face's structured rewrites of web pages, ungated). The authors report 17.18 against DCLM 13.77, but only at 21B tokens and only measured by themselves. The rewrite lives in a nested field and some rows are empty. A proxy A/B arm at most.
- **ClimbMix**: the clearest practitioner result, but CC BY-NC. Leave out.
- **PleIAs SYNTH**: no independent reproduction; the one independent test scored it lowest (10.03). Leave out.

### Other data decisions

- **Tokenizer: retrain on the new mix, same size (49,152), and commit the file.** The file that trained Ultra was never in the repo and is on the dead server. No evidence supports a different vocabulary size at 1B (SuperBPE is only tested at 8B with a 200k vocabulary).
- **Unique tokens, no repetition** in the stable phase. Every source has more than its share.
- **Packing: store document offsets in the rebuild, decide later.** nanochat's document-aligned best-fit loader is its default but its log reports no measured gain, it discards about 35% of raw tokens (verified), and document-masked attention measured as noise there. Not worth delaying the run.
- **Chat data in pretraining: decay phase only**, single-digit percent, as now. Nobody has measured more at this scale.

---

## 4. Training recipe

| Setting | Sept run | Recommended | Evidence | Confidence |
|---|---|---|---|---|
| Tokens per step | 524,288 | 1,048,576 (about 95k steps) | Three independent estimates put the optimum at 100B tokens at 1M to 2M: the Power Lines fit (about 1.0M), the DeepSeek fit (about 1.9M), nanochat's rule plus its own measurement that 1M beat 0.5M near 0.9B parameters (verified in its log) | medium |
| Learning rate | 3e-4 | 3e-4 to 4e-4; settle with the proxy sweep, default 3e-4 | NorMuon's own 1.1B run (width 2048, 16 layers, 50B tokens) used 4e-4 at 2M tokens per step with linear decay (verified). Its depth rule gives about 3.4e-4 at 22 layers. Square-root scaling of our own 3e-4 gives 4.2e-4 | low to medium |
| Weight decay on 2D hidden weights | 0.1 cautious | keep 0.1 cautious | NorMuon used 0.1. The Power Lines law suggests more, but it was fit on AdamW with a full-run decay; under WSD our integrated decay is already near its optimum (my arithmetic) | low |
| Weight decay on the tied embedding | 0.1 | 0 | NorMuon applies none outside hidden 2D weights (verified); OLMo 2 removed it for stability | medium |
| Decay | cosine to 0 over 15% | linear or 1-sqrt to 0 over 20% | At 1B on 350B tokens, WSD with a 25% linear decay gave the best pretraining loss, 0.016 below full-run cosine and 0.071 below no decay (verified, Yano et al., arXiv 2603.16127, runs per setting not stated) | medium on length, low on shape |
| Warmup, clipping, z-loss, 5 Newton-Schulz steps, momentum 0.95 | as is | as is | No evidence that changing them helps at this scale; Polar Express coefficients measured as no difference in nanochat | n/a |
| Precision | fp16 + loss scaling | bf16 autocast (already the code's default on H100) | | |

Two things to plan for:

- **Keep a permanent full checkpoint at the start of the decay.** The same Yano et al. study found that models pretrained with no decay scored higher after SFT (54.4 against 53.1 at 1B) even though their pretraining loss was worse. It is one paper with unstated seeds, so it does not change the plan, but the WSD branch point makes the test nearly free later: SFT from the decayed model and from a constant-rate continuation.
- **Watch for the bf16 attention failure.** A September 2026 paper (arXiv 2609.34272, abstract verified) shows fused bf16 attention gradients going wrong late in training at 450M: gradient norm up about 1000x after 25B tokens, final loss 0.2 nats worse, no NaNs. Their runs had QK-norm off (per the lane's read of the PDF); QK-norm bounds exactly the quantity that grows, so Ultra is probably protected, but that is untested. Log per-layer attention-logit maxima and gradient norm, alert on a sustained 10x rise, and have an fp32-attention switch for selected layers tested before launch.

Not worth doing: Gram Newton-Schulz (orthogonalization is about 1 to 2% of step time here), fp8 linears (17% faster in nanochat but only about 5% at matched quality), momentum warmup, batch-size ramps, EMA (nanochat: no help).

---

## 5. Architecture

**Keep the current design.** The evidence base is arXiv 2605.20798 (verified): a 1.2B baseline almost identical in shape to Ultra (24 layers, width 2048, 8 KV heads, SwiGLU 5632), about 23B tokens, AdamW, 3 seeds only for the baseline, QK-norm and one other method; everything else single-seed, and the 3B runs have no noise floor at all. The authors recommend no method.

| Change | Verdict | Why |
|---|---|---|
| Peri-LN (a second RMSNorm on each sublayer's output) | A/B first | Authors: 1.5B, 30B tokens, 5 seeds, loss 3.29 to 3.18 and benchmark average 53.7 to 56.6 (verified), but at learning rate 2e-3 where plain pre-norm is at its stability edge and apparently without QK-norm. Independent sandwich-norm result: +0.005 at 1.2B (z = +2.45, single seed), +0.010 at 3B. Used by Gemma 2/3 and OLMo 2. Near-zero cost. Unknown interaction with our 1/sqrt(layer) norm scaling, which targets the same problem |
| ReLU-squared MLP at matched parameters | A/B first | Two independent positives near 1B: 2605.20798 (z = +2.36 single seed, +0.009 at 3B) and nanochat, where SwiGLU lost to ReLU-squared at two sizes on steps, time and FLOPs (verified). One contrary result (SwiGLU 0.008 better at 1B on 88B tokens, in a normalized-transformer setup) |
| QK-norm, attention-output gate, value residual | keep | QK-norm is the best-supported item (z = +2.51 with 3 seeds, second-best at 3B). Gate and value residual measured inside the noise independently; no evidence says drop them |
| Published Softpick | skip | Its own authors (v4, April 2026): at 1.8B on 104B tokens, ARC-Easy 67.2 to 62.0, LAMBADA 49.4 to 43.5, "does not yet scale reliably" (verified) |
| Selective Attention | skip | One single-seed positive, an inconsistent implementation description, no fused kernel |
| Attention plus Gated DeltaNet hybrid | not this run | The only new result at exactly this size and budget (Ai2: bits per byte 0.682 to 0.669 at 1B on 100B tokens, authors only). It changes the model class, needs different kernels, and contradicts the request's description |
| Looped models, nGPT, differential attention, Canon layers, PoPE, multi-token and token-order prediction | skip | Lose at fixed training compute, are tied to another optimizer recipe, or have no independent evidence at 1B |

Shape: 22 layers at width 2048 with head size 128 matches current 1B to 2B releases. No controlled evidence supports changing it.

---

## 6. NAIRR and H100 plan

| Fact (verified at the official pages unless noted) | Purdue Anvil AI | SDSC Expanse AI |
|---|---|---|
| Nodes | 21, each 4x H100 80GB, 96 cores, 1 TB RAM | 34, each 4x H100, 72 cores, 1 TB RAM, 6.4 TB local NVMe |
| Partition, max job length | `ai`, 48 h | `nairr-gpu`, 48 h |
| Limits | 12 GPUs per user, 32 per allocation | 4 running jobs, whole nodes |
| Storage | 25 GB home, 5 TB project, 100 TB scratch purged after 30 idle days | No shared filesystem; node scratch wiped after every job; persistent data in Ceph object storage; 100 GB home |
| Queue | 119.5 h average wait over 30 days (range 0 to 224.5), 1,807 jobs | not published |

Start-Up terms (verified): 3 months, must start within 2 weeks of the award, one-page final report, results open.

Consequences:

- **Anvil is the simpler fit** because of storage. Expanse works but needs the 200 GB token set and every checkpoint staged through object storage each job.
- **The run is a chain of 4 jobs of about 47.5 h.** The trainer already checkpoints every 15 minutes and exits cleanly on SIGTERM; it needs a Slurm script that sends the signal 10 minutes early and resubmits.
- **Calendar time could be 2 to 4 weeks for 6 days of compute** if each link waits days. Submit the first link the day the account works. Shorter jobs backfill faster, so keep the A/B and data jobs small.
- **Throughput estimate (lane arithmetic, not measured): about 190k tokens/s on 4 GPUs, range 150k to 240k.** That is 6.1 days and about 585 GPU-hours for 100B tokens (range 465 to 740) against 540 planned. Measure in the first hour. If it comes in low, WSD lets the total be trimmed before the decay starts.
- **Data build:** set `HF_TOKEN` (anonymous limits are per IP and a cluster shares one), download the parquet files to scratch, tokenize from local disk (about 11 h on 96 cores). Whether a zero-GPU job is allowed on `ai` is unknown; budget it as a 1-GPU job.
- **Unknown until login:** whether compute nodes reach huggingface.co, installed PyTorch and CUDA versions, whether several processes per GPU are allowed (matters for Planck), and the under-18 account question.

---

## 7. One small A/B block before the long run

Purpose: decide the items the literature cannot. Budget about 30 GPU-hours from the reserve, as short single-GPU jobs that backfill.

- **Proxy:** Ultra's depth (22 layers) at a narrow width, about 120M parameters, about 3B tokens per run, 1M tokens per step, same data pipeline. Depth matters because Peri-LN and the norm scaling are depth fixes.
- **Noise floor first:** 3 seeds of the baseline. Published seed noise at this size is about 0.001 to 0.002 nats.
- **Arms, one run each, repeat any winner twice:** learning rate at 2e-4 and 4.5e-4; Peri-LN with and without the norm scaling; ReLU-squared MLP; the old data mix against the new one.
- **Decision rule, fixed in advance:** adopt a change only if it beats the baseline mean by more than 2 standard deviations in held-out loss and is not worse on the benchmark suite. Otherwise keep the current design.
- **Caution:** nanochat found settings tuned on its smallest model "actively hurt" at a larger one. Use the proxy to screen for clear effects, not to fine-tune values.

If the queue makes even this too slow, skip it and run the current architecture with the section 3 and 4 changes, which rest on published evidence at about 1B.

---

## 8. Corrections to the 22 September report

1. **Softpick.** That report called Softpick the best-evidenced architecture change, blocked only by the old GPUs. The cited study tested ReLU(softmax - 1/n) (its Appendix B.2, verified), which is not the published Softpick, ReLU(e^x - 1) divided by the sum of |e^x - 1|. Published Softpick loses at 1.8B in its own paper.
2. **Three claims were attributed to arXiv 2605.20798 that are not in it** (verified): multi-token prediction below 1.3B, LayerNorm scaling, tied versus untied embeddings. They need other sources before being repeated.
3. **"Your mix already matches what SmolLM2 and OLMo converged on" was wrong.** SmolLM2 used a FineWeb-Edu plus DCLM web blend and only 4% Cosmopedia, late.

---

## 9. What to expect from the base model

Zero-shot, before SFT (lane estimate from the anchors below): HellaSwag 55 to 62, PIQA 72 to 76, WinoGrande 56 to 61, ARC-Challenge 35 to 43, LAMBADA 45 to 58, MMLU near chance in multiple-choice format.

Anchors: Ai2's 1B models at exactly 100B tokens (DCLM recipe: HellaSwag 0.632, ARC-C 0.377, PIQA 0.752, WinoGrande 0.616, 3 seeds, probably few-shot); the 1.2B baseline at 23B tokens in 2605.20798 (HellaSwag 0.559, ARC-C 0.393); Llama 3.2 1B (HellaSwag 61.2). Our own dry run at 4.2B tokens in September (PIQA 70, WinoGrande 55, ARC-C 37.5, HellaSwag 45 on 200 examples) is on that trajectory.

For comparable numbers at the end: score the full test sets rather than 1,000-example subsets (the subsets carry about 3 points of noise), add the rest of the 12-task CLIMB average, and re-score two public checkpoints in the same harness.

Post-training: nothing published since 22 September changes the SFT and DPO plan.

---

## 10. Not verified, or not found

- Any seeded head-to-head at 1B or larger that beats NorMuon, or shows it failing: none found. NorMuon's own 1.1B numbers have no stated seeds.
- A measured bf16 versus fp32 quality comparison for a 1B dense run with QK-norm: none found.
- A controlled comparison of several data stages against one mix plus a decay-phase blend at this scale: none found.
- Whether FinePhrase's gain holds beyond 21B tokens; whether Peri-LN helps on top of QK-norm and Muon.
- Throughput on these systems, compute-node internet access, per-GPU process sharing: unknown until login.
- The size-versus-tokens gap and the weight-decay arithmetic are derivations, not measurements.

---

## 11. Build list before the award arrives

The two-week start clock runs from the award, so these should exist first. Roughly in order:

1. New pretraining mix in `data/prepare.py` (DCLM, FineMath, FineWiki, the Common Pile code set with its `json.gz` files and language metadata), tokenizer retrain path, document offsets stored per shard, download-then-tokenize mode.
2. `configs/ultra_h100.yaml`: 1M tokens per step, micro-batch found by the autotuner, embedding weight decay 0, 20% decay, bf16.
3. Slurm scripts: 4-link chain with early signal and resubmit, data-build job, throughput benchmark.
4. Telemetry and the safety switch: per-layer attention-logit maxima, gradient-norm alert, fp32 attention for selected layers, permanent checkpoint at the decay start.
5. Proxy config (22 layers, narrow) and the A/B arms: Peri-LN and ReLU-squared as options behind flags, with tests.
6. Full-test-set evaluation mode and the CLIMB task additions.
