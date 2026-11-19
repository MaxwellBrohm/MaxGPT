# Running MaxGPT-Ultra on the NAIRR H100 allocation

The November 2026 restart: Ultra trained from scratch on one 4x H100 80GB node. Why each choice was
made is in `docs/research_2026-11-18_restart.md`; this file is the procedure.

**Status: written before the award. Nothing here has run on the cluster.** The code is tested on
CPU (see "Tests"), the datasets were opened live from a laptop, and the system facts were read from
the official pages on 2026-11-18. Throughput, queue behaviour and the points under "Unknown until
login" are estimates until the first hour on the machine.

## The plan in one table

| Step | Command | GPU-hours (estimate) |
|---|---|---|
| 1. First hour: environment, checks | see below | under 1 |
| 2. Proxy data (3.3B tokens) | `STAGE=proxy sbatch -t 08:00:00 slurm/data_build.sbatch` | 2 to 5 (mostly the 500 GB download) |
| 3. Proxy A/B, 8 runs | `sbatch slurm/proxy_ab.sbatch` | about 30 |
| 4. Full data (100B + 8B tokens) | `sbatch slurm/data_build.sbatch` | 12 to 24, billed as a 1-GPU job |
| 5. Apply the A/B verdict | edit `configs/ultra_h100.yaml` | 0 |
| 6a. Launch test: the real model for an hour | see step 6 | about 5 |
| 6b. Pretraining, 5 queued links | `sbatch slurm/pretrain_chain.sbatch` (x5) | about 585 (range 465 to 740) |
| 7. Post-training | `RUNBOOK.md`, Phase G | about 60 |

Steps 3 and 4 run at the same time, and the launch test (6a) can run as soon as step 2 is done. Both
systems cap a job at 48 hours, and Anvil's H100 queue averaged a 119.5-hour wait in the 30 days to
2026-11-18, so six days of compute may take several weeks. Submit early, and keep everything except
the pretraining links small so it backfills.

## System facts (official pages, 2026-11-18)

| | Purdue Anvil AI | SDSC Expanse AI |
|---|---|---|
| Nodes | 21, each 4x H100 80GB, 96 cores, 1 TB RAM | 34, each 4x H100, 72 cores, 1 TB RAM, 6.4 TB local NVMe |
| Partition, job limit | `ai`, 48 h | `nairr-gpu` (whole nodes) and `nairr-gpu-shared`, 48 h |
| Limits | 12 GPUs per user, 32 per allocation | 4 running jobs on `nairr-gpu` |
| Charging | 1 SU = 1 GPU for 1 hour | GPUs x hours |
| Storage | 25 GB home; 5 TB project (kept while the allocation is active); 100 TB scratch, purged 30 days after last access | 100 GB home; no shared filesystem; node scratch wiped after every job; Ceph object storage for anything that must persist |

Start-Up terms: 3 months, first use within 2 weeks of the award, a one-page final report, open results.

**Anvil is the simple case and this procedure is written for it.** Clone the repository into the
project space and keep the default paths (everything under `data/` and `runs/`, about 1.2 TB in all:
500 GB of downloads, 220 GB of shards, 60 GB of rolling checkpoints, about 300 GB of milestones).
If the big files must live somewhere else, set `MAXGPT_DATA` (and `MAXGPT_SOURCES_ROOT`) in
`slurm/env.sh`: the data build writes every shard folder there and the training job reads all three
from there, the decay-phase shards included.
On Expanse AI the token shards and the latest checkpoint would have to be copied in from Ceph at the
start of every job and checkpoints copied out as they are written. That staging step is NOT written.
If the award lands on Expanse, write and test it before step 6.

## 1. First hour

```bash
ssh x-<username>@anvil.rcac.purdue.edu
cd $PROJECT && git clone https://github.com/MaxwellBrohm/MaxGPT.git && cd MaxGPT/maxgpt-ultra
cp slurm/env.example.sh slurm/env.sh        # then edit it: module loads, virtualenv
echo 'export HF_TOKEN=<your token>' >> ~/.bashrc && source ~/.bashrc
mkdir -p logs
```

Environment (any Python 3.10+ with PyTorch 2.x for CUDA 12 works; use whatever the site documents):

```bash
python -m venv $PROJECT/maxgpt-venv && source $PROJECT/maxgpt-venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Checks, in this order. Each answers one of the unknowns.

1. `python scripts/smoke_test.py && python scripts/test_restart.py && bash scripts/test_slurm_chain.sh && python scripts/test_ddp.py`
   on a login node (CPU, a few minutes): the code works in this environment. On Linux the last one
   starts two processes with the same `torch.distributed.run` line the job script uses, so it also
   proves the launcher on this site's PyTorch.
2. Inside a short interactive GPU job (`sinteractive -p ai -A <allocation> --gpus-per-node=1 -t 00:30:00`
   or the site's equivalent):
   - `python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"`
   - `curl -sI https://huggingface.co | head -1`: do compute nodes reach the internet? If not, run every
     `fetch_sources.py` line of `slurm/data_build.sbatch` on a login node first and wait for each to print
     `[fetch] done` before submitting the data job (never two downloads at once into the same folder);
     the build itself then needs no network except for the SFT and DPO sets (run `prepare_data.py`
     without `--skip-posttrain` where there is internet, or copy `data/sft.jsonl` and `data/prefs.jsonl` in).
   - `python scripts/probe_sources.py --mix pretrain --rows 10`: every dataset opens with the right fields.
3. After the first short job, `jobsu <jobid>` (Anvil): how a 1-GPU job with many cores is billed. If it
   bills more than 1 SU per hour, lower `--cpus-per-task` in `slurm/data_build.sbatch`.
4. Ask the site whether several processes may share one GPU (matters for the small Planck jobs, not for Ultra).

## 2 to 4. Data

```bash
STAGE=proxy sbatch -A <allocation> -t 08:00:00 slurm/data_build.sbatch   # sources + tokenizer + 3.3B-token proxy shards
sbatch -A <allocation> slurm/data_build.sbatch                           # the rest; safe to submit right after
```

The second job waits for the first: the script carries `--dependency=singleton`, so two data jobs
never run at once (they write the same downloads, tokenizer and shards).

What it does, each step skipped if its output exists and resumed if interrupted:

- downloads each source's share of files to `data/sources` (500.6 GB planned; `--dry-run` prints the plan).
  When a source is completely down, the script leaves a record of its files in that source's folder.
  The build reads exactly the recorded files and stops with the file's name if one is missing or cut
  short, or if the folder has files but no record (the download did not finish). The fix is always
  to run the same `fetch_sources.py` line again: it keeps the files that are whole and fetches the rest;
- trains the 49,152-token tokenizer on the new mix if `tokenizer/maxgpt-ultra.tokenizer.json` is missing.
  **Commit that file** (`git add -f` if needed) and copy it home: the September tokenizer was lost
  because it lived only on the old server;
- builds `data/shards` (100B tokens, about 11 hours on 96 cores), `data/sft.jsonl`, `data/prefs.jsonl`,
  holds out one shard as `data/shards_val_ultra`, and builds `data/shards_anneal` (8B tokens);
- prints each source's realized token share against its target and warns when one is more than
  2 points off or ran dry. Read that table before training on the shards.

The mix (`data/prepare.py`, `PRETRAIN_MIX`): FineWeb-Edu 42%, DCLM-baseline 28%, Stack v2 education
code 12%, FineMath 3+ 4%, InfiWebMath 3+ 4%, FineWiki 5%, Cosmopedia v2 5%.

If access to the gated `nvidia/Nemotron-CC-v2` is granted in time: run
`python scripts/probe_sources.py --mix pretrain-nemotron`, correct the config and field names in
`PRETRAIN_MIX_NEMOTRON` if the probe says so (they are unverified), and build with
`--mix pretrain-nemotron`. Decide before step 4; the two mixes cannot be switched mid-run.

That set comes under the NVIDIA Data Agreement for Model Training (read 2026-11-18, version of
August 15, 2025). It allows training and leaves the trained model with whoever trained it, but the
data itself may not be passed on. In practice: `chmod 700` its folder under `data/sources` and any
shards built from a mix that contains it, never upload those shards anywhere, delete them if the
agreement ends, and name the set and its share in the final report.

The model can still be published. The dataset card states that the agreement covers models released
under proprietary or open source licenses, and the ownership clause outlives the agreement. The card
also warns that a released model may fall under the Qwen and DeepSeek license terms because of
model-written text in this dataset family; its own table for Nemotron-CC-v2 lists two writers,
Mistral-Nemo-12B-Instruct and Qwen3-30B-A3B, both Apache-2.0 (checked 2026-11-18). Re-read the card
before a release, and list the training data in the model card.

## 3 and 5. Proxy A/B and its verdict

```bash
sbatch -A <allocation> slurm/proxy_ab.sbatch          # 8 single-GPU jobs, about 3 GPU-hours each
python scripts/proxy_verdict.py --runs runs/proxy --out runs/proxy/verdict.json
```

Three baseline seeds give the noise floor; the other runs change one thing each. The rule is fixed in
`scripts/proxy_verdict.py`: adopt only beyond two noise floors of held-out loss with the benchmark
average intact. What to do with each verdict, in `configs/ultra_h100.yaml`:

| Arm adopted | Change |
|---|---|
| `lr_4.5e-4` | `lr: 4.0e-4` (the proxy screens direction; 4e-4 is the top of the range the evidence supports) |
| `lr_2e-4` | `lr: 2.5e-4` |
| `peri_ln` | `model.peri_ln: true` |
| `peri_ln_noscale` and it beats `peri_ln` | `model.peri_ln: true`, `model.norm_scaling: false` |
| `relu2` | `model.mlp: relu2`, `model.mlp_hidden: 8448` |
| nothing | keep the file as it is |

The old-data arm is optional (`--array=8`, after building shards with `--mix pretrain-2026-09` into
`data/shards_proxy_old` and holding one out as `data/shards_proxy_old_val`). It is a check on the new
mix, judged on benchmarks only.

If the queue makes the A/B too slow to wait for, skip it and run the config as committed. Do not
tune further on the proxy: settings tuned on a small model have been measured to hurt a larger one.

## 6. Pretraining

### 6a. First, a one-hour launch test

The first link may wait days in the queue. A mistake that only shows on the real hardware (the
environment, the four-GPU launch, compiling the model, the evaluation code, the handover signal)
would end all five links within minutes of each other and send the run to the back of the queue.
So run the same job script once, briefly, on the proxy shards. It needs only step 2:

```bash
MAXGPT_RUN=runs/launch_test MAXGPT_TRAIN_SHARDS=data/shards_proxy_train MAXGPT_VAL_SHARDS=data/shards_proxy_val \
MAXGPT_EVAL_EVERY=50 sbatch -A <allocation> -J maxgpt-launchtest -t 01:00:00 slurm/pretrain_chain.sbatch
MAXGPT_RUN=runs/launch_test bash slurm/status.sh          # when it has ended
```

(If `MAXGPT_DATA` points somewhere else, the two proxy folders are there.) It is the full 1.1B model
with the real config, so it also gives the real speed. Check, in `logs/maxgpt-launchtest-<job>.out`
and the status output:

1. the line `[train] device=cuda gpus=4 params=1092.6M total_steps=95367 tokens/step=1048576`;
2. tokens per second. The plan assumes about 190,000; see "Numbers to know" for what to do if it is lower;
3. an evaluation and a benchmark score, both from step 50. In the real run these first happen
   45 minutes in, at step 500;
4. about 45 minutes in: "time limit near: asking the trainer to checkpoint and stop", then "link ended
   cleanly before the end of training", and a checkpoint in `runs/launch_test/checkpoints/`. That is
   the 48-hour handover, working on this system;
5. submit the same line once more with `-t 00:25:00`: the log must say `resumed=True (from step N)`.

Then delete `runs/launch_test` and queue the chain. A link started this way never touches
`runs/pretrain`. If step 4 has not finished yet, the test prints that the anneal shards were not
found; that is expected here and matters only for the real run.

### 6b. The chain

```bash
python scripts/find_micro_batch.py --config configs/ultra_h100.yaml --compile   # optional: can micro_batch be 16?
for i in 1 2 3 4 5; do sbatch -A <allocation> slurm/pretrain_chain.sbatch; done
bash slurm/status.sh
```

If `micro_batch` changes, change `grad_accum` so that `micro_batch x grad_accum x 2048` stays
1,048,576 (16 with 32). Never change tokens per step after the first step: the schedule counts steps.

How the chain behaves:

- only one link runs at a time (`--dependency=singleton`); each resumes from the newest checkpoint;
- 15 minutes before its time limit a link asks the trainer to checkpoint and exit; the next link continues;
- a link that finds `runs/pretrain/DONE` exits in seconds, so extra links cost nothing;
- a link that finds `GRAD_HALT` or `ANNEAL_MISSING` refuses to train (exit code 3) until a person has acted.

Numbers to know: 95,367 steps in all; the decay starts at step 76,293; a milestone checkpoint every
4,768 steps (5B tokens); evaluation every 500 steps; the benchmark suite at the first evaluation and
then at every 10,000th step.

The launch test gives `tok_per_s`, and `runs/pretrain/metrics.jsonl` confirms it in the first hour of
the first link. The plan assumes about 190,000 (range 150,000 to 240,000). 100B tokens then cost about
585 GPU-hours against 540 budgeted. If the real number is below 150,000, lower `total_tokens` before
step 76,293; the schedule adapts as long as it is changed before the decay begins and never after.

The decay-phase shards (`shards_anneal`, beside the training shards) must exist before step 76,293.
If they do not, the trainer stops there and writes `ANNEAL_MISSING` instead of running the one-shot
decay phase without its data. Each link's log names the three shard folders it was given, and the
trainer prints `annealing: <folder> blended in at 40%` at start once it has found them.

### If the gradient watch speaks

Fused bf16 attention has been reported to corrupt gradients late in training with no NaN
(arXiv 2609.34272, in a model without QK-norm; Ultra has QK-norm, so this is insurance).

- `grad_alert` rows in `metrics.jsonl` (a sustained 10x rise): look at `act_max_logit` in the eval
  rows. If one layer's maximum is climbing by orders of magnitude, act as below without waiting.
- `GRAD_HALT` (a sustained 100x rise; the run has stopped):
  1. find the last eval row where gradient norm and `act_max_logit` were still normal, and the newest
     `checkpoints/milestone_*.pt` at or before it;
  2. roll back: copy that milestone to `checkpoints/ckpt_<step>.pt` and point `checkpoints/latest.json`
     at it (`{"path": "ckpt_<step>.pt", "step": <step>}`);
  3. list the affected layers (or all 22) under `model.attn_fp32_layers` in the config. fp32 attention
     is slower: all layers cost roughly a third more time per step;
  4. delete `GRAD_HALT` and submit more links.

## 7. After pretraining

`RUNBOOK.md`, Phase G: score the base model on the suite, SFT, the DPO beta sweep, pick by
`scripts/eval_compare.py`. Two additions from the November research:

- `checkpoints/decay_start.pt` is kept on purpose. One study found models pretrained without a decay
  did better after SFT. If hours remain, continue from it at a constant rate for the last 20B tokens,
  run SFT on both, and compare.
- For final numbers fetch the full test sets (`python scripts/eval_suite.py --fetch --n 0 --suite-dir data/bench_full`)
  and score with those; the 1,000-example sets used during training carry about 3 points of noise.

Bring home: the final and `decay_start` checkpoints (about 15 GB each), `runs/*/metrics.jsonl`,
the tokenizer file, and `data/shards*/meta.json`.

## If hours run short

In order: skip the proxy A/B (30), skip the old-data arm, stop Ultra at 50B tokens by setting
`total_tokens: 5.0e10` before the decay would begin (the request's own fallback, saves about 290).

## Unknown until login

Compute-node internet access; installed PyTorch and CUDA versions; how CPU cores on a 1-GPU job are
billed; whether a zero-GPU job is allowed on `ai`; whether processes may share a GPU. Measured
throughput replaces every estimate above.

One more, worth asking the help desk on day one: does a job that is waiting on a dependency gain
queue priority while it waits? On a default Slurm setup it does not, so each of the five links would
start its queue wait only when the link before it ends, and five waits of the size measured in
November add up to weeks. If that is how the site works, ask whether they have a recommended way to
run a chain of 48-hour jobs.

## Tests

`scripts/test_restart.py` (the restart features and the `--anneal-data` option), `scripts/test_data.py`
(mix, interleaving, continuation, download records), `scripts/test_slurm_chain.sh` (the chain script,
checked under bash 3.2 and 5.3), `scripts/test_proxy_verdict.py` (the decision rule), plus the existing
suites. Every assertion was checked by breaking the code on purpose and watching the test fail.
One real file was downloaded from the hub through `fetch_sources.py` on a laptop to prove the download,
record and read path; the full 500 GB download has not been run anywhere yet.
