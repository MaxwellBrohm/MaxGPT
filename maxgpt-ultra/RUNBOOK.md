# MaxGPT-Ultra: training the 1.1B in WSL2

The ordered steps to set up WSL2 and launch the real ~1.1B Ultra run.

**Why WSL2:** the 1.1B does not fit in 12 GB on native Windows, the Linux-only memory features
(paged optimizer + expandable_segments) don't work there. In WSL2 they do, so the full 1.1B fits
with headroom. Bonus: on Linux, torch.compile and FlashAttention-2 work with no MSVC / triton-windows
hassle.

**Machines:** Mac = dev (you push code from there). WSL2 (Ubuntu on the PC) = training. Unless noted,
commands run in the Ubuntu terminal from `~/MaxGPT/maxgpt-ultra` with the venv active.

---

## Phase A — WSL2 + GPU (one-time, mostly done)
- WSL is **version 2**: in PowerShell `wsl -l -v` shows VERSION `2` (if `1`: `wsl --set-version Ubuntu 2`).
- `wsl --update` to get current GPU/CUDA passthrough.
- GPU visible in Ubuntu: `nvidia-smi` shows the 5070 at ~12227 MiB.
- `C:\Users\<you>\.wslconfig` has e.g. `memory=24GB`, `swap=8GB`, `vmIdleTimeout=-1` (system RAM, not VRAM;
  the full 12 GB VRAM is always available, WSL2 doesn't cap it).

## Phase B — Project + deps in WSL2
```bash
sudo apt update && sudo apt install -y python3-venv python3-pip git
cd ~ && git clone https://github.com/MaxwellBrohm/MaxGPT.git && cd MaxGPT
python3 -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r maxgpt-ultra/requirements.txt
pip install datasets bitsandbytes
```
Clone into `~` (Linux filesystem), **NOT** `/mnt/c/...` (that path is slow and would cripple the
100B-token data build). Verify (all three must succeed):
```bash
python -c "import torch; print('cuda', torch.cuda.is_available())"
python -c "import bitsandbytes as bnb; print('paged', hasattr(bnb.optim,'PagedAdamW8bit'))"
python -c "import torch; m=torch.nn.Linear(8,8).cuda(); print(torch.compile(m)(torch.randn(4,8,device='cuda')).shape)"
```
Want: `cuda True`, `paged True`, and a shape (compile just works on Linux, no MSVC).

## Phase C — Storage check
The 100B shards (~200 GB) live in the WSL2 filesystem, which sits on your **C: drive**. Make sure C:
has **~250 GB free**. If not, relocate the WSL2 disk to a bigger drive first.

## Phase D — Autotune micro_batch
```bash
cd ~/MaxGPT/maxgpt-ultra
python scripts/find_micro_batch.py --config configs/ultra.yaml --compile
```
With the paged optimizer + expandable_segments working, the 1.1B fits with headroom. Note the
recommended `micro_batch` and `grad_accum`.

## Phase E — Set the config
Edit `configs/ultra.yaml` with the values from Phase D:
```yaml
  micro_batch: <from autotuner>
  grad_accum:  <from autotuner>
```
(`total_tokens: 1.0e11` stays, that's the 100B unique-token target, one epoch.)

## Phase F — Run Ultra
```bash
source ~/MaxGPT/venv/bin/activate    # if not already active
cd ~/MaxGPT/maxgpt-ultra
python gui/server.py --config configs/ultra.yaml
```
Open `http://localhost:8800` in your **Windows browser** (WSL2 forwards localhost) and press **play**.
Stages: **data** (retrains tokenizer, tokenizes the 100B mix, builds SFT+OASST + prefs) -> **pretrain**
-> **sft** -> **dpo**.
- The **data stage is multi-day and RESUMABLE**: if it's interrupted (crash, reboot, closing the GUI),
  just press play again and it resumes from the last shard, no re-downloading. It's only "done" once
  `data/shards/meta.json` exists.
- When **pretrain** first starts, torch.compile warms up for a few minutes before steps appear.

## Phase G — When Ultra finishes (weeks out)

Post-training, in the order the research report (docs/research_2026-09-22.md section 3) argues
for. Every step below is built and tested; none has run on the real model yet.

1. **Score the base** on the fixed suite (the first of three evaluation points):
   `python scripts/eval_suite.py --config configs/ultra_lambda_final.yaml --checkpoint runs/pretrain/checkpoints/<final>.pt --out runs/eval/base.json`
2. **Data** (network, once): `python scripts/prepare_data.py --config configs/ultra.yaml` builds
   `data/sft.jsonl` (SmolTalk2 no-think + OpenAssistant, decontaminated against `data/bench`)
   and `data/prefs.jsonl` (UltraMix, reward-filtered), per `posttrain:` in the config. The
   pretraining shards are reused as-is.
3. **SFT**: the dashboard's SFT stage, or by hand
   `python scripts/sft.py --config configs/ultra_lambda_final.yaml --init <final ckpt> --tokenizer tokenizer/maxgpt-ultra.tokenizer.json --data data/sft.jsonl --replay-shards data/shards_train --out runs/sft`
   (document mask on, 10% replay, the pretraining optimizer; add `--optimizer adamw` for the
   comparison run). Score it: evaluation point two. Keep this checkpoint whatever DPO does.
4. **DPO beta sweep**: `bash scripts/dpo_sweep.sh runs/sft/checkpoints/<ckpt>.pt` runs 0.1 / 0.3 /
   0.5 with one shared reference cache and scores each; pick by the suite average and the
   question-level paired differences, never by preference win-rate, and watch
   `sample_mean_tokens` in the eval rows (a sharp rise = length exploitation). Evaluation point
   three. If every beta loses to SFT on the suite, ship SFT.
5. **Optional soup** (report 3.3): interpolate the DPO weights toward SFT, sweep the coefficient
   on a held-out slice; CPU only.
6. `python scripts/save_model.py --run runs/dpo --tokenizer tokenizer/maxgpt-ultra.tokenizer.json --name ultra --config configs/ultra.yaml`
   creates `models/ultra.pt` + `models/ultra.tokenizer.json` for the webui.

---

## Notes

- **Batch semantics after the multi-GPU change:** `micro_batch` is per GPU; `grad_accum` is the
  TOTAL number of micro-steps per optimizer step across all GPUs (must divide by the GPU
  count). tokens/step = micro_batch x grad_accum x seq_len on 1 GPU or 8, so the LR schedule
  never changes with the hardware. `precision: auto` = bf16 on the 5070, fp16 + loss scaling
  on cards without native bf16. `gpus: auto` makes the GUI launch every visible card with
  torchrun. For the school server see LAMBDA.md.
- **Mini** lives on your old Windows checkout (from the shakedown). Nothing to wipe, Ultra trains
  fresh in WSL2. To show Mini in the webui later, run `save_model --name mini` on the Windows side and
  we'll wire it up then.
- **WSL2 gotchas:** work in `~`, never `/mnt/c` (slow). Full 12 GB VRAM is available (no cap). compile +
  FlashAttention-2 work natively (no MSVC, no triton-windows, no `cl.exe`).
- **100B unique data** is intentional (over-trained recipe). The mix holds near 55/22/10/8/5; the cost
  is the download (~hundreds of GB), not disk.
- **Resumable data:** `tokenize_to_shards` checkpoints `data/shards/progress.json` at every shard
  boundary and auto-resumes; it's deleted once `meta.json` is written.

## Path reference (in WSL2)
| What | Path |
|---|---|
| Repo | `~/MaxGPT` |
| Configs | `maxgpt-ultra/configs/{shakedown,ultra}.yaml` |
| Pretrain shards | `maxgpt-ultra/data/shards/` (+ `meta.json`, `progress.json` while building) |
| SFT / DPO data | `maxgpt-ultra/data/{sft,prefs}.jsonl` |
| Run checkpoints | `maxgpt-ultra/runs/{pretrain,sft,dpo}/checkpoints/` |
| Models (webui) | `maxgpt-ultra/models/{mini,ultra}.{pt,tokenizer.json}` |
