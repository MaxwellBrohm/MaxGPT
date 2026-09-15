# MaxGPT-Ultra on the school Lambda server

`lambda.ccm.edu` is a shared Lambda box on the campus network: **10x NVIDIA TITAN RTX (24GB,
Turing)**, 64 cores, 503GB RAM, 3.3TB free disk, Ubuntu 20.04, driver 470 / CUDA 11.4 (the
cu118 torch build), internet access, tmux, no sudo. It was idle when we got it. Everything we
install lives in `$HOME`.

## Access

- On campus wifi from the Mac: `ssh lambda` (key-based; `Host lambda` in `~/.ssh/config`).
- From home: Omnissa Horizon Client -> `view.ccm.edu` -> ITLabs desktop -> PowerShell ->
  `ssh brohm.maxwell@lambda.ccm.edu`. Lambda has no public address; the VDI is the only door.
- The account's default password was "CCM" + student ID (uppercase). Changed at first login.

## What is set up (Sep 15 2026)

- `~/MaxGPT` = this repo; `~/venv` = Python 3.12 (uv) + torch 2.7.1+cu118 + deps, built by
  `scripts/lambda_setup.sh` (re-runnable). Smoke tests there: fp16 autocast, compile,
  efficient attention, paged 8-bit AdamW all OK.
- The 100B-token data build runs in tmux session `data` (`tmux attach -t data`, Ctrl-B D to
  leave). Log: `~/MaxGPT/data_build.log`; progress: `data/shards/progress.json` (resumable:
  if the box reboots, run the same command again and it continues from the last shard).
  Speed ~1.8M tok/s on 64 cores -> ~16 hours for all 100B.
- `configs/ultra_lambda.yaml` extends `configs/ultra.yaml` (same 1.1B model, so checkpoints
  are interchangeable with home) with the machine knobs: `precision: auto` (-> fp16 + loss
  scaling, Turing has no bf16), `gpus: auto` (every card in `CUDA_VISIBLE_DEVICES`),
  `grad_checkpointing: false` (24GB keeps the activations), `micro_batch` / `grad_accum`.

## Driver quirks handled in code (so nobody has to remember them)

- `torch.cuda.is_bf16_supported()` says True on Turing via slow emulation. The trainer asks
  for native bf16 only, so `precision: auto` resolves to fp16 + loss scaling there.
- Triton (torch.compile) builds kernels with a CUDA 12 ptxas that driver 470 cannot load
  ("device kernel image is invalid"). `cfg.configure_triton_ptxas()` points Triton at the
  CUDA 11.8 ptxas (`nvidia-cuda-nvcc-cu11`, installed by the bootstrap) on drivers < 525.
- Under DDP with gradient accumulation, PyTorch keeps a second full gradient copy unless
  the grads live in the all-reduce buckets; the trainer binds them there (see TECHNIQUES.md),
  which is what made the 1.1B fit 2 GPUs at micro_batch 2 without checkpointing.

## Sharing the box

- `nvidia-smi` first, every time; the process table shows everyone's jobs.
- Pin runs with `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` (leave two cards free unless agreed).
- Everything runs inside tmux (`tmux new -s ultra`, `tmux attach -t ultra`) so a dropped VDI
  or wifi never kills a run. The trainer also checkpoints every 15 minutes and resumes.
- `pkill -f scripts/train.py` over ssh kills your own ssh shell (the pattern matches its
  own command line); use `pkill -f "[s]cripts/train.py"`.

## Running Ultra

```
tmux new -s ultra
source ~/venv/bin/activate && cd ~/MaxGPT/maxgpt-ultra
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python scripts/find_micro_batch.py --config configs/ultra_lambda.yaml --compile   # once; set micro_batch/grad_accum from its answer
python gui/server.py --config configs/ultra_lambda.yaml                           # then press play
```

Dashboard from the Mac on campus: `ssh -L 8800:localhost:8800 lambda` in a second terminal,
then http://localhost:8800. From the VDI: the same `ssh -L ...` in a second PowerShell and
the VDI's browser. Pause/play/stop work as at home; the stages launch with torchrun on every
visible card, rank 0's output is the dashboard log, every rank's log lands in
`runs/<stage>/ranks/`.

Without the GUI: `torchrun --nnodes=1 --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:0
--nproc_per_node=8 scripts/train.py --config configs/ultra_lambda.yaml --data data/shards
--out runs/pretrain --tokenizer tokenizer/maxgpt-ultra.tokenizer.json --eval-data data/shards
--eval-every 500 --stop-file runs/pretrain/STOP` (touch the STOP file to pause).

## The A/B before the real run (`configs/ab/`)

Four 124M variants of the shakedown recipe on the same 1.1B tokens (shards 0-10, held-out
shard 11), 2 GPUs each: `adamw` (our recipe), `normuon` (NorMuon + cautious weight decay),
`adamw_arch` (attention gate + value residual + norm scaling), `normuon_arch` (both).
`python scripts/ab_report.py` prints a table; the decision is the held-out loss, averaged
over the last evals. Only a clear win moves into `configs/ultra.yaml`.

## Home PC

Stays paused while the video-editing model needs the 5070. The two checkouts share the same
code; the Lambda run is the primary one now.
