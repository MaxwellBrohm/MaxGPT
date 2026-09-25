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

**Reading the val curve.** `val_loss` / `val_ppl` in `runs/pretrain/metrics.jsonl` is the mean
cross-entropy over a fixed 655k-token slice of `data/shards_val_ultra` (40 batches of 8 x 2048),
so successive points are comparable. Before step ~6,500 (2026-09-24) each eval read the NEXT
327k tokens of the held-out set instead, and slices differ by up to 0.4 nats, so the early
points bounce (8.7 ppl at step 5,000, 13.2 at 5,500) without the model changing that much:
read the training loss for the trend there, and expect one level shift where the fix landed.

**Benchmark suite.** `data/bench/` holds the fixed suite (LAMBADA, PIQA, WinoGrande, ARC-C,
HellaSwag; 1,000 seeded examples each; `suite.json` lists n + source). The run scores it
every 10,000 steps inside the periodic eval and logs it as `"suite"` in the eval row (and
`suite_avg`); expect that eval to take ~3 extra minutes. To score any checkpoint by hand on a
free card:
`CUDA_VISIBLE_DEVICES=1 python scripts/eval_suite.py --config configs/ultra_lambda_final.yaml --checkpoint runs/pretrain/checkpoints/<ckpt>.pt --device cuda --out runs/eval/suite_<step>.json`.
Re-fetch the suite only on purpose (`python scripts/eval_suite.py --fetch --seed 1 --suite-dir data/bench_seed1`
for a noise-floor repeat); the numbers are only comparable across evaluations of the same files.

## The A/B before the real run (`configs/ab/`)

Four 124M variants of the shakedown recipe on the same 1.1B tokens (shards 0-10, held-out
shard 11), 2 GPUs each: `adamw` (our recipe), `normuon` (NorMuon + cautious weight decay),
`adamw_arch` (attention gate + value residual + norm scaling), `normuon_arch` (both).
`python scripts/ab_report.py` prints a table; the decision is the held-out loss, averaged
over the last evals. Only a clear win moves into `configs/ultra.yaml`.

## Card health and the run layout (measured Sep 21 2026)

`scripts/brake_probe.py` burned each card alone for 75 s (box otherwise idle):

| card | time to 80 C | max | clocks | verdict |
|---|---|---|---|---|
| 0, 1, 2, 3, 5, 7 | never | 70-78 C | full | healthy |
| 4, 9 | ~68 s | 81-82 C | full | healthy, warmer slot |
| 6, 8 | 10-11 s | 88 C | halved by HW slowdown | defective cooling (dead fan or thermal pad) |

The room absorbs the load (idle thermometer cards +1 C with 4 loaded); the chassis is the limit:
loaded Titans sit at 85-89 C and throttle as their steady state (their own regulator; safe by
design). Adjacency is what decides speed: a loaded card between two loaded cards ran at 15-35%
of one with idle neighbours (measured on the A/B: 53k vs 8-18k tokens/s). A per-GPU power cap
(`sudo nvidia-smi -pl 200`, IT only) would fix most of that. Layout used for Ultra: train on the
six SPACED cards `0,2,4,5,7,9` (no two adjacent), `1,3,6,8` idle, 6 and 8 as thermometers (the
watchdog baselines them 15 min into the load and trips on the rise beyond that), plus the CPU
sensor as a non-GPU room reading. Eight cards held the room but not the chassis (GPU 4 reached
92 C at 8.5 min). Never all 10.

## Throughput on unequal cards

The run pace is set by the slowest card in a DDP step. `train.rank_shares_auto` (on by default) measures
each card's speed and re-deals the micro-batches every 10 steps, logged as `rank shares [..] -> [..]`
in the pretrain log and as `rank_shares` in `runs/pretrain/metrics.jsonl`. Nothing to configure;
`rank_shares:` in `configs/ultra_lambda.yaml` is only the starting split. Watch `tok_per_s`: the
theoretical ceiling is the sum of the cards' solo speeds (`scripts/bench_micro.py`).

## Unattended operation (what runs on the box, and how to check it)

Everything lives in tmux on the box; nothing depends on a laptop being connected.

| tmux session | what | log |
|---|---|---|
| `ab_*` (4) | shakedown A/B, one card each | `~/MaxGPT/ab_<variant>.log`, table: `python scripts/ab_report.py` |
| `autostart` | `scripts/ultra_autostart.sh CARDS THERMO`: waits for the A/B, runs `ab_verdict.py` (a variant must beat AdamW by >= 0.5% held-out loss to be adopted), carves the val shard, starts the dashboard + watchdog, presses play, then keeps checking every 10 min (restarts a dead dashboard/watchdog, presses play when idle, never over a thermal pause) | `~/MaxGPT/ultra_autostart.log` |
| `ultra` | the dashboard (`gui/server.py --config configs/ultra_lambda_final.yaml`) on `CUDA_VISIBLE_DEVICES=CARDS` | `~/MaxGPT/ultra_server.log`, `runs/pretrain/metrics.jsonl` |
| `thermal` | `scripts/thermal_watch.py` in pause/resume mode: 92 C hard line, hardware slowdown, +8 C thermometer rise, +8 C CPU rise -> dashboard pause; auto play once cooled; > 3 trips/hour -> stays paused | `~/MaxGPT/thermal_watch.out`, `thermal_events.log`, `thermal.csv`, state in `~/MaxGPT/thermal_state` |

Check on it: `bash scripts/lambda_status.sh` (everything on one screen), `tail ~/MaxGPT/thermal_events.log`,
`tail ~/MaxGPT/ultra_autostart.log`; dashboard: `ssh -L 8800:localhost:8800 lambda` then http://localhost:8800.
Stop everything: `tmux kill-server` (the trainer checkpoints every 15 min; the next
`ultra_autostart.sh` resumes from the checkpoint). Pause only the training: `curl -X POST
http://127.0.0.1:8800/api/pause`.

**Two gotchas after a `git pull` on the box.** The dashboard server (tmux `ultra`) keeps the
pipeline code in memory: a pulled change to `gui/server.py` (new flags on the train command, a
new stage) does nothing until the server itself is restarted: `curl -X POST :8800/api/pause`,
wait for `"running": false`, `tmux kill-session -t ultra`, relaunch it with the exact command the
keeper uses (`scripts/ultra_autostart.sh`, cards `0,4,5,9`), then `curl -X POST :8800/api/start`.
Pause + play alone only restarts the trainer subprocess, which picks up pulled changes to
`train/`, `data/`, `eval/` and `scripts/train.py` but not the command it was launched with.
`bash scripts/lambda_dashboard_restart.sh` does the whole sequence (pause, relaunch on the same
cards, play, print the new trainer's startup lines).
**Swapping in a new anneal build without a visit:** `scripts/anneal_autoswap.sh <new dir>` (run in
tmux `anneal_swap`) waits for the build's `meta.json`, runs `scripts/anneal_check.py` (token
budget met, chat >= 4%, CodeSearchNet >= 6%: the numbers a usable decay mix needs, which the first
two builds missed), pauses the run, moves the shards into `data/shards_anneal` (the previous build
is kept as `data/shards_anneal.prev_<stamp>`), and presses play, unless the watchdog holds a
thermal pause, in which case play is left to the watchdog. Log: `~/MaxGPT/anneal_swap.log`.
The trainer's stdout is not in `~/MaxGPT/*.log`: read
`runs/pretrain/ranks/<newest>/attempt_0/0/stdout.log` (the `[train] ...` startup lines: annealing,
benchmark suite, resumed-from step) or `/api/stage/pretrain` (slow; it ships the whole metrics file).

## Home PC

Stays paused while the video-editing model needs the 5070. The two checkouts share the same
code; the Lambda run is the primary one now.
