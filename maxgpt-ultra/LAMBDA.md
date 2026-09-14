# MaxGPT-Ultra on the school Lambda server

`lambda.ccm.edu` is a shared Lambda box (Ubuntu 20.04) on the campus network. It has no
public address, so it is reached from campus wifi, or from home through the school VDI:
Omnissa Horizon Client -> `view.ccm.edu` -> ITLabs desktop -> PowerShell -> `ssh`.
Account `brohm.maxwell`, no sudo. Everything we install lives in `$HOME`.

## 1. Log in

```
ssh brohm.maxwell@lambda.ccm.edu
```

Answer `yes` to the host-key question (ED25519 `SHA256:7/aJBtVD1YqWpGUcchP8zVC1d1DpShaID2wst7U2WRI`).
The first login forces a password change and then drops the connection; reconnect with
the new password.

## 2. Bootstrap (one command, re-runnable)

```
git clone https://github.com/MaxwellBrohm/MaxGPT.git ~/MaxGPT
bash ~/MaxGPT/maxgpt-ultra/scripts/lambda_setup.sh
```

Installs uv, Python 3.12, `~/venv`, a torch wheel that matches the box's driver
(570+ -> cu128, the same torch 2.11 as home; 525-569 -> cu126; older -> cu118), the
project deps, then runs the smoke tests and prints a summary. Paste the summary into the
chat: the GPU model decides the config changes below.

## 3. Sharing the box

- `nvidia-smi` first, every time. The process table at the bottom shows everyone's jobs.
- Pin yourself to a free card: `export CUDA_VISIBLE_DEVICES=1` (or `1,2`). The trainer
  uses whatever card 0 is inside that mask; it never picks cards on its own.
- Run everything inside tmux (`tmux new -s ultra`, later `tmux attach -t ultra`) so
  closing the VDI or losing wifi does not kill the run.
- Ask before taking every card, even at night.

## 4. Data

Pick after the summary shows disk space and whether the box has internet:

- **Rebuild on the server** (default if it has internet and ~250 GB free). Copy only the
  tokenizer so the tokens come out byte-identical to the PC build, then run the same data
  stage as at home. Resumable, same progress bar in the dashboard.
  ```
  scp tokenizer/maxgpt-ultra.tokenizer.json brohm.maxwell@lambda.ccm.edu:MaxGPT/maxgpt-ultra/tokenizer/
  ```
- **Carry the finished shards** (~200 GB on the PC): external SSD to campus, then from the
  Mac on a wired jack:
  ```
  rsync -avP /Volumes/<ssd>/shards/ brohm.maxwell@lambda.ccm.edu:MaxGPT/maxgpt-ultra/data/shards/
  ```
  Only works if the Mac can log in directly (campus wifi login was still unverified when
  this was written; the VDI path always works for the ssh session itself).

## 5. Run

```
tmux new -s ultra
source ~/venv/bin/activate && cd ~/MaxGPT/maxgpt-ultra
export CUDA_VISIBLE_DEVICES=<free card>
python scripts/find_micro_batch.py --config configs/ultra.yaml --compile   # retune for the bigger card
python gui/server.py --config configs/ultra.yaml                           # then press play in the dashboard
```

Dashboard from the VDI: in a second PowerShell run
`ssh -L 8800:localhost:8800 brohm.maxwell@lambda.ccm.edu`, then open
http://localhost:8800 in the VDI's browser. Same trick from the Mac on campus.

## 6. Retune (after the summary)

- 24 GB+ card: turn off `grad_checkpointing` and `optimizer_8bit` in `configs/ultra.yaml`
  (both exist only to fit 12 GB), re-run the autotuner, expect a large step-speed jump.
- Pre-Ampere card (V100, RTX 20-series, `sm_70` / `sm_75`): no bf16. The trainer is
  bf16-only today; it needs an fp16 + GradScaler path in `train/trainer.py`. Not written
  yet, on purpose: build it only if the card needs it.
- Several free cards: multi-GPU (DDP) is a separate change, not in the trainer today.

## Home PC

Stays paused while the video-editing model needs the 5070. Do not press play there
until that is confirmed done.
