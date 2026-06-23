# MaxGPT: Mini -> Ultra runbook

The exact, ordered steps to stop Mini's DPO, preserve Mini, and launch the real Ultra run.

**Two machines are involved:**
- **Mac** = dev. You only push code from here.
- **PC** (RTX 5070) = training. Everything else happens here.

Unless a step says otherwise, PC commands run from inside the `maxgpt-ultra` folder. Destructive
file ops are shown as PowerShell; you can also just do them in File Explorer.

---

## Phase 0 — Get the latest code onto the PC

**On the Mac** (repo root, `Max's AI Model`):
```bash
git add -A
git commit -m "DPO speedup, KV-cache, micro_batch autotuner, save_model, resumable data prep" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
git push
```

**On the PC** (in the repo): `git pull`
This is what delivers `save_model.py`, `find_micro_batch.py`, and the resumable data prep.

---

## Phase 1 — Stop Mini's DPO and preserve it

1. In the training GUI, **pause/STOP the DPO stage** (it checkpoints on stop). Then Ctrl-C the GUI
   server in its terminal.
2. Lock Mini into `models/` (one command; it finds the latest checkpoint and verifies it loads):
   ```bash
   python scripts/save_model.py --run runs/dpo --tokenizer tokenizer/maxgpt-ultra.tokenizer.json --name mini --config configs/shakedown.yaml
   ```
   Creates `models/mini.pt` + `models/mini.tokenizer.json`.
   **Must happen before any deleting** — once the shared tokenizer/runs are wiped, Mini's webui
   fallback paths would otherwise resolve to Ultra's files.

---

## Phase 2 — Confirm Mini works in the webui (before destroying anything)

**From the repo root** (one level up from `maxgpt-ultra`):
```bash
streamlit run webui/app.py        # pip install streamlit if needed
```
Open it, pick **MaxGPT-Mini** (the Neo family), send a message, confirm it replies. First real webui
load of a Neo model, so if it looks wrong, stop and fix before proceeding. Ctrl-C when done.

(Only **Mini** exists to test now. Ultra is tested in Phase 7, after it trains.)

---

## Phase 3 — Find Ultra's micro_batch (GPU is idle now)

```bash
python scripts/find_micro_batch.py --config configs/ultra.yaml --compile
```
`--compile` gives the authoritative number (eager search first, then confirms under real
max-autotune near the boundary; ~10-20 min). Note the recommended `micro_batch` and `grad_accum`.

---

## Phase 4 — Clear Mini's data so Ultra starts clean

Mini is safe in `models/` now. PowerShell (or File Explorer):
```powershell
Remove-Item tokenizer\maxgpt-ultra.tokenizer.json     # forces tokenizer RETRAIN for the new mix
Remove-Item -Recurse -Force data\shards               # forces pretrain data rebuild
Remove-Item data\sft.jsonl, data\prefs.jsonl          # forces SFT/DPO data rebuild
Rename-Item runs runs_mini                            # so Ultra won't resume Mini's 113M checkpoints
```

---

## Phase 5 — Set Ultra's config

Edit `configs/ultra.yaml`, set the two values from Phase 3:
```yaml
  micro_batch: <from autotuner>
  grad_accum:  <from autotuner>
```
(Data budget needs no change: `total_tokens: 1.0e11` builds 100B **unique** tokens, one epoch,
which is the intended over-trained recipe. See Notes.)

---

## Phase 6 — Start Ultra

```bash
python gui/server.py --config configs/ultra.yaml
```
Open `http://localhost:8800`, press **play**. Stages run:
**data** (retrains tokenizer, tokenizes the 100B mix, builds SFT+OASST + prefs) -> **pretrain** ->
**sft** -> **dpo**.

- The **data stage is a multi-day job** (downloading + tokenizing ~100B tokens). It is **resumable**:
  if the PC reboots / crashes / you stop the GUI mid-build, just press play again and it picks up at
  the last shard boundary (no re-downloading). It is only "done" once `data/shards/meta.json` exists.
- When **pretrain** first starts, `torch.compile` warms up for a few minutes before steps appear
  (not frozen).

---

## Phase 7 — When Ultra finishes (weeks out)

```bash
python scripts/save_model.py --run runs/dpo --tokenizer tokenizer/maxgpt-ultra.tokenizer.json --name ultra --config configs/ultra.yaml
```
Then `streamlit run webui/app.py` from the repo root. **Both** Mini and Ultra now load.

---

## Easy-to-miss steps (why each matters)

1. **Preserve Mini before wiping** (Phase 1) — its webui fallbacks (`runs/dpo`, the shared
   tokenizer) become *Ultra's* after the wipe, so `models/mini.*` must exist first.
2. **Delete the tokenizer** (Phase 4) — `prepare_data` *skips* retraining if the file exists, so
   leaving it means Ultra trains on Mini's tokenizer.
3. **Move `runs/` aside** (Phase 4) — otherwise Ultra tries to resume Mini's 113M checkpoints and
   crashes on the size mismatch.
4. **GPU idle for the autotuner** (Phase 3) — stop everything else on the card first.
5. **Run the webui test before wiping** (Phase 2) — so a bad preserve surfaces while the source
   still exists.

---

## Notes

- **Why 100B unique (not a smaller set repeated):** over-training a small model on lots of unique
  tokens (well past Chinchilla-optimal) is how the strong modern small models are made; unique beats
  repetition. The mix holds near 55/22/10/8/5 at 100B (FineWeb-edu-dedup ~220B and Cosmopedia ~28B
  cover their shares; Wikipedia/code sit near their one-epoch ceiling, with FineWeb absorbing any
  shortfall). Cost is the download (~hundreds of GB, day+), not disk.
- **Resumability:** `tokenize_to_shards` checkpoints `data/shards/progress.json` at every shard
  boundary (rng + each source's stream position + the in-flight doc), so resume is exact (verified
  identical to an uninterrupted build). It auto-resumes on the next run; deletes `progress.json` when
  meta.json is written.

## Path reference

| What | Path |
|---|---|
| Mini checkpoint / tokenizer (webui) | `models/mini.pt`, `models/mini.tokenizer.json` |
| Ultra checkpoint / tokenizer (webui) | `models/ultra.pt`, `models/ultra.tokenizer.json` |
| Shared tokenizer (rebuilt per run) | `tokenizer/maxgpt-ultra.tokenizer.json` |
| Pretrain shards | `data/shards/` (+ `meta.json`, `progress.json` while building) |
| SFT / DPO data | `data/sft.jsonl`, `data/prefs.jsonl` |
| Run checkpoints | `runs/{pretrain,sft,dpo}/checkpoints/` |
| Configs | `configs/shakedown.yaml` (Mini), `configs/ultra.yaml` (Ultra) |
