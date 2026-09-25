"""The wider benchmark suite (docs/research_2026-09-22.md, section 2.4).

Zero-shot, scored by log-likelihood the way lm-evaluation-harness scores base models:
  lambada     EleutherAI/lambada_openai (test): predict the last word of a passage. acc = the greedy
              continuation is the target word (every token of it); also the target perplexity. This
              is the single-key-lookup probe for attention-output damage the research report asks for.
  piqa        ybisk/piqa (validation, parquet branch): goal + two solutions.
  winogrande  allenai/winogrande xl (validation): the two options fill the blank; partial scoring
              compares the log-likelihood of the text AFTER the blank under each filled-in prefix.
  arc_c       allenai/ai2_arc ARC-Challenge (test): question + 3-5 answers.
  hellaswag   Rowan/hellaswag (validation): context + 4 endings, lm-eval's text cleanup.
Multiple choice reports acc (raw log-likelihood) and acc_norm (per byte of the choice); the suite
average uses acc_norm for piqa / arc_c / hellaswag and acc for lambada / winogrande, the usual
convention in small-model tables.

One fixed suite. fetch_suite() downloads a seeded subset of each set ONCE into data/bench/*.jsonl
(the only step that needs the network); every later evaluation reads those files, so numbers are
comparable across checkpoints and stages (end of pretraining, post-SFT, post-DPO) and across the
periodic in-run evaluations. Noise floor: every accuracy carries its binomial standard error, and
fetching with another --seed draws a different subset of the same size, so a same-checkpoint
repeat measures the example-sampling noise directly.

Scoring is batched with right padding: under causal attention a padded tail cannot change earlier
positions, so batched scores equal one-at-a-time scores (scripts/test_eval.py [7] checks it).
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import time

import torch

TASKS = ("lambada", "piqa", "winogrande", "arc_c", "hellaswag")
PRIMARY = {"lambada": "acc", "piqa": "acc_norm", "winogrande": "acc", "arc_c": "acc_norm", "hellaswag": "acc_norm"}
SOURCES = {"lambada": "EleutherAI/lambada_openai test", "piqa": "ybisk/piqa validation (refs/convert/parquet)",
           "winogrande": "allenai/winogrande winogrande_xl validation", "arc_c": "allenai/ai2_arc ARC-Challenge test",
           "hellaswag": "Rowan/hellaswag validation"}


# --- fetching (network; run once) -------------------------------------------------------------
def _hs_clean(text: str) -> str:
    """lm-eval's HellaSwag preprocessing: drop the WikiHow markup, collapse doubled spaces."""
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def fetch_task(task: str, n: int = 1000, seed: int = 0) -> list[dict]:
    """A seeded subset of `n` examples of `task`, as the uniform records the scorers read."""
    from datasets import load_dataset
    rows: list[dict] = []
    if task == "lambada":
        for t in load_dataset("EleutherAI/lambada_openai", "default", split="test")["text"]:
            i = t.rfind(" ")
            if i > 0:
                rows.append({"context": t[:i], "target": t[i:]})
    elif task == "piqa":
        for ex in load_dataset("ybisk/piqa", revision="refs/convert/parquet", split="validation"):
            rows.append({"context": f"Question: {ex['goal']}\nAnswer:",
                         "choices": [" " + ex["sol1"], " " + ex["sol2"]], "answer": int(ex["label"])})
    elif task == "winogrande":
        for ex in load_dataset("allenai/winogrande", "winogrande_xl", split="validation"):
            s = ex["sentence"]
            i = s.index("_")
            rows.append({"contexts": [s[:i] + ex["option1"], s[:i] + ex["option2"]],
                         "continuation": " " + s[i + 1:].strip(), "answer": int(ex["answer"]) - 1})
    elif task == "arc_c":
        for ex in load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"):
            labels = ex["choices"]["label"]
            if ex["answerKey"] not in labels:
                continue
            rows.append({"context": f"Question: {ex['question']}\nAnswer:",
                         "choices": [" " + t for t in ex["choices"]["text"]], "answer": labels.index(ex["answerKey"])})
    elif task == "hellaswag":
        for ex in load_dataset("Rowan/hellaswag", split="validation"):
            ctx = _hs_clean(ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"].capitalize())
            rows.append({"context": ctx, "choices": [" " + _hs_clean(e) for e in ex["endings"]], "answer": int(ex["label"])})
    else:
        raise ValueError(f"unknown task {task!r}; known: {TASKS}")
    random.Random(seed).shuffle(rows)
    return rows[:n]


def write_suite(out_dir: str, suite: dict[str, list[dict]], seed: int = 0) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for task, rows in suite.items():
        with open(os.path.join(out_dir, f"{task}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {"seed": seed, "created": time.strftime("%Y-%m-%d %H:%M"),
                "tasks": {t: {"n": len(rows), "source": SOURCES.get(t, "")} for t, rows in suite.items()}}
    with open(os.path.join(out_dir, "suite.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def fetch_suite(out_dir: str, n: int = 1000, seed: int = 0, tasks=TASKS) -> dict[str, list[dict]]:
    suite = {}
    for task in tasks:
        t0 = time.time()
        suite[task] = fetch_task(task, n=n, seed=seed)
        print(f"[suite] {task}: {len(suite[task])} examples ({time.time() - t0:.1f}s)", flush=True)
    write_suite(out_dir, suite, seed=seed)
    return suite


def load_suite(dir_: str, tasks=TASKS, n: int | None = None) -> dict[str, list[dict]]:
    """The fixed suite from data/bench; a task whose file is missing is skipped."""
    suite = {}
    for task in tasks:
        p = os.path.join(dir_, f"{task}.jsonl")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        suite[task] = rows[:n] if n else rows
    return suite


# --- scoring ----------------------------------------------------------------------------------
@torch.no_grad()
def score_continuations(model, tokenizer, pairs, device: str = "cpu", batch_size: int = 8,
                        max_len: int = 512, amp=None) -> list[tuple[float, int, int, bool]]:
    """For each (context, continuation): (sum log p(continuation | context), n_tokens, n_bytes, greedy),
    where greedy says whether every continuation token was the argmax. Context and continuation are
    tokenized separately (as lm-eval does). Every sequence starts with <|endoftext|>, the separator
    the model saw before every document in training, so a context begins the way a document does;
    the oldest context tokens are dropped if the pair exceeds max_len. Batches are sorted by length
    and right-padded; causal attention keeps the padding from touching the scored positions."""
    model.eval()
    items = []
    for i, (ctx, cont) in enumerate(pairs):
        k = tokenizer.encode(cont)[:max_len - 2]
        c = [tokenizer.bos_id] + tokenizer.encode(ctx)[-(max_len - 1 - len(k)):]
        items.append((i, c + k, len(c), len(k), len(cont.encode("utf-8"))))
    order = sorted(range(len(items)), key=lambda j: len(items[j][1]))
    out: list = [None] * len(items)
    pad = getattr(tokenizer, "eos_id", 0)
    use_amp = amp is not None and str(device).startswith("cuda")
    for b in range(0, len(order), batch_size):
        batch = [items[j] for j in order[b:b + batch_size]]
        T = max(len(it[1]) for it in batch)
        x = torch.full((len(batch), T), pad, dtype=torch.long)
        for r, it in enumerate(batch):
            x[r, :len(it[1])] = torch.tensor(it[1], dtype=torch.long)
        x = x.to(device)
        with torch.autocast(device_type="cuda", dtype=amp, enabled=use_amp):
            logits, _ = model(x)
        for r, (i, ids, n_ctx, n_k, n_b) in enumerate(batch):
            if n_k == 0:
                out[i] = (0.0, 0, n_b, True)
                continue
            pos = torch.arange(n_ctx - 1, n_ctx - 1 + n_k, device=device)   # positions that predict the continuation
            tgt = torch.tensor(ids[n_ctx:], device=device)
            logp = torch.log_softmax(logits[r, pos].float(), dim=-1)
            out[i] = (float(logp.gather(1, tgt[:, None]).sum()), n_k, n_b,
                      bool((logp.argmax(-1) == tgt).all()))
    return out


def _se(p: float, n: int) -> float:
    return math.sqrt(max(p * (1 - p), 1e-12) / n) if n else 0.0


def eval_lambada(model, tokenizer, rows, **kw) -> dict:
    res = score_continuations(model, tokenizer, [(r["context"], r["target"]) for r in rows], **kw)
    n = len(rows)
    acc = sum(g for _, _, _, g in res) / max(1, n)
    toks = sum(k for _, k, _, _ in res)
    nll = -sum(lp for lp, _, _, _ in res) / max(1, toks)
    return {"acc": acc, "se": _se(acc, n), "ppl": math.exp(min(20.0, nll)), "n": n}


def eval_mc(model, tokenizer, rows, **kw) -> dict:
    """piqa / arc_c / hellaswag: one context, k choices; acc by raw log-likelihood, acc_norm per byte."""
    pairs = [(r["context"], ch) for r in rows for ch in r["choices"]]
    res = score_continuations(model, tokenizer, pairs, **kw)
    n, raw_ok, norm_ok, j = len(rows), 0, 0, 0
    for r in rows:
        k = len(r["choices"])
        sc = res[j:j + k]
        j += k
        raw = [lp for lp, _, _, _ in sc]
        norm = [lp / max(1, nb) for lp, _, nb, _ in sc]
        raw_ok += int(max(range(k), key=lambda i: raw[i]) == r["answer"])
        norm_ok += int(max(range(k), key=lambda i: norm[i]) == r["answer"])
    acc, acc_norm = raw_ok / max(1, n), norm_ok / max(1, n)
    return {"acc": acc, "acc_norm": acc_norm, "se": _se(acc_norm, n), "n": n}


def eval_winogrande(model, tokenizer, rows, **kw) -> dict:
    """Partial scoring: the same continuation under each option-filled prefix; higher likelihood wins."""
    pairs = [(c, r["continuation"]) for r in rows for c in r["contexts"]]
    res = score_continuations(model, tokenizer, pairs, **kw)
    n, ok, j = len(rows), 0, 0
    for r in rows:
        k = len(r["contexts"])
        sc = [lp for lp, _, _, _ in res[j:j + k]]
        j += k
        ok += int(max(range(k), key=lambda i: sc[i]) == r["answer"])
    acc = ok / max(1, n)
    return {"acc": acc, "se": _se(acc, n), "n": n}


def run_suite(model, tokenizer, suite: dict[str, list[dict]], device: str = "cpu", batch_size: int = 8,
              precision: str = "auto") -> dict:
    """All tasks in `suite` -> {task: metrics, 'avg': mean of the primary metrics, 'avg_se': its
    standard error, 'precision': the autocast dtype used, 'seconds': wall time}."""
    amp = None
    if str(device).startswith("cuda"):
        from train.trainer import amp_dtype
        amp = amp_dtype(precision, "cuda")           # it keys on the plain device kind
    # Score with the eager module: the trainer hands over its DDP-wrapped, torch.compile'd model,
    # and the suite's variable-length batches would recompile it shape by shape (minutes each).
    for attr in ("module", "_orig_mod", "module"):
        model = getattr(model, attr, model)
    kw = dict(device=device, batch_size=batch_size, amp=amp)
    t0, out = time.time(), {}
    for task, rows in suite.items():
        if not rows:
            continue
        if task == "lambada":
            out[task] = eval_lambada(model, tokenizer, rows, **kw)
        elif task == "winogrande":
            out[task] = eval_winogrande(model, tokenizer, rows, **kw)
        else:
            out[task] = eval_mc(model, tokenizer, rows, **kw)
    prim = [out[t][PRIMARY.get(t, "acc")] for t in out]
    ses = [out[t]["se"] for t in out]
    out["avg"] = sum(prim) / max(1, len(prim))
    out["avg_se"] = math.sqrt(sum(s * s for s in ses)) / max(1, len(ses))
    out["precision"] = str(amp).replace("torch.", "") if amp is not None else "fp32"
    out["seconds"] = round(time.time() - t0, 1)
    return out


def format_table(res: dict) -> str:
    lines = [f"{'task':<11}{'n':>6}{'acc':>8}{'acc_norm':>10}{'se':>7}{'ppl':>8}"]
    for t in TASKS:
        m = res.get(t)
        if not m:
            continue
        lines.append(f"{t:<11}{m['n']:>6}{100 * m['acc']:>8.1f}"
                     f"{(100 * m['acc_norm']) if 'acc_norm' in m else float('nan'):>10.1f}"
                     f"{100 * m['se']:>7.1f}{m['ppl'] if 'ppl' in m else float('nan'):>8.2f}")
    lines.append(f"{'average':<11}{'':>6}{100 * res['avg']:>8.1f}{'':>10}{100 * res['avg_se']:>7.1f}"
                 f"   ({res.get('precision')}, {res.get('seconds')}s)")
    return "\n".join(lines)


def last_suite_step(metrics_path: str) -> int | None:
    """The step of the most recent eval row that carries suite results, or None."""
    if not os.path.exists(metrics_path):
        return None
    last = None
    with open(metrics_path, encoding="utf-8") as f:
        for line in f:
            if '"suite"' not in line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("event") == "eval" and "suite" in r:
                last = int(r["step"])
    return last
