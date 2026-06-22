"""Promote a finished run's checkpoint + tokenizer into models/ under a stable name, so the
webui keeps loading it after data/ and runs/ get wiped for the next model.

  python scripts/save_model.py --run runs/dpo --tokenizer tokenizer/maxgpt-ultra.tokenizer.json \
                               --name mini --config configs/shakedown.yaml

Resolves the run's latest checkpoint (runs/<>/checkpoints/latest.json), copies it to
models/<name>.pt and the tokenizer to models/<name>.tokenizer.json, then rebuilds the model
+ tokenizer from the COPIES to prove they load before you delete the originals.
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # maxgpt-ultra/


def resolve_ckpt(run_or_ckpt: str) -> str:
    """A direct .pt path, or a run dir whose checkpoints/latest.json names the newest one."""
    if run_or_ckpt.endswith(".pt"):
        return run_or_ckpt
    latest = os.path.join(run_or_ckpt, "checkpoints", "latest.json")
    if not os.path.exists(latest):
        sys.exit(f"[save_model] no checkpoints/latest.json under {run_or_ckpt} (did that stage run?)")
    name = json.load(open(latest, encoding="utf-8"))["path"]
    return os.path.join(run_or_ckpt, "checkpoints", name)


def main() -> None:
    ap = argparse.ArgumentParser(description="Promote a run's checkpoint + tokenizer into models/.")
    ap.add_argument("--run", required=True, help="run dir (e.g. runs/dpo) or a direct .pt file")
    ap.add_argument("--tokenizer", required=True, help="tokenizer json to bundle alongside it")
    ap.add_argument("--name", required=True, help="output name: mini or ultra")
    ap.add_argument("--config", default=None, help="model yaml, to verify the weights load (recommended)")
    ap.add_argument("--models-dir", default="models")
    args = ap.parse_args()

    src = resolve_ckpt(args.run)
    if not os.path.exists(src):
        sys.exit(f"[save_model] checkpoint not found: {src}")
    if not os.path.exists(args.tokenizer):
        sys.exit(f"[save_model] tokenizer not found: {args.tokenizer}")

    os.makedirs(args.models_dir, exist_ok=True)
    dst_ckpt = os.path.join(args.models_dir, f"{args.name}.pt")
    dst_tok = os.path.join(args.models_dir, f"{args.name}.tokenizer.json")
    shutil.copy2(src, dst_ckpt)
    shutil.copy2(args.tokenizer, dst_tok)
    print(f"[save_model] checkpoint  {src}  ->  {dst_ckpt}")
    print(f"[save_model] tokenizer   {args.tokenizer}  ->  {dst_tok}")

    # verify the copies actually load, so a problem surfaces here, not in the webui later
    import torch
    from tokenizer.tokenizer import UltraTokenizer
    ck = torch.load(dst_ckpt, map_location="cpu", weights_only=False)
    step = ck.get("step", "?")
    tok = UltraTokenizer(dst_tok)
    if args.config:
        from model import ModelConfig, MaxGPTUltra
        model = MaxGPTUltra(ModelConfig.from_yaml(args.config))
        model.load_state_dict(ck["model"])
        print(f"[save_model] verified: {model.num_params() / 1e6:.1f}M params load cleanly, "
              f"step {step}, tokenizer vocab {tok.vocab_size}")
    else:
        print(f"[save_model] copied (step {step}, tokenizer vocab {tok.vocab_size}); "
              f"pass --config to also verify the weights load")
    print(f"[save_model] '{args.name}' is preserved in {args.models_dir}/ and ready for the webui.")


if __name__ == "__main__":
    main()
