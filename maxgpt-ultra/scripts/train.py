"""Train MaxGPT-Ultra. The "run and walk away" entry point for the 5070 box.

  python scripts/train.py --config configs/ultra.yaml --data data/shards --out runs/ultra

Resumes automatically from the latest checkpoint in --out if one exists, so you can
stop (or lose power) and just run the same command again. Pause/play from the GUI uses
the same mechanism.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # maxgpt-ultra/

import yaml
import torch

from model import ModelConfig, MaxGPTUltra
from data import PackedShardDataset
from train.trainer import Trainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="model+train YAML (e.g. configs/ultra.yaml)")
    ap.add_argument("--data", required=True, help="shard directory containing meta.json")
    ap.add_argument("--out", default="runs/ultra", help="output dir (checkpoints + metrics)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="optional cap (for testing)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)
    mcfg = ModelConfig.from_yaml(args.config)
    tcfg = raw["train"]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model = MaxGPTUltra(mcfg)
    data = PackedShardDataset(args.data, mcfg.seq_len)
    trainer = Trainer(model, data, tcfg, device, args.out, seed=args.seed)
    resumed = trainer.resume_if_available()

    print(f"[train] device={device} params={model.num_params()/1e6:.1f}M "
          f"total_steps={trainer.total_steps} tokens/step={trainer.tokens_per_step} "
          f"resumed={resumed} (from step {trainer.step})")
    trainer.train(max_steps=args.max_steps)
    print(f"[train] done at step {trainer.step}; checkpoints + metrics in {args.out}")


if __name__ == "__main__":
    main()
