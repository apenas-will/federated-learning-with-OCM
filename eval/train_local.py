#!/usr/bin/env python3
"""Train the queue router on ONE branch's data only (no federation).

This is the "trained only in Brazil" baseline for the results slide. By
default it uses the same hyperparameters as the federated run and the same
amount of local compute a branch spends during it
(num-server-rounds x local-steps optimizer steps), read from app/pyproject.toml.

Usage:
    python eval/train_local.py --branch brasil
    python eval/train_local.py --branch brasil --steps 40 --device cpu --threads 3   # quick timing check
"""

import argparse
import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import torch  # noqa: E402

from bank_router import task  # noqa: E402


def run_config() -> dict:
    pyproject = tomllib.loads((ROOT / "app" / "pyproject.toml").read_text())
    return pyproject["tool"]["flwr"]["app"]["config"]


def main() -> None:
    cfg = run_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--branch", default="brasil", help="branch whose data/clients/<branch>/train.jsonl is used")
    ap.add_argument("--data", default=None, help="explicit train JSONL path or URL (overrides --branch)")
    ap.add_argument("--steps", type=int, default=cfg["num-server-rounds"] * cfg["local-steps"])
    ap.add_argument("--batch-size", type=int, default=cfg["batch-size"])
    ap.add_argument("--lr", type=float, default=cfg["learning-rate"])
    ap.add_argument("--lora-r", type=int, default=cfg["lora-r"])
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = PyTorch default)")
    ap.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="adapter output dir (default: models/adapters/<branch>-only)")
    args = ap.parse_args()

    task.set_threads(args.threads)
    torch.manual_seed(args.seed)
    device = task.pick_device(args.device)
    data = args.data or str(ROOT / "data" / "clients" / args.branch / "train.jsonl")
    out = Path(args.out or ROOT / "models" / "adapters" / f"{args.branch}-only")

    records = task.load_records(data)
    print(f"{len(records)} training messages from {data} {task.queue_counts(records)}")
    print(f"device={device} steps={args.steps} batch={args.batch_size} lr={args.lr} lora_r={args.lora_r}")

    tok = task.load_tokenizer()
    model = task.load_model(lora_r=args.lora_r, device=device)
    stats = task.train(
        model, tok, records, steps=args.steps, batch_size=args.batch_size, lr=args.lr,
        seed=args.seed, log_every=max(args.steps // 10, 1), log_prefix=f"[{args.branch}-only] ",
    )
    model.save_pretrained(out)
    (out / "train_stats.json").write_text(json.dumps({**stats, **vars(args), "data": data}, indent=2))
    print(f"done in {stats['seconds']:.0f}s ({stats['seconds'] / args.steps:.2f}s/step); adapter saved to {out}")


if __name__ == "__main__":
    main()
