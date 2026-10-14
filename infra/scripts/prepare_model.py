#!/usr/bin/env python3
"""Download IBM Granite 4.0 350M and store it as fp32 safetensors.

The published weights are bf16. Every ClientApp computes in fp32 on CPU, and
loading bf16 means a converted copy in memory on top of the original; with
fp32 on disk the weights are memory-mapped instead (~2 GB less peak RSS per
branch). The output directory is the build context of app/Containerfile.
"""

import argparse
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="ibm-granite/granite-4.0-350m")
    ap.add_argument("--download-dir", default=str(ROOT / "models" / "granite-4.0-350m"))
    ap.add_argument("--out", default=str(ROOT / "models" / "granite-4.0-350m-fp32"))
    args = ap.parse_args()

    out = Path(args.out)
    if (out / "model.safetensors").exists():
        print(f"{out} already exists, nothing to do")
        return

    src = Path(snapshot_download(
        args.repo, local_dir=args.download_dir,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "README.md"],
    ))
    model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32)
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(src).save_pretrained(out)
    shutil.copy(src / "README.md", out / "README.md")  # model card, includes the Apache-2.0 license
    size = sum(p.stat().st_size for p in out.glob("*.safetensors")) / 2**20
    print(f"saved {args.repo} as fp32 to {out} ({size:.0f} MiB)")


if __name__ == "__main__":
    main()
