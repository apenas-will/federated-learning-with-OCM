#!/usr/bin/env python3
"""Compare the base model, a single-branch model and the federated model.

Evaluates every available model on the balanced global test set (all five
queues equally represented, data no single branch holds) and on the test
messages used in the slides, then writes:
    results/evaluation.json   full numbers
    results/summary.md        tables ready to paste into the slides

Models (skipped if the adapter is missing):
    base           Granite 4.0 350M, no fine-tuning
    brasil-only    models/adapters/brasil-only    (eval/train_local.py --branch brasil)
    federated      models/adapters/federated      (OCM run, infra/scripts/70-fetch-model.sh)
    federated-sim  models/adapters/federated-sim  (local processes, infra/scripts/local-federation.sh)

Usage:
    python eval/evaluate.py
    python eval/evaluate.py --limit 300      # quicker, subset of the test set
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from peft import PeftModel  # noqa: E402

from bank_router import task  # noqa: E402

# Test messages shown in the results slide, with the queue they should reach.
SLIDE_MESSAGES = [
    ("My transfer still hasn't arrived.", "transfers"),
    ("My card was cloned.", "cards"),
    ("Why is the exchange rate on my purchase wrong?", "fx_fees"),
]

ADAPTERS = {
    "brasil-only": ROOT / "models" / "adapters" / "brasil-only",
    "federated": ROOT / "models" / "adapters" / "federated",
    "federated-sim": ROOT / "models" / "adapters" / "federated-sim",
}


def load(name: str, device):
    model = task.load_model(with_lora=False, device=device)
    if name != "base":
        model = PeftModel.from_pretrained(model, ADAPTERS[name]).to(device)
    return model


def available_models() -> list[str]:
    return ["base"] + [n for n, p in ADAPTERS.items() if (p / "adapter_model.safetensors").exists()]


def evaluate_all(names, test, device, batch_size=32) -> dict:
    tok = task.load_tokenizer()
    out = {}
    for name in names:
        model = load(name, device)
        scores = task.evaluate(model, tok, test, batch_size=batch_size)
        test_preds = task.predict(model, tok, [r["input"] for r in test], batch_size=batch_size)
        preds = task.predict(model, tok, [m for m, _ in SLIDE_MESSAGES])
        out[name] = {
            "test": scores,
            "test_predictions": [p["queue"] for p in test_preds],
            "slide_messages": [
                {"message": m, "expected": q, "predicted": p["queue"], "confidence": p["probs"][p["queue"]]}
                for (m, q), p in zip(SLIDE_MESSAGES, preds)
            ],
        }
        print(f"{name:12s} accuracy={scores['accuracy']:.3f} "
              + " ".join(f"{q}={v:.2f}" for q, v in scores["per_queue"].items()), flush=True)
        del model
    return out


def disagreements(results: dict, test, worse: str, better: str, per_queue: int = 3) -> list[dict]:
    """Short test messages that `better` routes correctly and `worse` does not."""
    if worse not in results or better not in results:
        return []
    picked = []
    for q in task.QUEUES:
        rows = [
            {"message": r["input"], "expected": q, worse: results[worse]["test_predictions"][i],
             better: results[better]["test_predictions"][i]}
            for i, r in enumerate(test)
            if r["queue"] == q
            and results[better]["test_predictions"][i] == q
            and results[worse]["test_predictions"][i] != q
        ]
        picked += sorted(rows, key=lambda x: len(x["message"]))[:per_queue]
    return picked


def summary_markdown(results: dict, n_test: int, examples: list[dict] | None = None) -> str:
    names = list(results)
    lines = [f"## Accuracy on the balanced test set ({n_test} messages, all queues)", ""]
    lines.append("| Queue | " + " | ".join(names) + " |")
    lines.append("|---|" + "---|" * len(names))
    for q in task.QUEUES:
        lines.append(f"| {q} | " + " | ".join(f"{results[n]['test']['per_queue'][q]:.0%}" for n in names) + " |")
    lines.append("| **overall** | " + " | ".join(f"**{results[n]['test']['accuracy']:.0%}**" for n in names) + " |")
    lines += ["", "## Slide messages", ""]
    lines.append("| Message | Expected | " + " | ".join(names) + " |")
    lines.append("|---|---|" + "---|" * len(names))
    for i, (msg, expected) in enumerate(SLIDE_MESSAGES):
        cells = []
        for n in names:
            p = results[n]["slide_messages"][i]
            mark = "✓" if p["predicted"] == expected else "✗"
            cells.append(f"{p['predicted']} {mark} ({p['confidence']:.0%})")
        lines.append(f'| "{msg}" | {expected} | ' + " | ".join(cells) + " |")
    if examples:
        keys = [k for k in examples[0] if k not in ("message", "expected")]
        lines += ["", f"## Candidate messages: {keys[1]} right, {keys[0]} wrong", "",
                  "Real test-set messages (never seen in training), shortest first.", ""]
        lines.append("| Message | Expected | " + " | ".join(keys) + " |")
        lines.append("|---|---|" + "---|" * len(keys))
        for e in examples:
            lines.append(f'| "{e["message"]}" | {e["expected"]} | ' + " | ".join(e[k] for k in keys) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", default=str(ROOT / "data" / "test_balanced.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="evaluate on a random subset (0 = all)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    device = task.pick_device(args.device)
    test = task.load_records(args.test)
    if args.limit:
        random.Random(0).shuffle(test)
        test = test[: args.limit]
    names = available_models()
    print(f"device={device} test={len(test)} models={names}")

    results = evaluate_all(names, test, device)
    better = "federated" if "federated" in results else "federated-sim"
    examples = disagreements(results, test, worse="brasil-only", better=better)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps({"models": results, "candidate_messages": examples}, indent=2))
    md = summary_markdown(results, len(test), examples)
    (out / "summary.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
