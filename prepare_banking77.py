#!/usr/bin/env python3
"""Prepare Banking77 for the federated learning demo.

What it does:
  1. Downloads Banking77 (10,003 training examples, 3,080 test examples, 77 intents).
  2. Groups the 77 intents into 5 support queues.
  3. Splits the training set unevenly (non-IID) across the branches: each branch
     keeps most of the queue that dominates its support traffic; the neutral
     queues are shared equally.
  4. Writes everything as JSONL in instruction/input/response format, ready for
     LoRA fine-tuning.

Usage:
    pip install datasets
    python prepare_banking77.py --out data
    python prepare_banking77.py --out data --dominance 0.9 --max-per-queue 1200

Output:
    data/clients/<branch>/train.jsonl   each branch's local training set
    data/clients/<branch>/val.jsonl     each branch's local validation set
    data/test.jsonl                     global test set (all queues)
    data/test_balanced.jsonl            test set with the same number of examples per queue
    data/demo_messages.json             short test messages for the results slide
    data/label_map.json                 intent -> queue
    data/stats.json                     counts per branch and per queue
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# 77 intents -> 5 queues
# ---------------------------------------------------------------------------
QUEUES = {
    "cards": {
        "pt": "Cartões",
        "intents": [
            "activate_my_card", "apple_pay_or_google_pay", "atm_support",
            "card_about_to_expire", "card_acceptance", "card_arrival",
            "card_delivery_estimate", "card_linking", "card_not_working",
            "card_payment_not_recognised", "card_swallowed",
            "cash_withdrawal_not_recognised", "change_pin", "compromised_card",
            "contactless_not_working", "declined_card_payment",
            "declined_cash_withdrawal", "disposable_card_limits",
            "get_disposable_virtual_card", "get_physical_card",
            "getting_spare_card", "getting_virtual_card", "lost_or_stolen_card",
            "order_physical_card", "pending_card_payment",
            "pending_cash_withdrawal", "pin_blocked", "reverted_card_payment",
            "supported_cards_and_currencies", "virtual_card_not_working",
            "visa_or_mastercard", "wrong_amount_of_cash_received",
        ],
    },
    "transfers": {
        "pt": "Transferências",
        "intents": [
            "balance_not_updated_after_bank_transfer", "beneficiary_not_allowed",
            "cancel_transfer", "declined_transfer",
            "direct_debit_payment_not_recognised", "failed_transfer",
            "pending_transfer", "receiving_money", "transfer_into_account",
            "transfer_not_received_by_recipient", "transfer_timing",
        ],
    },
    "top_ups": {
        "pt": "Recargas e depósitos",
        "intents": [
            "automatic_top_up", "balance_not_updated_after_cheque_or_cash_deposit",
            "pending_top_up", "top_up_by_cash_or_cheque", "top_up_failed",
            "top_up_limits", "top_up_reverted", "topping_up_by_card",
            "verify_top_up",
        ],
    },
    "fx_fees": {
        "pt": "Câmbio e tarifas",
        "intents": [
            "card_payment_fee_charged", "card_payment_wrong_exchange_rate",
            "cash_withdrawal_charge", "exchange_charge", "exchange_rate",
            "exchange_via_app", "extra_charge_on_statement",
            "fiat_currency_support", "refund_not_showing_up", "request_refund",
            "top_up_by_bank_transfer_charge", "top_up_by_card_charge",
            "transaction_charged_twice", "transfer_fee_charged",
            "wrong_exchange_rate_for_cash_withdrawal",
        ],
    },
    "account_identity": {
        "pt": "Conta e identidade",
        "intents": [
            "age_limit", "country_support", "edit_personal_details",
            "lost_or_stolen_phone", "passcode_forgotten", "terminate_account",
            "unable_to_verify_identity", "verify_my_identity",
            "verify_source_of_funds", "why_verify_identity",
        ],
    },
}

# Dominant queue of each branch. Queues not listed here are neutral.
# The "pt" queue names above are Portuguese labels used in the slides.
CLIENTS = {"brasil": "transfers", "mexico": "cards", "portugal": "fx_fees"}

QUEUE_IDS = list(QUEUES)
INSTRUCTION = (
    "Classify the bank customer message into exactly one support queue. "
    "Answer with the queue name only. Queues: " + ", ".join(QUEUE_IDS) + "."
)


def norm(name: str) -> str:
    """The dataset has inconsistent names ('Refund_not_showing_up', 'reverted_card_payment?')."""
    return name.strip().lower().rstrip("?")


INTENT_TO_QUEUE = {i: q for q, spec in QUEUES.items() for i in spec["intents"]}


def load_banking77():
    """Return (train, test) as lists of (text, normalized intent)."""
    from datasets import load_dataset

    last_err = None
    for repo in ("PolyAI/banking77", "legacy-datasets/banking77"):
        try:
            ds = load_dataset(repo)
            break
        except Exception as err:  # try the next repository
            last_err = err
    else:
        raise SystemExit(f"Could not download Banking77: {last_err}")

    names = [norm(n) for n in ds["train"].features["label"].names]

    def rows(split):
        return [(r["text"].strip(), names[r["label"]]) for r in ds[split]]

    return rows("train"), rows("test")


def check_mapping(intents_in_data):
    mapped = set(INTENT_TO_QUEUE)
    total = sum(len(s["intents"]) for s in QUEUES.values())
    if total != len(mapped):
        raise SystemExit("An intent is mapped to more than one queue.")
    missing = set(intents_in_data) - mapped
    extra = mapped - set(intents_in_data)
    if missing or extra:
        raise SystemExit(
            f"Incomplete mapping.\n  Without a queue: {sorted(missing)}\n  Not in the dataset: {sorted(extra)}"
        )


def record(text, intent):
    q = INTENT_TO_QUEUE[intent]
    return {
        "instruction": INSTRUCTION,
        "input": text,
        "response": q,
        "queue": q,
        "queue_pt": QUEUES[q]["pt"],
        "intent": intent,
    }


def split_among_clients(train, dominance, max_per_queue, rng):
    """Distribute each queue's examples among the branches."""
    by_queue = defaultdict(list)
    for text, intent in train:
        by_queue[INTENT_TO_QUEUE[intent]].append((text, intent))

    owner = {q: c for c, q in CLIENTS.items()}
    names = list(CLIENTS)
    out = {c: [] for c in names}

    for q in QUEUE_IDS:
        items = by_queue[q]
        rng.shuffle(items)
        if max_per_queue:
            items = items[:max_per_queue]
        if q in owner:
            others = [c for c in names if c != owner[q]]
            weights = {owner[q]: dominance, **{c: (1 - dominance) / len(others) for c in others}}
        else:
            weights = {c: 1 / len(names) for c in names}
        start = 0
        for i, c in enumerate(names):
            end = len(items) if i == len(names) - 1 else start + round(len(items) * weights[c])
            out[c].extend(items[start:end])
            start = end
    for c in names:
        rng.shuffle(out[c])
    return out


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for text, intent in rows:
            f.write(json.dumps(record(text, intent), ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data", help="output directory")
    ap.add_argument("--dominance", type=float, default=1.0,
                    help="fraction of a dominant queue kept by its owner branch (1.0 = only that branch sees the queue)")
    ap.add_argument("--val-frac", type=float, default=0.1, help="validation fraction in each branch")
    ap.add_argument("--max-per-queue", type=int, default=None,
                    help="cap on training examples per queue (balances and speeds things up)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not 1 / len(CLIENTS) <= args.dominance <= 1:
        raise SystemExit("--dominance must be between 1/n_branches and 1.0")

    rng = random.Random(args.seed)
    out = Path(args.out)

    train, test = load_banking77()
    check_mapping({i for _, i in train} | {i for _, i in test})

    stats = {"params": vars(args), "clients": {}, "test": {}}
    for client, rows in split_among_clients(train, args.dominance, args.max_per_queue, rng).items():
        n_val = max(1, round(len(rows) * args.val_frac))
        val, tr = rows[:n_val], rows[n_val:]
        write_jsonl(out / "clients" / client / "train.jsonl", tr)
        write_jsonl(out / "clients" / client / "val.jsonl", val)
        stats["clients"][client] = {
            "dominant_queue": CLIENTS[client],
            "train": len(tr),
            "val": len(val),
            "train_by_queue": dict(Counter(INTENT_TO_QUEUE[i] for _, i in tr)),
        }

    write_jsonl(out / "test.jsonl", test)
    test_by_queue = defaultdict(list)
    for row in test:
        test_by_queue[INTENT_TO_QUEUE[row[1]]].append(row)
    n_min = min(len(v) for v in test_by_queue.values())
    balanced = []
    for q in QUEUE_IDS:
        rng.shuffle(test_by_queue[q])
        balanced.extend(test_by_queue[q][:n_min])
    rng.shuffle(balanced)
    write_jsonl(out / "test_balanced.jsonl", balanced)
    stats["test"] = {"all": len(test), "balanced": len(balanced),
                     "by_queue": {q: len(v) for q, v in test_by_queue.items()}}

    # Short test messages for the results slide: 3 per queue.
    demo = {q: [t for t, _ in sorted(test_by_queue[q], key=lambda r: len(r[0]))
                if 25 <= len(t) <= 70][:3] for q in QUEUE_IDS}
    (out / "demo_messages.json").write_text(json.dumps(demo, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "label_map.json").write_text(json.dumps(
        {"queues": {q: s["pt"] for q, s in QUEUES.items()}, "intent_to_queue": INTENT_TO_QUEUE,
         "clients": CLIENTS, "instruction": INSTRUCTION}, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Data written to {out}/")
    for c, s in stats["clients"].items():
        print(f"  {c:9s} train={s['train']:5d} val={s['val']:4d}  {s['train_by_queue']}")
    print(f"  test      total={len(test)}  balanced={len(balanced)}")


if __name__ == "__main__":
    main()
