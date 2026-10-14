"""Model, data and training code shared by the ServerApp, the ClientApp and the eval scripts.

The task: route a bank customer message to one of five support queues. We
fine-tune a small causal LM with LoRA; only the LoRA adapter weights are
trained and exchanged, the base model stays frozen (and local) everywhere.

Prediction is constrained to the five queues: the first token of every queue
name is distinct, so the model's answer is the queue whose first token gets
the highest probability right after "Queue:".
"""

from __future__ import annotations

import json
import math
import os
import random
import time
import urllib.request
from collections.abc import Sequence
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

QUEUES = ["cards", "transfers", "top_ups", "fx_fees", "account_identity"]
PROMPT = (
    "Route the bank customer message to one support queue: "
    + ", ".join(QUEUES)
    + ".\nMessage: {text}\nQueue:"
)
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Inside the container image the model is baked in at /models; on a laptop the
# eval scripts point BASE_MODEL_PATH at models/granite-4.0-350m-fp32. The
# weights are stored in fp32 so they load by mmap without a conversion copy.
BASE_MODEL = os.environ.get("BASE_MODEL_PATH", "models/granite-4.0-350m-fp32")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_records(source: str) -> list[dict]:
    """Read a JSONL dataset from a local path or an http(s) URL.

    In the clusters, `source` is the branch-local data service
    (http://branch-data.bank-data.svc.cluster.local:8000/...), which is only
    resolvable and reachable from inside that branch's cluster.
    """
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=30) as resp:
            text = resp.read().decode("utf-8")
    else:
        text = Path(source).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def queue_counts(records: Sequence[dict]) -> dict[str, int]:
    counts = {q: 0 for q in QUEUES}
    for r in records:
        counts[r["queue"]] += 1
    return counts


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def set_threads(num_threads: int) -> None:
    """Several clients share the same CPUs in the demo, so cap each one."""
    if num_threads > 0:
        torch.set_num_threads(num_threads)
    # On Linux/arm64 the oneDNN (mkldnn) CPU kernels made LoRA backward ~10x
    # slower and peaked at ~9 GB per process; the default ATen kernels do not.
    torch.backends.mkldnn.enabled = False


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tokenizer(base_model: str = BASE_MODEL):
    tok = AutoTokenizer.from_pretrained(base_model)
    tok.padding_side = "left"  # answers sit at the end of every row
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(
    base_model: str = BASE_MODEL,
    lora_r: int = 8,
    with_lora: bool = True,
    device: torch.device | None = None,
):
    """Frozen base model, optionally wrapped with a freshly initialised LoRA adapter."""
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32)
    if with_lora:
        cfg = LoraConfig(
            r=lora_r,
            lora_alpha=2 * lora_r,
            lora_dropout=0.05,
            target_modules=LORA_TARGETS,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, cfg)
    if device is not None:
        model.to(device)
    return model


def get_adapter_state(model) -> dict[str, torch.Tensor]:
    """The only weights that ever leave a branch: the LoRA adapter."""
    return {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}


def set_adapter_state(model, state: dict[str, torch.Tensor]) -> None:
    result = set_peft_model_state_dict(model, state)
    unexpected = getattr(result, "unexpected_keys", None)
    if unexpected:
        raise ValueError(f"Adapter keys do not match the model: {unexpected[:3]}")


def num_bytes(state: dict[str, torch.Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in state.values())


def base_model_bytes(base_model: str = BASE_MODEL) -> int:
    path = Path(base_model)
    if path.is_dir():
        return sum(f.stat().st_size for f in path.glob("*.safetensors"))
    return 0


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------
def _queue_first_token_ids(tok) -> list[int]:
    ids = [tok.encode(" " + q, add_special_tokens=False)[0] for q in QUEUES]
    if len(set(ids)) != len(ids):
        raise ValueError("Queue names must start with distinct tokens for this tokenizer")
    return ids


def _left_pad(rows: list[list[int]], pad_id: int, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(r) for r in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, width - len(r):] = torch.tensor(r)
        mask[i, width - len(r):] = 1
    # Positions must ignore the left padding so RoPE sees the same offsets as unpadded text.
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    return ids.to(device), mask.to(device), pos.to(device)


def _prompt_ids(tok, text: str) -> list[int]:
    return tok.encode(PROMPT.format(text=text.strip()), add_special_tokens=False)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(
    model,
    tok,
    records: Sequence[dict],
    steps: int,
    batch_size: int = 4,
    lr: float = 5e-4,
    seed: int = 0,
    log_every: int = 0,
    log_prefix: str = "",
) -> dict[str, float]:
    """Run `steps` optimizer steps of LoRA fine-tuning on randomly sampled records.

    Only the answer tokens (queue name + end-of-text) contribute to the loss,
    and only their logits are computed, which keeps the 100k-token vocabulary
    from dominating time and memory.
    """
    device = next(model.parameters()).device
    rng = random.Random(seed)
    order = list(range(len(records)))
    rng.shuffle(order)
    eos = tok.eos_token_id

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    model.train()

    losses, cursor, start = [], 0, time.time()
    for step in range(steps):
        batch = []
        for _ in range(batch_size):
            if cursor == len(order):
                rng.shuffle(order)
                cursor = 0
            batch.append(records[order[cursor]])
            cursor += 1

        rows, answer_lens = [], []
        for r in batch:
            answer = tok.encode(" " + r["queue"], add_special_tokens=False) + [eos]
            rows.append(_prompt_ids(tok, r["input"]) + answer)
            answer_lens.append(len(answer))
        ids, mask, pos = _left_pad(rows, tok.pad_token_id, device)

        k = max(answer_lens)
        out = model(input_ids=ids, attention_mask=mask, position_ids=pos, logits_to_keep=k + 1)
        logits = out.logits[:, :k, :].float()  # positions T-k-1 .. T-2 predict tokens T-k .. T-1
        targets = ids[:, -k:].clone()
        for i, a in enumerate(answer_lens):
            targets[i, : k - a] = -100  # those positions are still prompt tokens
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(loss.item())

        if log_every and (step + 1) % log_every == 0:
            rate = (time.time() - start) / (step + 1)
            print(f"{log_prefix}step {step + 1}/{steps} loss={sum(losses[-log_every:]) / log_every:.3f} ({rate:.1f}s/step)", flush=True)

    return {
        "train_loss": sum(losses) / max(len(losses), 1),
        "seconds": time.time() - start,
        "examples": steps * batch_size,
    }


# ---------------------------------------------------------------------------
# Inference / evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(model, tok, texts: Sequence[str], batch_size: int = 16) -> list[dict]:
    """Return, for every text, the predicted queue and the probability of each queue."""
    device = next(model.parameters()).device
    queue_ids = torch.tensor(_queue_first_token_ids(tok), device=device)
    model.eval()
    rows = [_prompt_ids(tok, t) for t in texts]
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]))  # batch similar lengths, less padding
    results: list[dict] = [{} for _ in rows]
    for i in range(0, len(order), batch_size):
        idx = order[i : i + batch_size]
        ids, mask, pos = _left_pad([rows[j] for j in idx], tok.pad_token_id, device)
        logits = model(input_ids=ids, attention_mask=mask, position_ids=pos, logits_to_keep=1).logits[:, -1, :]
        probs = torch.softmax(logits.float()[:, queue_ids], dim=-1).cpu()
        for j, p in zip(idx, probs):
            k = int(p.argmax())
            results[j] = {"queue": QUEUES[k], "probs": {q: round(float(p[n]), 4) for n, q in enumerate(QUEUES)}}
    return results


def evaluate(model, tok, records: Sequence[dict], batch_size: int = 16) -> dict:
    """Overall and per-queue accuracy."""
    preds = predict(model, tok, [r["input"] for r in records], batch_size=batch_size)
    hits = {q: [0, 0] for q in QUEUES}
    for r, p in zip(records, preds):
        hits[r["queue"]][1] += 1
        hits[r["queue"]][0] += int(p["queue"] == r["queue"])
    total = sum(n for _, n in hits.values())
    correct = sum(c for c, _ in hits.values())
    return {
        "accuracy": correct / total if total else math.nan,
        "per_queue": {q: (c / n if n else math.nan) for q, (c, n) in hits.items()},
        "n": total,
    }
