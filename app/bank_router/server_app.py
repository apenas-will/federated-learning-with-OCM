"""ServerApp: the aggregator, running on the hub.

It owns the global LoRA adapter, sends it to every branch each round, and
averages the adapters that come back (FedAvg). The hub has no customer data,
so there is no central evaluation: accuracy is measured by each branch on its
own validation set and only the numbers are reported back.
"""

import json
import os
import time
from collections.abc import Iterable
from pathlib import Path

from flwr.app import ArrayRecord, ConfigRecord, Context, Message
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from bank_router import task

app = ServerApp()

# The SuperExec-ServerApp pod deployed by the controller has no volume, so the
# result lands on the pod filesystem; `make fetch-model` copies it out.
OUTPUT_DIR = Path(os.environ.get("FL_OUTPUT_DIR", "/tmp/bank-router"))


def _branch(msg: Message) -> str:
    info = msg.content.config_records.get("info")
    return str(info["branch"]) if info else f"node-{msg.metadata.src_node_id}"


class LoggingFedAvg(FedAvg):
    """FedAvg that prints (and remembers) what each branch sent back, for the demo."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.per_branch: dict[int, dict] = {}  # round -> {"train": {...}, "evaluate": {...}}

    def aggregate_train(self, server_round: int, replies: Iterable[Message]):
        replies = list(replies)
        seen = self.per_branch.setdefault(server_round, {}).setdefault("train", {})
        for msg in replies:
            if msg.has_content():
                mb = sum(a.numel() * a.element_size() for a in msg.content["arrays"].to_torch_state_dict().values()) / 2**20
                seen[_branch(msg)] = {"update_mb": round(mb, 2), **dict(msg.content["metrics"])}
                print(f"[hub] round {server_round}: received {mb:.1f} MB LoRA update from {_branch(msg)}", flush=True)
        arrays, metrics = super().aggregate_train(server_round, replies)
        print(f"[hub] round {server_round}: averaged {len(replies)} adapters into the new global adapter", flush=True)
        return arrays, metrics

    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]):
        replies = list(replies)
        per_branch = {
            _branch(m): m.content["metrics"]["accuracy"] for m in replies if m.has_content()
        }
        self.per_branch.setdefault(server_round, {})["evaluate"] = per_branch
        metrics = super().aggregate_evaluate(server_round, replies)
        line = " | ".join(f"{b} {acc:.3f}" for b, acc in sorted(per_branch.items()))
        overall = metrics["accuracy"] if metrics else float("nan")
        print(f"[hub] round {server_round}: local validation accuracy {line} | weighted {overall:.3f}", flush=True)
        return metrics


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    task.set_threads(int(cfg["num-threads"]))
    num_rounds = int(cfg["num-server-rounds"])
    min_nodes = int(cfg["min-nodes"])

    model = task.load_model(lora_r=int(cfg["lora-r"]))
    initial = task.get_adapter_state(model)
    print(
        f"[hub] global model = frozen base ({task.base_model_bytes() / 2**20:.0f} MB, already in every branch) "
        f"+ LoRA adapter ({task.num_bytes(initial) / 2**20:.1f} MB, the only part that travels)",
        flush=True,
    )
    print(f"[hub] waiting for {min_nodes} branches, then running {num_rounds} rounds", flush=True)

    strategy = LoggingFedAvg(
        fraction_train=1.0,
        fraction_evaluate=1.0,
        min_train_nodes=min_nodes,
        min_evaluate_nodes=min_nodes,
        min_available_nodes=min_nodes,
    )
    start = time.time()
    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(initial),
        num_rounds=num_rounds,
        train_config=ConfigRecord({}),
        evaluate_config=ConfigRecord({}),
    )

    out = OUTPUT_DIR / f"run-{context.run_id}"
    task.set_adapter_state(model, result.arrays.to_torch_state_dict())
    model.save_pretrained(out / "adapter")
    history = {
        "run_id": context.run_id,
        "run_config": dict(cfg),
        "seconds": round(time.time() - start, 1),
        "train": {r: dict(m) for r, m in result.train_metrics_clientapp.items()},
        "evaluate": {r: dict(m) for r, m in result.evaluate_metrics_clientapp.items()},
        "per_branch": strategy.per_branch,
    }
    (out / "history.json").write_text(json.dumps(history, indent=2))
    latest = OUTPUT_DIR / "latest"
    latest.unlink(missing_ok=True)
    latest.symlink_to(out.name)
    print(f"[hub] done in {history['seconds']:.0f}s; final adapter saved to {out}/adapter", flush=True)

