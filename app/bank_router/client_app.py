"""ClientApp: runs inside each branch cluster, next to that branch's data.

The flower-addon SuperNode injects `cluster-name` into the node config, which
we use as the branch name. Training data is read from the branch-local data
service; it is never sent anywhere. What goes back to the aggregator is the
LoRA adapter (a few MB) plus a couple of metrics.
"""

import zlib

from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from bank_router import task

app = ClientApp()


def _setup(context: Context) -> tuple[dict, str, str]:
    cfg = context.run_config
    task.set_threads(int(cfg["num-threads"]))
    branch = str(context.node_config.get("cluster-name", f"node-{context.node_id}"))
    data_endpoint = str(context.node_config.get("data-endpoint", cfg["data-endpoint"])).rstrip("/")
    return cfg, branch, data_endpoint


def _model(cfg: dict, msg: Message):
    tok = task.load_tokenizer()
    model = task.load_model(lora_r=int(cfg["lora-r"]), device=task.pick_device(str(cfg["device"])))
    task.set_adapter_state(model, msg.content["arrays"].to_torch_state_dict())
    return tok, model


@app.train()
def train(msg: Message, context: Context) -> Message:
    cfg, branch, data_endpoint = _setup(context)
    server_round = int(msg.content["config"]["server-round"])
    tag = f"[{branch}] round {server_round}:"

    records = task.load_records(f"{data_endpoint}/train.jsonl")
    counts = task.queue_counts(records)
    print(f"{tag} {len(records)} local messages loaded from {data_endpoint} {counts}", flush=True)

    tok, model = _model(cfg, msg)
    stats = task.train(
        model,
        tok,
        records,
        steps=int(cfg["local-steps"]),
        batch_size=int(cfg["batch-size"]),
        lr=float(cfg["learning-rate"]),
        seed=server_round * 100_003 + zlib.crc32(branch.encode()),
        log_every=max(int(cfg["local-steps"]) // 4, 1),
        log_prefix=f"{tag} ",
    )
    adapter = task.get_adapter_state(model)
    print(
        f"{tag} trained on {stats['examples']} local examples in {stats['seconds']:.0f}s "
        f"(loss {stats['train_loss']:.3f}); sending LoRA update of "
        f"{task.num_bytes(adapter) / 2**20:.1f} MB, base model "
        f"({task.base_model_bytes() / 2**20:.0f} MB) and data stay here",
        flush=True,
    )

    content = RecordDict(
        {
            "arrays": ArrayRecord(adapter),
            "metrics": MetricRecord({"train_loss": stats["train_loss"], "num-examples": len(records)}),
            "info": ConfigRecord({"branch": branch}),
        }
    )
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    cfg, branch, data_endpoint = _setup(context)
    server_round = int(msg.content["config"]["server-round"])

    records = task.load_records(f"{data_endpoint}/val.jsonl")
    tok, model = _model(cfg, msg)
    result = task.evaluate(model, tok, records)
    print(
        f"[{branch}] round {server_round}: global model accuracy on {result['n']} local "
        f"validation messages = {result['accuracy']:.3f}",
        flush=True,
    )

    content = RecordDict(
        {
            "metrics": MetricRecord({"accuracy": result["accuracy"], "num-examples": result["n"]}),
            "info": ConfigRecord({"branch": branch}),
        }
    )
    return Message(content=content, reply_to=msg)
