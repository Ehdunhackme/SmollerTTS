"""Train the released acoustic architecture from random initialization."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .acoustic import AcousticModel
from .checkpoints import atomic_json, capture_rng, load_dataset, restore_rng, sha256
from .data import Steps, batch, init_worker
from .loss import acoustic_objective, training_objective


def to_device(values, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in values.items()}


def learning_rate(step, config):
    return config["learning_rate"] * min(1., step / config["warmup_steps"]) * .5 ** (step / config["decay_half_life"])


def validate_config(config):
    for key in ("steps", "warmup_steps", "decay_half_life", "max_batch", "frame_budget", "checkpoint_every", "evaluate_every"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("learning_rate", "gradient_clip"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if not math.isfinite(config["weight_decay"]) or config["weight_decay"] < 0:
        raise ValueError("Invalid weight decay")
    if len(config["betas"]) != 2 or any(not 0 <= x < 1 for x in config["betas"]):
        raise ValueError("Invalid AdamW betas")


@torch.no_grad()
def evaluate(model, rows, device):
    """Aligned durations, predicted prosody. Listening uses predicted timing too."""
    model.eval()
    totals = {}
    for start in range(0, len(rows), 4):
        selected = rows[start:start + 4]
        _, metrics = acoustic_objective(model, to_device(batch(selected), device), recorded=False)
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.) + float(value) * len(selected)
    model.train()
    return {name: value / len(rows) for name, value in totals.items()}


def run(data, output, config, device="cpu", workers=0, threads=4, resume=None, stop_after=None, verify_data=False):
    validate_config(config)
    output, data = Path(output), Path(data)
    if output.exists():
        raise FileExistsError("Use a new output directory, including when resuming")
    if workers < 0 or threads < 1:
        raise ValueError("Invalid worker/thread count")
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("This recipe supports CPU or CUDA")
    # Required by deterministic CUDA matmul, before the first CUDA allocation.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(threads)
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = config["tf32"]
    torch.backends.cudnn.allow_tf32 = config["tf32"]
    splits, norm, summary = load_dataset(data, verify_features=verify_data)
    model = AcousticModel(norm["mean"], norm["std"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                 betas=tuple(config["betas"]), weight_decay=config["weight_decay"])
    package = Path(__file__).parent
    identity = {"config": config, "data": sha256(data / "summary.json"), "device": str(device),
                "torch": str(torch.__version__), "numpy": str(np.__version__), "threads": threads,
                "sources": {name: sha256(package / name) for name in
                            ("train.py", "acoustic.py", "blocks.py", "data.py", "loss.py", "checkpoints.py")}}
    start = 0
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved.get("format") != "smollertts-training-1" or saved["identity"] != identity:
            raise ValueError("Resume identity differs: data, recipe, source, device or environment changed")
        model.load_state_dict(saved["acoustic"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
        start = saved["step"]
    stop = min(config["steps"], stop_after if stop_after is not None else config["steps"])
    if not start < stop:
        raise ValueError("Stop step must be greater than checkpoint step")
    sampler = Steps(splits["train"], start, stop, seed=config["seed"],
                    max_batch=config["max_batch"], frame_budget=config["frame_budget"])
    loader = DataLoader(sampler, batch_size=None, num_workers=workers, pin_memory=device.type == "cuda",
                        worker_init_fn=init_worker, generator=torch.Generator().manual_seed(config["seed"]),
                        persistent_workers=workers > 0)
    output.mkdir(parents=True)
    (output / "checkpoints").mkdir()
    atomic_json(output / "config.json", config)
    state = {"status": "running", "step": start, "target_step": config["steps"], "stop_step": stop,
             "parameters": sum(p.numel() for p in model.parameters()), "identity": identity,
             "training_conditioning": "recorded prosody", "evaluation_conditioning": "predicted prosody",
             "vocoder_used_in_training_loss": False, "train_recordings": len(splits["train"]),
             "dev_recordings": len(splits["dev"]), "data_summary": summary,
             "resumed_from_sha256": sha256(resume) if resume else None}

    def checkpoint(step):
        saved = {"format": "smollertts-training-1", "step": step, "identity": identity,
                 "acoustic": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": capture_rng()}
        temporary = output / "checkpoint.tmp"
        torch.save(saved, temporary)
        os.replace(temporary, output / "latest.pt")
        destination = output / "checkpoints" / f"step-{step:06d}.pt"
        if not destination.exists():
            temporary = destination.with_suffix(".tmp")
            torch.save(saved, temporary)
            os.replace(temporary, destination)
        atomic_json(output / "status.json", state)

    requested = False
    def halt(*_):
        nonlocal requested
        requested = True
    previous = {sig: signal.signal(sig, halt) for sig in (signal.SIGINT, signal.SIGTERM)}
    step = start
    try:
        checkpoint(start)
        with (output / "metrics.jsonl").open("x", encoding="utf-8", buffering=1) as log:
            for step, values in enumerate(loader, start + 1):
                tick = time.perf_counter()
                ids = values["ids"]
                values = to_device(values, device)
                lr = learning_rate(step, config)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = training_objective(model, values)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite loss")
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                record = {key: float(value) for key, value in metrics.items()}
                record.update(step=step, lr=lr, gradient=float(grad), seconds=time.perf_counter() - tick,
                              examples=len(ids), batch_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
                log.write(json.dumps(record, allow_nan=False) + "\n")
                state.update(step=step, latest=record)
                if step == start + 1 or step % 25 == 0 or step == stop:
                    atomic_json(output / "status.json", state)
                    print(json.dumps({"step": step, "loss": record["total"], "seconds": record["seconds"]}), flush=True)
                if step % config["checkpoint_every"] == 0:
                    checkpoint(step)
                if step == 1000 or step % config["evaluate_every"] == 0 or step == stop:
                    metrics = evaluate(model, splits["dev"], device)
                    atomic_json(output / f"dev-{step:06d}.json", {"step": step, "recordings": len(splits["dev"]),
                                "duration_conditioning": "aligned", "prosody_conditioning": "predicted", "metrics": metrics})
                if requested:
                    break
        state["status"] = "interrupted" if requested else "finished"
        checkpoint(step)
    except BaseException as error:
        state.update(status="failed", error=str(error))
        atomic_json(output / "status.json", state)
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return output / "latest.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/acoustic.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--verify-data", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run(args.data, args.output, config, args.device, args.workers, args.threads, args.resume, args.stop_after, args.verify_data)


if __name__ == "__main__":
    main()
