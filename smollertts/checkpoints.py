"""Portable training state and atomic writes. Checkpoints contain no Python modules."""
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def capture_rng():
    algorithm, keys, position, has_gauss, cached = np.random.get_state()
    return {"python": random.getstate(), "numpy": (algorithm, keys.tolist(), position, has_gauss, cached),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(state):
    random.setstate(state["python"])
    algorithm, keys, position, has_gauss, cached = state["numpy"]
    np.random.set_state((algorithm, np.array(keys, dtype=np.uint32), position, has_gauss, cached))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the original CUDA device count")
        torch.cuda.set_rng_state_all(state["cuda"])


def load_acoustic(path):
    from .acoustic import AcousticModel
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = AcousticModel()
    model.load_state_dict(checkpoint["acoustic"], strict=True)
    return model, checkpoint


def validate_vocabulary(vocabulary):
    canonical = json.dumps(vocabulary, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != "7f62e1365935c8b19047018f9e71b92269e093b9ebb0cc7eb9560e0f747ac8dd":
        raise ValueError("Vocabulary differs from the released phone IDs; changing it requires a frontend port")


def load_dataset(directory, verify_features=False):
    """Resolve relative paths without putting machine-specific paths in identity."""
    directory = Path(directory).resolve()
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    required = {"train.jsonl", "dev.jsonl", "test.jsonl", "normalization.json", "vocabulary.json"}
    if not required.issubset(summary["manifests"]):
        raise ValueError("Dataset summary must hash every split, normalization and vocabulary")
    for name, expected in summary["manifests"].items():
        if Path(name).name != name or sha256(directory / name) != expected:
            raise ValueError(f"Dataset manifest changed: {name}")
    if not summary.get("synthetic_fixture", False):
        validate_vocabulary(json.loads((directory / "vocabulary.json").read_text(encoding="utf-8")))
    norm = json.loads((directory / "normalization.json").read_text(encoding="utf-8"))
    for key in ("mean", "std"):
        values = np.asarray(norm[key])
        if values.shape != (80,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid mel normalization: {key}")
    if np.any(np.asarray(norm["std"]) <= 0):
        raise ValueError("Mel standard deviations must be positive")
    splits, seen = {}, set()
    for split in ("train", "dev", "test"):
        rows = [json.loads(line) for line in (directory / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            if row["split"] != split or row["id"] in seen:
                raise ValueError(f"Duplicate or cross-split utterance: {row['id']}")
            seen.add(row["id"])
            if "features" in row:
                path = Path(row["features"])
                row["features"] = str(path if path.is_absolute() else directory / path)
                if verify_features and sha256(row["features"]) != row["feature_sha256"]:
                    raise ValueError(f"Feature changed: {row['id']}")
            elif split != "test":
                raise ValueError(f"Missing features: {row['id']}")
        splits[split] = rows
    if not splits["train"] or not splits["dev"]:
        raise ValueError("Training and development splits must both be nonempty")
    return splits, norm, summary
