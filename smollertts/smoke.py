"""Exercise training on tiny synthetic features. This is not a speech-quality test."""
import argparse
import json
from pathlib import Path

import numpy as np

from .checkpoints import atomic_json, sha256
from .train import run


def make_data(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "features").mkdir()
    rng = np.random.default_rng(2718)
    rows, training_mels = [], []
    for index in range(6):
        frames = 24 + 4 * index
        durations = np.array([2, 5, frames - 14, 5, 2], dtype=np.int64)
        mel = rng.normal(-4, .6, (frames, 80)).astype(np.float32)
        prosody = rng.normal(0, .2, (frames, 3)).astype(np.float32)
        prosody[:, 1] = rng.integers(0, 2, frames)
        path = output / "features" / f"fixture-{index}.npz"
        np.savez(path, tokens=np.array([1, 22, 12, 69, 1], dtype=np.int64),
                 stress=np.array([0, 1, 0, 0, 0], dtype=np.int64), durations=durations, mel=mel, prosody=prosody)
        split = "train" if index < 4 else "dev"
        rows.append({"id": f"fixture-{index}", "split": split, "frames": frames,
                     "features": path.relative_to(output).as_posix(), "feature_sha256": sha256(path)})
        if split == "train":
            training_mels.append(mel)
    mel = np.concatenate(training_mels).astype(np.float64)
    atomic_json(output / "normalization.json", {"mean": mel.mean(0).tolist(), "std": mel.std(0).tolist(), "source": "train only"})
    # The smoke path never downloads data or needs the pronunciation frontend.
    atomic_json(output / "vocabulary.json", {"fixture": True, "size": 132})
    for split in ("train", "dev", "test"):
        (output / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows if row["split"] == split), encoding="utf-8")
    atomic_json(output / "summary.json", {"synthetic_fixture": True,
                "manifests": {name: sha256(output / name) for name in
                              ("train.jsonl", "dev.jsonl", "test.jsonl", "normalization.json", "vocabulary.json")}})
    return output


def config():
    return {"seed": 2718, "steps": 4, "learning_rate": .0005, "warmup_steps": 2,
            "decay_half_life": 20000, "betas": [.9, .98], "weight_decay": .01,
            "gradient_clip": 5., "max_batch": 2, "frame_budget": 256,
            "checkpoint_every": 2, "evaluate_every": 2, "tf32": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new smoke-test directory")
    data = make_data(args.output / "data")
    checkpoint = run(data, args.output / "train", config(), threads=1, verify_data=True)
    print(f"CPU training smoke test passed: {checkpoint}. Synthetic features; not a usable voice.")


if __name__ == "__main__":
    main()
