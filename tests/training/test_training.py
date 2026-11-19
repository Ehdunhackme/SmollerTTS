import json

import numpy as np
import pytest
import torch

from smollertts.acoustic import AcousticModel
from smollertts.checkpoints import atomic_json, load_acoustic, load_dataset, sha256, validate_vocabulary
from smollertts.data import Steps, batch, item
from smollertts.loss import training_objective
from smollertts.smoke import config, make_data
from smollertts.train import run


def assert_state_equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_state_equal(a, b)
    else:
        assert left == right


def test_exact_resume_and_recipe_guard(tmp_path):
    data = make_data(tmp_path / "data")
    recipe = config()
    full = run(data, tmp_path / "full", recipe, threads=1, verify_data=True)
    first = run(data, tmp_path / "first", recipe, threads=1, stop_after=2)
    resumed = run(data, tmp_path / "resumed", recipe, threads=1, resume=first)
    full_state = torch.load(full, weights_only=True)
    resumed_state = torch.load(resumed, weights_only=True)
    for name in ("step", "acoustic", "optimizer", "rng", "identity"):
        assert_state_equal(full_state[name], resumed_state[name])
    full_metrics = [json.loads(line) for line in (full.parent / "metrics.jsonl").read_text().splitlines()][2:]
    resumed_metrics = [json.loads(line) for line in (resumed.parent / "metrics.jsonl").read_text().splitlines()]
    for a, b in zip(full_metrics, resumed_metrics):
        for key in ("lr", "total", "gradient", "batch_sha256"):
            assert a[key] == b[key]
    model, _ = load_acoustic(resumed)
    assert sum(p.numel() for p in model.parameters()) == 1_882_807
    with pytest.raises(ValueError, match="Resume identity"):
        run(data, tmp_path / "bad-resume", {**recipe, "learning_rate": .001}, threads=1, resume=first)
    with pytest.raises(FileExistsError):
        run(data, tmp_path / "full", recipe, threads=1)


def test_training_reaches_every_learned_branch(tmp_path):
    torch.set_num_threads(1)
    data = make_data(tmp_path / "data")
    splits, norm, _ = load_dataset(data)
    model = AcousticModel(norm["mean"], norm["std"])
    values = batch(splits["train"][:2])
    loss, _ = training_objective(model, values)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
    for prefix in ("duration", "phone_prosody", "mel_out", "prosody_residual", "condition"):
        gradients = [p.grad.abs().sum() for name, p in model.named_parameters() if prefix in name]
        assert gradients and sum(gradients) > 0, prefix


def test_padding_targets_do_not_change_objective(tmp_path):
    torch.set_num_threads(1)
    splits, norm, _ = load_dataset(make_data(tmp_path / "data"))
    model = AcousticModel(norm["mean"], norm["std"])
    values = batch(splits["train"][:2])
    original, _ = training_objective(model, values)
    # Only target mel values change. Inputs and real target frames are identical.
    values["mel"][values["frame_mask"][..., 0] == 0] = 10000
    changed, _ = training_objective(model, values)
    torch.testing.assert_close(original, changed, rtol=0, atol=0)


def test_sampler_keeps_step_sequence_and_train_only(tmp_path):
    splits, _, _ = load_dataset(make_data(tmp_path / "data"))
    full = Steps(splits["train"], 0, 10)
    resumed = Steps(splits["train"], 5, 10)
    assert [full[i]["ids"] for i in range(5, 10)] == [resumed[i]["ids"] for i in range(5)]
    with pytest.raises(ValueError, match="Only training"):
        Steps(splits["dev"], 0, 5)


def test_manifest_and_features_detect_modification(tmp_path):
    data = make_data(tmp_path / "data")
    path = data / "features/fixture-0.npz"
    with np.load(path) as saved:
        arrays = dict(saved)
    arrays["durations"][0] += 1
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="Feature changed"):
        load_dataset(data, verify_features=True)
    with pytest.raises(ValueError, match="cover every mel frame"):
        item(str(path))
    (data / "train.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest changed"):
        load_dataset(data)


def test_split_overlap_rejected(tmp_path):
    data = make_data(tmp_path / "data")
    row = json.loads((data / "dev.jsonl").read_text().splitlines()[0])
    row["id"] = "fixture-0"
    (data / "dev.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    summary = json.loads((data / "summary.json").read_text())
    summary["manifests"]["dev.jsonl"] = sha256(data / "dev.jsonl")
    atomic_json(data / "summary.json", summary)
    with pytest.raises(ValueError, match="cross-split"):
        load_dataset(data)


def test_changed_phone_ids_are_not_browser_compatible():
    from pathlib import Path
    vocabulary = json.loads((Path(__file__).resolve().parents[2] / "configs/vocabulary.json").read_text(encoding="utf-8"))
    validate_vocabulary(vocabulary)
    vocabulary["a"], vocabulary["b"] = vocabulary["b"], vocabulary["a"]
    with pytest.raises(ValueError, match="frontend port"):
        validate_vocabulary(vocabulary)
