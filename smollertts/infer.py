"""Synthesize native listening samples with predicted durations and prosody."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .assets import load_vocoder
from .blocks import duration_indices
from .checkpoints import atomic_json, load_acoustic, sha256


def predicted_durations(tokens, duration):
    values = (duration.clamp(0, math.log(101)).expm1() + .5).floor().long()
    return torch.where(tokens == 1, values.clamp(0, 60), values.clamp(1, 100))


@torch.no_grad()
def synthesize(model, vocoder, tokens, stress, durations=None):
    if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape != stress.shape or tokens.shape[1] > 220:
        raise ValueError("Expected one chunk of at most 220 phonemes")
    if ((tokens <= 0) | (tokens >= 132) | (tokens == 2) | (tokens == 45)).any() or ((stress < 0) | (stress > 2)).any():
        raise ValueError("Unsupported phone or stress")
    encoded, duration, phone = model.encode(tokens, stress)
    durations = predicted_durations(tokens[0], duration[0]) if durations is None else durations
    if not 2 <= int(durations.sum()) <= 6000:
        raise ValueError("Predicted duration outside synthesis limits")
    indices, position = duration_indices(durations)
    mel, _ = model.native_decode(encoded, phone, indices[None], position[None], torch.ones(1, len(indices), 1, device=encoded.device))
    return vocoder(mel.transpose(1, 2))[0, 0], durations


def render(checkpoint, vocoder_path, prompts, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new listening output directory")
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    model, saved = load_acoustic(checkpoint)
    model.eval()
    vocoder = load_vocoder(vocoder_path)
    output.mkdir(parents=True)
    records = []
    for index, row in enumerate(json.loads(Path(prompts).read_text(encoding="utf-8"))["samples"]):
        pieces = []
        for chunk in row["chunks"]:
            wave, _ = synthesize(model, vocoder, torch.tensor([chunk["tokens"]]), torch.tensor([chunk["stress"]]))
            pieces.append(wave.numpy())
        wave = np.concatenate(pieces)
        if not np.isfinite(wave).all():
            raise ValueError("Non-finite waveform")
        path = output / f"sample-{index + 1:02d}.wav"
        sf.write(path, wave, 22050, subtype="PCM_16")
        records.append({"text": row["text"], "file": path.name, "sha256": sha256(path), "seconds": len(wave) / 22050})
    atomic_json(output / "samples.json", {"step": saved["step"], "checkpoint_sha256": sha256(checkpoint),
                "conditioning": "predicted durations and predicted prosody", "samples": records,
                "note": "Native baseline; browser punctuation and user delivery controls are not applied."})
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocoder", type=Path, default=Path("assets/vocoder"))
    parser.add_argument("--prompts", type=Path, default=Path("configs/listening-prompts.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.checkpoint, args.vocoder, args.prompts, args.output), indent=2))
