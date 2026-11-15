"""Fit per-recording controls for all five languages to test renderer capacity.

This is an oracle reconstruction diagnostic, not a trained text-to-speech model.
It does not promote candidates or access final-test recordings.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pyworld
import soundfile as sf
import torch

from xiaotts import LANGUAGES
from xiaotts.acoustics import read_audio, extract_controls
from xiaotts.dsp import Synthesizer, spectral_loss, SAMPLE_RATE, HOP
from xiaotts.fetch import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/renderer-gate"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seconds", type=float, default=4)
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(17)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    renderer = Synthesizer().to(device)
    assert sum(p.numel() for p in renderer.parameters()) == 0
    rows = [json.loads(line) for line in (args.root / "manifests/train.jsonl").read_text().splitlines()]
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(diagnostic="oracle acoustic reconstruction, not TTS", device=device,
                  renderer_parameters=0, listening_review="pending", samples=[])
    for language in LANGUAGES:
        row = next(row for row in rows if row["language"] == language and 2 <= row["seconds"] <= args.seconds)
        controls, wave, world = extract_controls(read_audio(row["audio"], args.seconds))
        target = torch.from_numpy(wave)[None].to(device)
        initial = torch.from_numpy(controls)[None].to(device)
        filters = torch.nn.Parameter(initial[:, :, 2:].clone())
        optimizer = torch.optim.Adam([filters], lr=0.03)
        def render(seed=17):
            return renderer(torch.cat((initial[:, :, :2], filters), -1), seed=seed)
        with torch.no_grad():
            first = render()
            first_loss = spectral_loss(first, target).item()
        begin = time.monotonic()
        last_loss, steps_done = first_loss, 0
        for step in range(args.steps):
            if time.time() >= (args.deadline or float("inf"))-30:
                raise RuntimeError("Session deadline reached during renderer diagnostic")
            optimizer.zero_grad()
            loss = spectral_loss(render(seed=step), target)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite reconstruction loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([filters], 5)
            optimizer.step()
            last_loss, steps_done = loss.item(), step+1
            if step % 25 == 0:
                print(language, step, round(last_loss, 4), flush=True)
        with torch.no_grad():
            fitted = render(seed=1234)
            fitted_loss = spectral_loss(fitted, target).item()
        for label, audio in (("reference", wave), ("initial", first[0].cpu().numpy()),
                              ("fitted", fitted[0].cpu().numpy()),
                              ("world-reference", pyworld.synthesize(*world, SAMPLE_RATE,
                               frame_period=HOP/SAMPLE_RATE*1000))):
            sf.write(args.output / f"{language}-{label}.wav", audio, SAMPLE_RATE, subtype="FLOAT")
        np.savez_compressed(args.output / f"{language}-controls.npz",
                            controls=torch.cat((initial[:, :, :2], filters), -1).detach()[0].cpu().numpy())
        report["samples"].append(dict(language=language, id=row["id"], text=row["text"],
                                       text_is_truncated=len(wave)/SAMPLE_RATE < row["seconds"]-0.1,
                                       initial_loss=first_loss, fitted_loss=fitted_loss,
                                       steps=steps_done, seconds=time.monotonic()-begin,
                                       finite=bool(torch.isfinite(fitted).all()),
                                       peak=float(fitted.abs().max())))
        atomic_json(args.output / "report.json", report)
        print(report["samples"][-1], flush=True)


if __name__ == "__main__":
    main()
