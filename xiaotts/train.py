"""Bounded from-scratch training with all five languages in every batch.

The debugging run is not the full baseline or an automatic research campaign.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from xiaotts import LANGUAGES
from xiaotts.dsp import Synthesizer, spectral_loss, SAMPLE_RATE, HOP
from xiaotts.fetch import atomic_json
from xiaotts.model import XiaoTTS, ModelConfig, expand_durations
from xiaotts.provenance import snapshot


@lru_cache(maxsize=512)
def read_features(path):
    with np.load(path) as source:
        return {key: torch.from_numpy(np.array(source[key])) for key in source.files}


def collate(rows, device=None):
    items = [read_features(row["features"]) for row in rows]
    batch = len(items)
    phones = max(len(item["tokens"]) for item in items)
    frames = max(len(item["controls"]) for item in items)
    output = {key: torch.zeros(batch, phones, dtype=torch.long) for key in ("tokens", "tones", "stress", "durations")}
    output.update(language=torch.zeros(batch, dtype=torch.long),
                  indices=torch.zeros(batch, frames, dtype=torch.long),
                  positions=torch.zeros(batch, frames, 2), mask=torch.zeros(batch, frames),
                  controls=torch.zeros(batch, frames, 67))
    for b, item in enumerate(items):
        n, t = len(item["tokens"]), len(item["controls"])
        for key in ("tokens", "tones", "stress", "durations"):
            output[key][b, :n] = item[key]
        output["language"][b] = item["language"]
        indices, positions = expand_durations(item["durations"])
        output["indices"][b, :t], output["positions"][b, :t] = indices, positions
        output["mask"][b, :t], output["controls"][b, :t] = 1, item["controls"]
    return ({key: value.to(device) for key, value in output.items()} if device else output), items


class BalancedSteps(Dataset):
    """Reproducible per-step draws, independent of loader worker scheduling."""
    def __init__(self, training, steps, seed, per_language=1):
        self.training, self.steps, self.seed, self.per_language = training, steps, seed, per_language

    def __len__(self):
        return self.steps

    def __getitem__(self, index):
        rng = random.Random(self.seed + index * 1000003)
        rows = [rng.choice(self.training[language]) for _ in range(self.per_language) for language in LANGUAGES]
        batch, items = collate(rows)
        # Do not share cached feature storages across process boundaries: every
        # retained shared tensor holds a file descriptor on Linux. Clone only the
        # waveforms needed by the trainer; the padded batch is already fresh.
        return batch, [dict(frames=len(item["controls"]), waveform=item["waveform"].clone()) for item in items]


def initialize_worker(worker_id):
    torch.set_num_threads(1)


def acoustic_loss(duration, controls, batch):
    target, mask = batch["controls"], batch["mask"]
    tokens = (batch["tokens"] != 0).float()
    def average(value, weighting=mask):
        return (value*weighting).sum()/weighting.sum().clamp_min(1)
    voicing = (target[:, :, 1] > 0).float()
    losses = dict(
        duration=average(F.smooth_l1_loss(duration, batch["durations"].float().log1p(), reduction="none"), tokens),
        pitch=average((controls[:, :, 0]-target[:, :, 0]).abs(), mask*voicing),
        voicing=average(F.binary_cross_entropy_with_logits(controls[:, :, 1], voicing, reduction="none")),
        energy=average(F.smooth_l1_loss(controls[:, :, 2], target[:, :, 2], reduction="none")),
        envelope=average((controls[:, :, 3:]-target[:, :, 3:].clamp(-12, 0)).abs().mean(-1)),
    )
    total = losses["duration"] + 2*losses["pitch"] + losses["voicing"] + losses["energy"] + losses["envelope"]
    return total, losses


@torch.no_grad()
def samples(model, renderer, rows, destination, device):
    destination.mkdir(parents=True, exist_ok=True)
    metadata = []
    model.eval()
    for row in rows:
        batch, items = collate([row], device)
        encoded, duration = model.encode(batch["tokens"], batch["language"], batch["tones"], batch["stress"])
        predicted = duration[0].clamp(0, math.log(101)).expm1().round().long().clamp(1, 100)
        indices, positions = expand_durations(predicted)
        controls = model.decode(encoded, indices[None], positions[None], torch.ones(1, len(indices), device=device))
        audio = renderer(controls, seed=1234)[0].cpu().numpy()
        sf.write(destination / (row["id"]+"-predicted.wav"), audio, SAMPLE_RATE, subtype="FLOAT")
        sf.write(destination / (row["id"]+"-reference.wav"), items[0]["waveform"].numpy(), SAMPLE_RATE, subtype="FLOAT")
        metadata.append(dict(id=row["id"], language=row["language"], text=row["text"],
                             input="aligned phoneme sequence; predicted durations and acoustics",
                             generated_seconds=len(audio)/SAMPLE_RATE, reference_seconds=row["seconds"],
                             peak=float(np.max(np.abs(audio)))))
    atomic_json(destination / "samples.json", metadata)
    model.train()


@torch.no_grad()
def validate(model, rows, device):
    model.eval()
    result = {}
    for language in LANGUAGES:
        selected = [row for row in rows if row["language"] == language][:8]
        batch, _ = collate(selected, device)
        duration, controls = model(batch["tokens"], batch["language"], batch["tones"], batch["stress"],
                                   batch["indices"], batch["positions"], batch["mask"])
        loss, parts = acoustic_loss(duration, controls, batch)
        result[language] = dict(loss=float(loss), **{key: float(value) for key, value in parts.items()})
    model.train()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("data/features/debug"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--spectral-start", type=int, default=200)
    parser.add_argument("--wave-frames", type=int, default=128)
    parser.add_argument("--wave-weight", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=0, help="CPU feature prefetch workers")
    parser.add_argument("--per-language", type=int, default=1, help="Clips per language in every batch")
    parser.add_argument("--wave-pitch-grad", action="store_true",
                        help="Experimental: backpropagate spectral loss through oscillator phase")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new run directory; never overwrite a prior experiment")
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rows = [json.loads(line) for line in (args.features / "index.jsonl").read_text().splitlines()]
    training = {language: [row for row in rows if row["language"] == language and row["split"] == "train"]
                for language in LANGUAGES}
    development = [row for row in rows if row["split"] == "dev"]
    if any(not training[language] or not any(row["language"] == language for row in development)
           for language in LANGUAGES):
        raise ValueError("All five languages require training and development data")
    if args.workers < 0 or args.per_language < 1:
        raise ValueError("Invalid loader worker or batch count")
    loader_options = dict(num_workers=args.workers, batch_size=None, pin_memory=device == "cuda",
                          worker_init_fn=initialize_worker)
    if args.workers:
        loader_options.update(persistent_workers=True, prefetch_factor=2, multiprocessing_context="spawn")
    loader = iter(DataLoader(BalancedSteps(training, args.steps, args.seed, args.per_language), **loader_options))
    config = ModelConfig(width=args.width)
    model = XiaoTTS(config).to(device)
    renderer = Synthesizer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    state = dict(status="running", initialization="random", parameters=model.parameter_count,
                 renderer_parameters=sum(p.numel() for p in renderer.parameters()),
                 languages=list(LANGUAGES), training_clips={key: len(value) for key,value in training.items()},
                 seed=args.seed, model=asdict(config), arguments={key: str(value) if isinstance(value, Path) else value
                                                              for key,value in vars(args).items()},
                 features=json.loads((args.features/"summary.json").read_text()),
                 torch=torch.__version__, device=torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                 started=time.time(), step=0)
    state.update(snapshot(args.output))
    atomic_json(args.output/"run.json", state)
    stop = False
    def request_stop(signum, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    def checkpoint(step):
        temporary = args.output/"checkpoint.tmp"
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=asdict(config),
                        step=step, seed=args.seed, feature_hash=state["features"]["source_hash"]), temporary)
        os.replace(temporary, args.output/"latest.pt")
    started, last_step, data_wait_seconds = time.monotonic(), 0, 0.0
    fixed_samples = [next(row for row in development if row["language"] == language) for language in LANGUAGES]
    model.train()
    with (args.output/"metrics.jsonl").open("a", buffering=1) as metrics:
        for step in range(1, args.steps+1):
            if stop or time.time() >= (args.deadline or float("inf"))-45:
                break
            loading = time.monotonic()
            batch, items = next(loader)
            data_wait_seconds += time.monotonic()-loading
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            scale = min(step/100, 1)*(0.2+0.8*(1+math.cos(math.pi*step/args.steps))/2)
            for group in optimizer.param_groups:
                group["lr"] = args.lr*scale
            optimizer.zero_grad(set_to_none=True)
            duration, controls = model(batch["tokens"], batch["language"], batch["tones"], batch["stress"],
                                       batch["indices"], batch["positions"], batch["mask"])
            loss, parts = acoustic_loss(duration, controls, batch)
            if step >= args.spectral_start:
                length = min(args.wave_frames, min(item["frames"] for item in items))
                predicted_crops, targets = [], []
                for index, item in enumerate(items):
                    start = random.randint(0, item["frames"]-length)
                    crop = controls[index, start:start+length]
                    if step < args.spectral_start+300:
                        crop = torch.cat((batch["controls"][index, start:start+length, :2], crop[:, 2:]), -1)
                    elif not args.wave_pitch_grad:
                        # Phase accumulation amplifies spectral pitch gradients over
                        # long windows. Pitch keeps direct supervised gradients.
                        crop = torch.cat((crop[:, :1].detach(), crop[:, 1:]), -1)
                    predicted_crops.append(crop)
                    targets.append(item["waveform"][start*HOP:(start+length-1)*HOP])
                audio = renderer(torch.stack(predicted_crops))
                wave_loss = spectral_loss(audio, torch.stack(targets).to(device))
                loss = loss + args.wave_weight*wave_loss
                parts["spectral"] = wave_loss
            if not torch.isfinite(loss):
                checkpoint(last_step)
                raise ValueError(f"Non-finite training loss at step {step}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            last_step = step
            if step % 25 == 0 or step == 1:
                record = dict(step=step, loss=float(loss), grad_norm=float(norm),
                              elapsed=time.monotonic()-started,
                              data_wait_seconds=data_wait_seconds,
                              **{key: float(value) for key,value in parts.items()})
                metrics.write(json.dumps(record)+"\n")
                print(json.dumps(record), flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                evaluation = validate(model, development, device)
                atomic_json(args.output/f"dev-{step:06d}.json", evaluation)
                samples(model, renderer, fixed_samples, args.output/f"samples-{step:06d}", device)
                checkpoint(step)
                state.update(step=step, elapsed=time.monotonic()-started, latest_dev=evaluation)
                atomic_json(args.output/"run.json", state)
        checkpoint(last_step)
        state.update(status="finished" if last_step == args.steps else "stopped", step=last_step,
                     elapsed=time.monotonic()-started, finished=time.time(),
                     data_wait_seconds=data_wait_seconds,
                     peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else 0)
        atomic_json(args.output/"run.json", state)


if __name__ == "__main__":
    main()
