"""Synthesize actual text using one shared trained model and the dictionary frontend."""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import soundfile as sf
import torch
from xiaotts import LANGUAGES
from xiaotts.dsp import Synthesizer, SAMPLE_RATE
from xiaotts.fetch import atomic_json
from xiaotts.frontend import Frontend
from xiaotts.model import XiaoTTS, ModelConfig, expand_durations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--language", required=True, choices=LANGUAGES)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = XiaoTTS(ModelConfig(**state["config"])).eval().to(args.device)
    model.load_state_dict(state["model"])
    renderer = Synthesizer().to(args.device)
    frontend = Frontend(args.data_root, args.language, json.loads(args.vocabulary.read_text()))
    start = time.monotonic()
    inputs, pronunciations = frontend(args.text)
    inputs = {key: value.to(args.device) for key, value in inputs.items()}
    with torch.no_grad():
        encoded, duration = model.encode(**inputs)
        durations = duration[0].clamp(0, math.log(101)).expm1().round().long().clamp(1, 100)
        indices, positions = expand_durations(durations)
        controls = model.decode(encoded, indices[None], positions[None],
                                torch.ones(1, len(indices), device=args.device))
        waveform = renderer(controls, seed=1234)[0].cpu().numpy()
    elapsed = time.monotonic()-start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, waveform, SAMPLE_RATE, subtype="FLOAT")
    atomic_json(args.output.with_suffix(".json"), dict(text=args.text, language=args.language,
                checkpoint=str(args.checkpoint), parameters=model.parameter_count, device=args.device,
                pronunciations=pronunciations, synthesis_seconds=elapsed, audio_seconds=len(waveform)/SAMPLE_RATE,
                real_time_factor=elapsed/(len(waveform)/SAMPLE_RATE), peak=float(np.max(np.abs(waveform))),
                initialization_excluded=True, browser_measurement=False))
    print(args.output.with_suffix(".json").read_text())


if __name__ == "__main__":
    main()
