"""Compare known-phone inference with predicted versus true development durations.

Both modes receive alignment phones from the reference recording. They are
diagnostics, not actual-text TTS results and never eligible for promotion.
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch
from xiaotts.benchmark import digest, evaluator_sources
from xiaotts.dsp import Synthesizer, SAMPLE_RATE
from xiaotts.fetch import atomic_json
from xiaotts.model import XiaoTTS, ModelConfig, expand_durations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, default=Path('evaluation/dev-v1.json'))
    parser.add_argument('--features', type=Path, default=Path('data/features/full'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous diagnostics')
    panel = json.loads(args.panel.read_text())
    if panel['sources'] != evaluator_sources() or any(row['split'] != 'dev' for row in panel['rows']):
        raise ValueError('Changed protocol or non-development input')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    torch.set_num_threads(4)
    model = XiaoTTS(ModelConfig(**state['config'])).eval()
    model.load_state_dict(state['model'])
    renderer = Synthesizer().eval()
    started = time.monotonic()
    for mode in ('predicted-durations', 'true-durations'):
        destination = args.output/mode
        destination.mkdir(parents=True)
        report = dict(status='running', system='Known-phone '+mode+' diagnostic',
                      panel_sha256=digest(args.panel), samples=[], gpu_disabled=True,
                      source=dict(purpose='Reference-alignment input; not actual-text TTS',
                                  checkpoint_sha256=digest(args.checkpoint), step=state['step'],
                                  renderer_sha256=digest('xiaotts/dsp.py'),
                                  script_sha256=digest(Path(__file__))),
                      input='Aligned reference phonemes with '+mode)
        for row in panel['rows']:
            item = dict(id=row['id'], language=row['language'], text=row['text'], reference_seconds=row['seconds'])
            try:
                cached = args.features/(row['id']+'.npz')
                if not cached.exists():
                    raise ValueError('No accepted reference alignment for this diagnostic')
                with np.load(cached) as data:
                    inputs = {key:torch.from_numpy(np.array(data[key])).reshape(1, -1)
                              for key in ('tokens', 'tones', 'stress')}
                    inputs['language'] = torch.tensor([int(data['language'])])
                    true_duration = torch.from_numpy(np.array(data['durations']))
                before = time.monotonic()
                with torch.no_grad():
                    encoded, duration = model.encode(**inputs)
                    durations = true_duration if mode == 'true-durations' else duration[0].clamp(0, math.log(101)).expm1().round().long().clamp(1, 100)
                    indices, positions = expand_durations(durations)
                    controls = model.decode(encoded, indices[None], positions[None], torch.ones(1, len(indices)))
                    wave = renderer(controls, seed=1234)[0].numpy()
                if not np.isfinite(wave).all():
                    raise ValueError('Non-finite generated waveform')
                path = destination/f"{row['id']}.wav"
                sf.write(path, wave, SAMPLE_RATE, subtype='FLOAT')
                item.update(rate=SAMPLE_RATE, generated_seconds=len(wave)/SAMPLE_RATE,
                            render_seconds=time.monotonic()-before, wave_sha256=digest(path))
            except (ValueError, OSError) as error:
                item['failure'] = str(error)
            report['samples'].append(item)
            atomic_json(destination/'render.json', report)
        report.update(status='finished', seconds=time.monotonic()-started)
        atomic_json(destination/'render.json', report)
        print(json.dumps(dict(mode=mode, clips=len(report['samples']), failures=sum('failure' in item for item in report['samples']))), flush=True)


if __name__ == '__main__':
    main()
