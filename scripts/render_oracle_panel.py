"""Render true acoustic controls for the fixed dev panel to locate the bottleneck.

This deliberately uses the reference recordings, so it is reconstruction evidence,
never TTS performance. It cannot qualify as a research candidate or final model.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch
from xiaotts.acoustics import extract_controls, read_audio
from xiaotts.benchmark import digest, evaluator_sources
from xiaotts.dsp import Synthesizer, SAMPLE_RATE
from xiaotts.fetch import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, default=Path('evaluation/dev-v1.json'))
    parser.add_argument('--features', type=Path, default=Path('data/features/full'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/oracle-dev-v1'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing diagnostics')
    panel = json.loads(args.panel.read_text())
    if panel['sources'] != evaluator_sources() or any(row['split'] != 'dev' for row in panel['rows']):
        raise ValueError('Changed protocol or non-development input')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    renderer = Synthesizer().eval()
    report = dict(status='running', system='True-control XiaoTTS renderer reconstruction',
                  panel_sha256=digest(args.panel), samples=[],
                  source=dict(purpose='Reference-audio reconstruction; not text-to-speech',
                              renderer_sha256=digest('xiaotts/dsp.py'),
                              acoustic_analysis_sha256=digest('xiaotts/acoustics.py'),
                              feature_summary_sha256=digest(args.features/'summary.json')),
                  input='Accurate acoustic controls extracted from original development recordings',
                  gpu_disabled=True, learned_parameters=0)
    started = time.monotonic()
    for row in panel['rows']:
        item = dict(id=row['id'], language=row['language'], text=row['text'], reference_seconds=row['seconds'])
        try:
            cached = args.features/(row['id']+'.npz')
            if cached.exists():
                with np.load(cached) as data:
                    controls = np.array(data['controls'])
                item['controls'] = 'cached reference features'
            else:
                # Alignment failure is irrelevant to this audio-only diagnostic.
                # Analyze the recording without supplying phones or filtering it out.
                controls, _, _ = extract_controls(read_audio(row['audio']))
                item['controls'] = 'reference features extracted without alignment'
            before = time.monotonic()
            with torch.no_grad():
                wave = renderer(torch.from_numpy(controls)[None], seed=1234)[0].numpy()
            if not np.isfinite(wave).all():
                raise ValueError('Non-finite reconstruction')
            path = args.output/f"{row['id']}.wav"
            sf.write(path, wave, SAMPLE_RATE, subtype='FLOAT')
            item.update(rate=SAMPLE_RATE, render_seconds=time.monotonic()-before,
                        generated_seconds=len(wave)/SAMPLE_RATE, wave_sha256=digest(path))
        except (ValueError, OSError) as error:
            item['failure'] = str(error)
        report['samples'].append(item)
        atomic_json(args.output/'render.json', report)
        print(json.dumps({key:item[key] for key in ('id', 'generated_seconds', 'failure') if key in item}), flush=True)
    report.update(status='finished', seconds=time.monotonic()-started)
    atomic_json(args.output/'render.json', report)


if __name__ == '__main__':
    main()
