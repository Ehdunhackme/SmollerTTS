"""Exercise an actual interrupted trainer, including dropout and waveform noise."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from xiaotts import LANGUAGES


def test_recovery_matches_uninterrupted_training(tmp_path):
    features = tmp_path/'features'
    features.mkdir()
    rng = np.random.default_rng(42)
    rows = []
    for language_id, language in enumerate(LANGUAGES):
        for clip in range(3):
            controls = rng.normal(0, 0.1, (24, 67)).astype(np.float32)
            controls[:, 0] = 5.2
            controls[:, 1] = 8
            controls[:, 2] = -3.5
            controls[:, 3:] -= 2
            path = features/f'{language}-{clip}.npz'
            np.savez(path, tokens=np.arange(3+clip, 9+clip, dtype=np.int64),
                     tones=np.arange(6, dtype=np.int64), stress=np.arange(6, dtype=np.int64)%3,
                     durations=np.full(6, 4, dtype=np.int64), controls=controls,
                     waveform=(rng.normal(0, 0.02, 23*256)).astype(np.float32), language=language_id)
            rows.append(dict(id=f'{language}-{clip}', language=language, features=str(path),
                             split='dev' if clip == 2 else 'train', text='recovery test', seconds=23*256/22050))
    (features/'index.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
    (features/'summary.json').write_text(json.dumps(dict(source_hash='test', dataset='test')))
    (features/'vocabulary.json').write_text('{}')
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '-1'}
    root = Path(__file__).resolve().parents[1]
    common = [sys.executable, '-m', 'xiaotts.train', '--features', str(features), '--steps', '8',
              '--width', '16', '--spectral-start', '1', '--wave-frames', '16', '--eval-every', '8', '--device', 'cpu']
    for name, extra in [('full', []), ('partial', ['--stop-after', '4']),
                        ('resumed', ['--resume', str(tmp_path/'partial/latest.pt')])]:
        result = subprocess.run([*common, '--output', str(tmp_path/name), *extra], env=environment,
                                cwd=root, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout+'\n'+result.stderr
    full = torch.load(tmp_path/'full/latest.pt', weights_only=True, map_location='cpu')
    resumed = torch.load(tmp_path/'resumed/latest.pt', weights_only=True, map_location='cpu')
    for key in full['model']:
        torch.testing.assert_close(full['model'][key], resumed['model'][key], rtol=0, atol=0)
    for parameter in full['optimizer']['state']:
        for key, expected in full['optimizer']['state'][parameter].items():
            torch.testing.assert_close(expected, resumed['optimizer']['state'][parameter][key], rtol=0, atol=0)
    assert full['step'] == resumed['step'] == 8
    torch.testing.assert_close(full['rng']['torch'], resumed['rng']['torch'])
    record = json.loads((tmp_path/'resumed/run.json').read_text())
    assert record['inherited_steps'] == 4 and record['inherited_training_seconds'] > 0
    assert record['elapsed'] >= record['inherited_training_seconds']
