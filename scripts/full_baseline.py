"""Wait for the exact full-data job, audit its output, and start the baseline.

This is one recorded baseline run, not an autonomous architecture sweep.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from xiaotts import LANGUAGES
from xiaotts.fetch import atomic_json


def main():
    process_path = Path('runs/full-data-preparation/process.json')
    initial = json.loads(process_path.read_text())
    expected = (initial['pid'], initial['started'])
    print('Waiting for full-data-preparation', expected, flush=True)
    while True:
        state = json.loads(process_path.read_text())
        if (state['pid'], state['started']) != expected:
            raise RuntimeError('Preparation process identity changed')
        if state['status'] != 'running':
            if state['status'] != 'finished' or state['exit_code'] != 0:
                raise RuntimeError('Full data preparation failed; inspect before restarting')
            break
        try:
            os.kill(state['pid'], 0)
        except ProcessLookupError:
            time.sleep(2)
            if json.loads(process_path.read_text())['status'] == 'running':
                raise RuntimeError('Preparation process disappeared without completion')
        time.sleep(10)
    features = Path('data/features/full')
    summary = json.loads((features/'summary.json').read_text())
    rows = [json.loads(line) for line in (features/'index.jsonl').read_text().splitlines()]
    dataset = json.loads(Path('data/manifests/dataset.json').read_text())
    if summary['dataset'] != dataset['sha256'] or len(rows) != summary['accepted']:
        raise ValueError('Feature provenance mismatch')
    if any(row['split'] not in ('train', 'dev') for row in rows):
        raise ValueError('Final test data must not enter baseline preparation')
    report = {}
    for language in LANGUAGES:
        report[language] = {}
        for split in ('train', 'dev'):
            selected = [row for row in rows if row['language'] == language and row['split'] == split]
            if len(selected) < (1000 if split == 'train' else 100):
                raise ValueError(f'Too few usable {language}/{split} recordings; inspect failures')
            if any(not Path(row['features']).is_file() for row in selected):
                raise ValueError('Missing cached feature file')
            report[language][split] = dict(clips=len(selected), hours=sum(row['seconds'] for row in selected)/3600,
                                          frames=sum(row['frames'] for row in selected),
                                          tone_frames=sum(row['tone_frames'] for row in selected))
    atomic_json(Path('artifacts/full-feature-audit.json'), dict(summary=summary, per_language=report))
    print(json.dumps(report, indent=2), flush=True)
    subprocess.run([sys.executable, '-m', 'xiaotts.train', '--features', str(features),
                    '--output', 'runs/full-baseline-v1', '--steps', '50000', '--eval-every', '2500',
                    '--workers', '4', '--per-language', '1'], check=True)


if __name__ == '__main__':
    main()
