"""Evaluate and export the baseline after its exact supervised job succeeds."""
import json
from pathlib import Path
import subprocess
import sys
from xiaotts.jobs import wait_success
from xiaotts.fetch import atomic_json

print('Waiting for the full baseline pipeline', flush=True)
wait_success('runs/full-baseline-pipeline/process.json')
run = Path('runs/full-baseline-v1')
record = json.loads((run/'run.json').read_text())
if record['status'] != 'finished' or record['arguments'].get('train_clips_per_language', 0):
    raise RuntimeError('The full baseline is not complete')
subprocess.run([sys.executable, '-m', 'xiaotts.benchmark', 'evaluate', '--checkpoint', str(run/'latest.pt'),
                '--output', 'runs/full-baseline-dev-v1'], check=True)
subprocess.run([sys.executable, '-m', 'xiaotts.export', '--checkpoint', str(run/'latest.pt'),
                '--output', 'artifacts/full-baseline-export'], check=True)
atomic_json(Path('artifacts/full-baseline-review.json'), dict(
    training=record,
    evaluation=json.loads(Path('runs/full-baseline-dev-v1/report.json').read_text()),
    export=json.loads(Path('artifacts/full-baseline-export/export.json').read_text()),
    purpose='Baseline evidence for choosing the first research trial; not a final model qualification'))
print('Baseline evaluation and native export finished', flush=True)
