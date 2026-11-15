"""Score external browser recordings after baseline GPU work has finished."""
import subprocess
import sys
from xiaotts.jobs import wait_success

print('Waiting for the full baseline and its review before comparator ASR', flush=True)
wait_success('runs/full-baseline-review/process.json')
subprocess.run([sys.executable, '-m', 'scripts.score_comparator',
                '--directory', 'artifacts/comparators/sanotts-dev-v1'], check=True)
wait_success('runs/oracle-panel-render/process.json')
subprocess.run([sys.executable, '-m', 'scripts.score_comparator',
                '--directory', 'artifacts/oracle-dev-v1'], check=True)
subprocess.run([sys.executable, '-m', 'scripts.render_alignment_diagnostic',
                '--checkpoint', 'runs/full-baseline-v1/latest.pt',
                '--output', 'artifacts/baseline-alignment-dev-v1'], check=True)
for mode in ('predicted-durations', 'true-durations'):
    directory = 'artifacts/baseline-alignment-dev-v1/'+mode
    subprocess.run([sys.executable, '-m', 'scripts.seal_render', directory], check=True)
    subprocess.run([sys.executable, '-m', 'scripts.score_comparator', '--directory', directory], check=True)
