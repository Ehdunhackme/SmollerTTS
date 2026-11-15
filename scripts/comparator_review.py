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
