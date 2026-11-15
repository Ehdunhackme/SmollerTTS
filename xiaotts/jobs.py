"""Wait on a particular supervised job without treating a stale record as live."""
import json
import os
from pathlib import Path
import time


def wait_success(path):
    path = Path(path)
    initial = json.loads(path.read_text())
    identity = (initial['pid'], initial['started'])
    while True:
        state = json.loads(path.read_text())
        if (state['pid'], state['started']) != identity:
            raise RuntimeError('Job identity changed while waiting')
        if state['status'] != 'running':
            if state['status'] != 'finished' or state['exit_code'] != 0:
                raise RuntimeError(f'Dependent job did not succeed: {path}')
            return state
        try:
            os.kill(state['pid'], 0)
            command_file = Path('/proc')/str(state['pid'])/'cmdline'
            if command_file.exists():
                actual = command_file.read_bytes().rstrip(b'\0').decode().split('\0')
                if actual == ['']:
                    # A just-exited zombie can precede the supervisor's terminal
                    # record by one polling interval.
                    time.sleep(2)
                    continue
                if actual[1:] != state['command'][1:]:
                    raise RuntimeError('Recorded PID belongs to a different command')
        except (ProcessLookupError, FileNotFoundError):
            time.sleep(2)
            if json.loads(path.read_text())['status'] == 'running':
                raise RuntimeError('Recorded job disappeared without completion')
        time.sleep(10)
