"""Run a bounded process group, recording exact PID, command, deadline and exit.

This limits computation only, not provider billing. It never kills unrelated jobs.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    deadline = args.deadline or float("inf")
    if args.timeout_seconds:
        deadline = min(deadline, time.time()+args.timeout_seconds)
    if not command or time.time() >= deadline:
        raise SystemExit("Missing command or session deadline already reached")
    root = Path("runs") / args.name
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "process.json"
    if state_path.exists():
        raise SystemExit("Run directory already exists; inspect previous process before restarting")
    state = dict(command=command, started=time.time(), deadline=deadline if deadline != float("inf") else None,
                 provider_billing_stop=False, status="starting")
    def save():
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(state_path)
    with (root / "output.log").open("a", buffering=1) as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        state.update(pid=process.pid, supervisor_pid=os.getpid(), status="running")
        save()
        def terminate(signum=None, frame=None):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        signal.signal(signal.SIGTERM, terminate)
        signal.signal(signal.SIGINT, terminate)
        while process.poll() is None and time.time() < deadline:
            time.sleep(1)
        timed_out = process.poll() is None
        if timed_out:
            terminate()
        state.update(status="deadline" if timed_out else "finished", exit_code=process.wait(),
                     finished=time.time())
        save()
        raise SystemExit(state["exit_code"] or (124 if timed_out else 0))


if __name__ == "__main__":
    main()
