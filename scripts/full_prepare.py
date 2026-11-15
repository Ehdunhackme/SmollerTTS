"""Wait on a verified lexicon process, then prepare the full frozen train/dev data."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

path = Path("runs/lexicon-augmentation/process.json")
while True:
    state = json.loads(path.read_text())
    if state["status"] != "running":
        if state["status"] != "finished" or state["exit_code"] != 0:
            raise SystemExit("Full lexicon process failed; inspect its output before continuing")
        break
    try:
        os.kill(state["pid"], 0)
    except ProcessLookupError:
        time.sleep(2)
        current = json.loads(path.read_text())
        if current["status"] == "running":
            raise SystemExit("Recorded lexicon process is missing; no blind restart")
    time.sleep(5)
subprocess.run([sys.executable, "-m", "xiaotts.alignment", "--limit", "0", "--label", "full",
                "--lexicons", "data/lexicons", "--parallel-languages", "2", "--jobs", "6"], check=True)
subprocess.run([sys.executable, "-m", "xiaotts.features", "--label", "full", "--workers", "8"], check=True)
