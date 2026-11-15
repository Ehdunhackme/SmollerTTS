"""Continue preparation only after the known downloader finishes extracting."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--deadline", type=float)
parser.add_argument("--download-pid", type=int, required=True)
args = parser.parse_args()
deadline = args.deadline or float("inf")
while not all((Path("data/provenance") / f"{language}.json").exists()
              for language in ("en", "zh", "es", "fr", "de")):
    if time.time() > deadline-300:
        raise SystemExit("Deadline approaching; no new preparation work started")
    process = Path(f"/proc/{args.download_pid}/cmdline")
    if not process.exists() or b"xiaotts.fetch" not in process.read_bytes():
        raise SystemExit("Downloader is no longer running and sources are incomplete; inspect logs/fetch.log")
    time.sleep(10)
subprocess.run([sys.executable, "-m", "xiaotts.manifest"], check=True)
subprocess.run([sys.executable, "-m", "xiaotts.alignment", "--limit", "32", "--label", "debug"], check=True)
