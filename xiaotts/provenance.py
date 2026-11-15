"""Capture the actual source used for an experiment, including uncommitted work."""
import hashlib
from pathlib import Path
import subprocess
import tarfile


def snapshot(destination):
    root = Path(__file__).resolve().parent.parent
    files = sorted([*root.glob("xiaotts/*.py"), *root.glob("scripts/*.py"),
                    *root.glob("scripts/*.sh"), root/"pyproject.toml"])
    digest = hashlib.sha256()
    with tarfile.open(Path(destination)/"source.tar.gz", "w:gz") as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            digest.update(relative.encode()+b"\0"+path.read_bytes()+b"\0")
            archive.add(path, arcname=relative)
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    return dict(source_sha256=digest.hexdigest(), git_commit=result.stdout.strip() if result.returncode == 0 else None,
                source_archive="source.tar.gz")
