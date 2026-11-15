"""Download official existing corpora with resume, checksums and safe extraction.

Run on the training server: python -m xiaotts.fetch --root data
Raw audio and dataset licenses are separate from the code's MIT license.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
import zipfile

import requests

SOURCES = {
    "en": dict(url="https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
               page="https://keithito.com/LJ-Speech-Dataset/", license="Public domain in US",
               archive="LJSpeech-1.1.tar.bz2"),
}
for code, language in (("zh", "chinese"), ("es", "spanish"), ("fr", "french"), ("de", "german")):
    slug = f"bryanpark/{language}-single-speaker-speech-dataset"
    SOURCES[code] = dict(url=f"https://www.kaggle.com/api/v1/datasets/download/{slug}",
                         page=f"https://www.kaggle.com/datasets/{slug}",
                         metadata=f"https://www.kaggle.com/api/v1/datasets/view/{slug}",
                         license="CC0 per dataset listing", archive=f"css10-{code}.zip")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def download(url, target):
    if target.exists():
        return
    part = target.with_suffix(target.suffix + ".part")
    for attempt in range(5):
        start = part.stat().st_size if part.exists() else 0
        try:
            # Kaggle download endpoints accept GET; HEAD can return 404.
            with requests.get(url, headers={"Range": f"bytes={start}-"} if start else {},
                              stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                append = start > 0 and response.status_code == 206
                if append and not response.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
                    raise ValueError("Server resumed at an unexpected offset")
                expected = int(response.headers.get("Content-Length", 0))
                received, last = 0, time.monotonic()
                with part.open("ab" if append else "wb") as output:
                    for chunk in response.iter_content(4 * 1024 * 1024):
                        output.write(chunk)
                        received += len(chunk)
                        if time.monotonic() - last > 20:
                            print(f"{target.name}: {output.tell()/1e9:.2f} GB", flush=True)
                            last = time.monotonic()
                if expected and received != expected:
                    raise IOError(f"Incomplete download: {received}/{expected}")
            os.replace(part, target)
            return
        except (requests.RequestException, OSError) as error:
            print(f"{target.name}: attempt {attempt+1}: {error}", flush=True)
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 16))


def safe_destination(root, name):
    destination = (root / name).resolve()
    if not destination.is_relative_to(root.resolve()):
        raise ValueError(f"Unsafe archive member: {name}")
    return destination


def extract(archive, root):
    root.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                destination = safe_destination(root, member.filename)
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Archive symlinks are not allowed")
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as src, destination.open("wb") as dst:
                    shutil.copyfileobj(src, dst, 4 * 1024 * 1024)
    else:
        with tarfile.open(archive) as source:
            for member in source:
                destination = safe_destination(root, member.name)
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(member) as src, destination.open("wb") as dst:
                        shutil.copyfileobj(src, dst, 4 * 1024 * 1024)
                else:
                    raise ValueError(f"Unsupported archive member: {member.name}")


def fetch_one(root, language):
    source = SOURCES[language]
    marker = root / "provenance" / f"{language}.json"
    if marker.exists():
        print(f"{language}: extraction already recorded", flush=True)
        return
    archive = root / "archives" / source["archive"]
    archive.parent.mkdir(parents=True, exist_ok=True)
    if "metadata" in source:
        result = requests.get(source["metadata"], timeout=60)
        result.raise_for_status()
        atomic_json(root / "provenance" / f"{language}-source-metadata.json", result.json())
    print(f"{language}: downloading {source['url']}", flush=True)
    download(source["url"], archive)
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    print(f"{language}: extracting {archive.stat().st_size/1e9:.2f} GB", flush=True)
    extract(archive, root / "raw" / language)
    atomic_json(marker, {**source, "language": language, "sha256": digest.hexdigest(),
                        "bytes": archive.stat().st_size, "completed_unix": time.time()})
    print(f"{language}: complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--languages", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_one, args.root, language) for language in args.languages]
        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
