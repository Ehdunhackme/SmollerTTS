"""Prepare an English LJSpeech MFA corpus and native acoustic feature cache."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
from importlib.metadata import version as package_version
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import soundfile as sf
import torch

from .checkpoints import atomic_json, sha256, validate_vocabulary
from .features import SAMPLE_RATE, HOP, aligned_phones, native_mel, recorded_prosody


def write_rows(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def corpus(source, output, include_test=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new corpus directory")
    rows, ids = [], set()
    with (source / "metadata.csv").open(encoding="utf-8", newline="") as stream:
        for fields in csv.reader(stream, delimiter="|"):
            if len(fields) != 3 or not re.fullmatch(r"LJ\d{3}-\d{4}", fields[0]) or fields[0] in ids:
                raise ValueError(f"Invalid/duplicate LJSpeech metadata row: {fields[:1]}")
            name, _, text = fields
            ids.add(name)
            split = "dev" if name.startswith("LJ043-") else "test" if name.startswith("LJ048-") else "train"
            text = " ".join(re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)*|[0-9]+", text.lower().replace("’", "'")))
            if not text:
                raise ValueError(f"Empty normalized transcript: {name}")
            rows.append({"id": name, "split": split, "text": text})
    # Keep evaluation out of training, including exact duplicate transcripts.
    rank = {"test": 0, "dev": 1, "train": 2}
    seen, selected, duplicates = set(), [], []
    for row in sorted(rows, key=lambda r: (rank[r["split"]], r["id"])):
        if row["text"] in seen:
            duplicates.append(row["id"])
            continue
        seen.add(row["text"])
        selected.append(row)
    if not all(any(r["split"] == split for r in selected) for split in ("train", "dev", "test")):
        raise ValueError("Expected LJ043 development and LJ048 test chapters")
    (output / "corpus").mkdir(parents=True)
    for row in selected:
        if row["split"] == "test" and not include_test:
            continue
        audio = source / "wavs" / (row["id"] + ".wav")
        destination = output / "corpus" / audio.name
        if not audio.is_file():
            raise FileNotFoundError(audio)
        try:
            os.link(audio, destination)
        except OSError:
            shutil.copy2(audio, destination)
        row["audio"] = destination.relative_to(output).as_posix()
        destination.with_suffix(".lab").write_text(row["text"], encoding="utf-8")
    selected.sort(key=lambda r: r["id"])
    write_rows(output / "index.jsonl", selected)
    atomic_json(output / "corpus.json", {"dataset": "LJSpeech 1.1", "metadata_sha256": sha256(source / "metadata.csv"),
                "counts": dict(Counter(r["split"] for r in selected)), "duplicate_ids": duplicates,
                "test_audio_materialized": include_test, "index_sha256": sha256(output / "index.jsonl"),
                "split_note": "Reconstructed contributor split: LJ043 dev, LJ048 test, other chapters train. Historical accepted-ID exclusions are not recovered."})
    return selected


def process(job):
    row, audio, alignment_path, output, vocabulary, espeak = job
    try:
        torch.set_num_threads(1)
        wave, rate = sf.read(audio, dtype="float64")
        if rate != SAMPLE_RATE or wave.ndim != 1 or len(wave) < 1024 or not np.isfinite(wave).all():
            raise ValueError("Expected finite 22,050 Hz mono audio")
        if np.abs(wave).max() < 1e-5 or (np.abs(wave) >= .999).mean() > .02:
            raise ValueError("Silent or excessively clipped recording")
        alignment = json.loads(Path(alignment_path).read_text(encoding="utf-8"))
        tokens, stress, durations, coverage = aligned_phones(alignment, len(wave), vocabulary, espeak)
        if np.any((tokens <= 0) | (tokens >= 132) | (tokens == 2) | (tokens == 45)):
            raise ValueError("Unsupported phone in alignment")
        frames = len(wave) // HOP
        prosody = recorded_prosody(wave)
        with torch.no_grad():
            mel = native_mel(torch.from_numpy(wave[:frames * HOP].astype(np.float32))[None])[0].T.numpy()
        if mel.shape != (frames, 80) or prosody.shape != (frames, 3) or not np.isfinite(mel).all() or not np.isfinite(prosody).all():
            raise ValueError("Invalid feature shapes or values")
        destination = Path(output) / "features" / (row["id"] + ".npz")
        np.savez_compressed(destination, tokens=tokens, stress=stress, durations=durations, mel=mel, prosody=prosody)
        result = {key: row[key] for key in ("id", "text", "split")}
        result.update(features=destination.relative_to(output).as_posix(), frames=frames, seconds=frames * HOP / SAMPLE_RATE,
                      feature_sha256=sha256(destination), source_audio_sha256=sha256(audio), alignment_sha256=sha256(alignment_path), **coverage)
        return result, mel.astype(np.float64).sum(0), np.square(mel.astype(np.float64)).sum(0)
    except Exception as error:
        return {"id": row["id"], "split": row["split"], "error": str(error)}, None, None


def features(source, alignments, output, vocabulary_path, workers=4, espeak="espeak-ng", include_test=False):
    source, alignments, output = Path(source).resolve(), Path(alignments).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new feature directory")
    if workers < 1:
        raise ValueError("workers must be positive")
    version = subprocess.run([espeak, "--version"], check=True, capture_output=True, text=True, encoding="utf-8").stdout.strip()
    vocabulary = json.loads(Path(vocabulary_path).read_text(encoding="utf-8"))
    validate_vocabulary(vocabulary)
    original = json.loads((source / "corpus.json").read_text(encoding="utf-8"))
    if include_test and not original["test_audio_materialized"]:
        raise ValueError("Recreate the corpus with --include-test before preparing final-test features")
    if sha256(source / "index.jsonl") != original["index_sha256"]:
        raise ValueError("Corpus index changed")
    rows = [json.loads(line) for line in (source / "index.jsonl").read_text(encoding="utf-8").splitlines()]
    aligned = {}
    for path in alignments.rglob("*.json"):
        if path.stem in aligned:
            raise ValueError(f"Duplicate alignment file: {path.stem}")
        aligned[path.stem] = path
    jobs, reserved = [], []
    for row in rows:
        if row["split"] == "test" and not include_test:
            reserved.append({key: row[key] for key in ("id", "text", "split")})
            continue
        audio = source / row["audio"]
        jobs.append((row, audio, aligned.get(row["id"], alignments / (row["id"] + ".json")), output, vocabulary, espeak))
    (output / "features").mkdir(parents=True)
    atomic_json(output / "vocabulary.json", vocabulary)
    kept, rejected, sums, squares, count = [], [], np.zeros(80), np.zeros(80), 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, (row, total, square) in enumerate(pool.map(process, jobs, chunksize=8), 1):
            if "error" in row:
                rejected.append(row)
            else:
                kept.append(row)
                if row["split"] == "train":
                    sums += total
                    squares += square
                    count += row["frames"]
            if index % 100 == 0:
                print(json.dumps({"prepared": index, "total": len(jobs), "rejected": len(rejected)}), flush=True)
    atomic_json(output / "rejected.json", rejected)
    if len(rejected) > max(5, int(.01 * len(jobs))) or not count or not any(r["split"] == "dev" for r in kept):
        raise ValueError(f"Feature preparation failed; inspect {output / 'rejected.json'}")
    mean = sums / count
    std = np.sqrt(np.maximum(squares / count - mean ** 2, .01))
    atomic_json(output / "normalization.json", {"mean": mean.tolist(), "std": std.tolist(), "frames": count, "source": "train only"})
    for split in ("train", "dev", "test"):
        write_rows(output / f"{split}.jsonl", [r for r in kept + reserved if r["split"] == split])
    atomic_json(output / "summary.json", {"counts": dict(Counter(r["split"] for r in kept)), "test_ids": sum(r["split"] == "test" for r in rows),
                "training_hours": sum(r["seconds"] for r in kept if r["split"] == "train") / 3600,
                "manifests": {name: sha256(output / name) for name in ("train.jsonl", "dev.jsonl", "test.jsonl", "normalization.json", "vocabulary.json")},
                "corpus": original, "espeak": version, "test_audio_read": include_test,
                "versions": {name: package_version(name) for name in ("torch", "numpy", "librosa", "soundfile", "scipy", "pyworld")},
                "frontend": "Native HiFi-GAN magnitude mel, 22050 Hz, FFT 1024, hop 256, 80 bands, 0-8000 Hz, log floor 1e-5",
                "sources": {name: sha256(Path(__file__).with_name(name)) for name in ("prepare.py", "features.py")}})
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    raw = commands.add_parser("corpus", help="Create .wav/.lab pairs for MFA")
    raw.add_argument("--ljspeech", type=Path, required=True)
    raw.add_argument("--output", type=Path, required=True)
    raw.add_argument("--include-test", action="store_true", help="Only for a separate final-test preparation")
    prepared = commands.add_parser("features", help="Extract targets from MFA JSON alignments")
    prepared.add_argument("--corpus", type=Path, required=True)
    prepared.add_argument("--alignments", type=Path, required=True)
    prepared.add_argument("--output", type=Path, required=True)
    prepared.add_argument("--vocabulary", type=Path, default=Path("configs/vocabulary.json"))
    prepared.add_argument("--workers", type=int, default=4)
    prepared.add_argument("--espeak", default="espeak-ng")
    prepared.add_argument("--include-test", action="store_true")
    args = parser.parse_args()
    if args.command == "corpus":
        rows = corpus(args.ljspeech, args.output, args.include_test)
    else:
        rows = features(args.corpus, args.alignments, args.output, args.vocabulary, args.workers, args.espeak, args.include_test)
    print(json.dumps({"output": str(args.output), "counts": dict(Counter(row["split"] for row in rows))}))


if __name__ == "__main__":
    main()
