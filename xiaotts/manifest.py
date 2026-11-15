"""Audit source audio headers, deduplicate, and freeze passage-level splits."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import soundfile as sf

from xiaotts import LANGUAGES
from xiaotts.fetch import atomic_json


def text_key(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(char for char in text if char.isalnum())


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_rows(root, language):
    directory = root / "raw" / language
    pattern = "metadata.csv" if language == "en" else "transcript.txt"
    # Some Kaggle archives include an identical nested copy. One canonical manifest
    # plus content deduplication prevents counting those recordings twice.
    candidates = sorted(directory.rglob(pattern), key=lambda path: (len(path.parts), str(path)))
    if not candidates:
        raise FileNotFoundError(f"No {pattern} under {directory}")
    metadata = candidates[0]
    for index, line in enumerate(metadata.read_text(encoding="utf-8-sig").splitlines()):
        fields = line.split("|")
        if len(fields) < 3:
            yield {"reject": "malformed_metadata", "line": index}
            continue
        if language == "en":
            path = metadata.parent / "wavs" / (fields[0] + ".wav")
            text = fields[2]
            book = fields[0].split("-")[0]
        else:
            path = metadata.parent / fields[0]
            # CSS10's generated Mandarin pinyin contains ambiguous-reading errors.
            # Use the original Hanzi with the verified alignment dictionary instead.
            text = fields[1] if language == "zh" else fields[2]
            book = str(Path(fields[0]).parent)
        yield dict(id=language + "_" + path.stem, language=language,
                   audio=str(path.resolve()), text=text.strip(), book=book, source_index=index)


def build(root, minimum_holdout=200):
    directory = root / "manifests"
    if (directory / "dataset.json").exists():
        raise FileExistsError("Frozen manifest already exists; use a new data root for changes")
    directory.mkdir(parents=True, exist_ok=True)
    all_rows, rejects, summary = [], [], {}
    for language in LANGUAGES:
        if not (root / "provenance" / f"{language}.json").exists():
            raise RuntimeError(f"Download/extraction incomplete for {language}")
        unique, rows = set(), []
        book_indices = Counter()
        for row in source_rows(root, language):
            if "reject" in row:
                rejects.append({"language": language, **row})
                continue
            key = text_key(row["text"])
            if len(key) < 2 or key in unique:
                rejects.append({"id": row["id"], "reject": "empty_or_duplicate_text"})
                continue
            unique.add(key)
            try:
                audio = sf.info(row["audio"])
                sr, frames = audio.samplerate, audio.frames
                seconds = frames / sr
                if audio.channels != 1 or not 0.5 <= seconds <= 25:
                    raise ValueError("Unsupported channels or duration")
            except (OSError, EOFError, RuntimeError, ValueError) as error:
                rejects.append({"id": row["id"], "reject": "audio_header", "detail": str(error)})
                continue
            row.update(sample_rate=sr, samples=frames, seconds=seconds,
                       text_sha256=digest(key), group=f"{row['book']}:{book_indices[row['book']]//100:04d}")
            book_indices[row["book"]] += 1
            rows.append(row)
        groups = defaultdict(list)
        for row in rows:
            groups[row["group"]].append(row)
        ordered = sorted(groups, key=lambda key: digest(f"xiaotts-split-v1:{language}:{key}"))
        assigned = {}
        for split in ("test", "dev"):
            count = 0
            while count < minimum_holdout:
                if not ordered:
                    raise ValueError(f"Insufficient groups for {language} holdouts")
                key = ordered.pop(0)
                assigned[key] = split
                count += len(groups[key])
        for row in rows:
            row["split"] = assigned.get(row["group"], "train")
        if sum(row["split"] == "train" for row in rows) < minimum_holdout:
            raise ValueError(f"Training corpus unexpectedly small for {language}")
        summary[language] = {split: dict(clips=sum(r["split"] == split for r in rows),
                                          hours=sum(r["seconds"] for r in rows if r["split"] == split)/3600)
                             for split in ("train", "dev", "test")}
        all_rows.extend(rows)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in all_rows)
    (directory / "all.jsonl").write_text(payload, encoding="utf-8")
    for split in ("train", "dev", "test"):
        (directory / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows if row["split"] == split),
            encoding="utf-8")
    atomic_json(directory / "rejected.json", rejects)
    atomic_json(directory / "dataset.json", dict(version="xiaotts-split-v1", sha256=digest(payload),
                grouping="100 consecutive retained clips within source book/section", summary=summary,
                rejected=len(rejects), final_test_locked=True))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    args = parser.parse_args()
    build(args.root)


if __name__ == "__main__":
    main()
