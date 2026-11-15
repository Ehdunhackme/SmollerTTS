"""Cache aligned acoustic targets, phonemes, tones, stress and original audio."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
from xiaotts import LANGUAGES
from xiaotts.acoustics import read_audio, extract_controls
from xiaotts.alignment import MODELS, dictionary_entries
from xiaotts.dsp import SAMPLE_RATE, HOP
from xiaotts.fetch import atomic_json
from xiaotts.phones import label_word, split_phone


def vocabulary(root):
    symbols = {"<pad>", "<sil>", "<unk>"}
    for _, dictionary in MODELS.values():
        for _, _, pronunciation in dictionary_entries(root / "mfa/pretrained_models/dictionary" / (dictionary+".dict")):
            for phone in pronunciation:
                symbols.add(split_phone(phone)[0])
    ordered = ["<pad>", "<sil>", "<unk>"] + sorted(symbols-{"<pad>", "<sil>", "<unk>"})
    if len(ordered) > 384:
        raise ValueError(f"Phoneme inventory exceeds reserved embedding capacity: {len(ordered)}")
    return {phone: i for i, phone in enumerate(ordered)}


def process(job):
    row, aligned, destination, vocab = job
    try:
        alignment = json.loads(Path(aligned).read_text())
        entries = alignment["tiers"]["phones"]["entries"]
        words = alignment["tiers"]["words"]["entries"]
        if any(phone in ("spn", "<unk>", "unk") for _, _, phone in entries):
            raise ValueError("Unknown-word/noise phone in alignment")
        labels = [split_phone(phone) for _, _, phone in entries]
        stress_matches = 0
        for start, end, word in words:
            indices = [i for i, (a, b, phone) in enumerate(entries)
                       if start-1e-4 <= (a+b)/2 < end and phone not in ("", "sil", "sp")]
            if not indices or not word:
                continue
            marked, matched = label_word([entries[i][2] for i in indices], word, row["language"])
            stress_matches += int(matched)
            for index, label in zip(indices, marked):
                labels[index] = label
        wave = read_audio(row["audio"])
        clipped_fraction = float((np.abs(wave) >= 0.999).mean())
        if clipped_fraction > 0.02:
            raise ValueError(f"Excessive source clipping: {clipped_fraction:.3f}")
        controls, waveform, _ = extract_controls(wave)
        intervals, full_labels, cursor = [], [], 0.0
        for (start, end, phone), label in zip(entries, labels):
            if end <= start or start < cursor-0.021:
                raise ValueError("Invalid or overlapping alignment")
            if start > cursor+0.005:
                intervals.append((cursor, start))
                full_labels.append(("<sil>", 0, 0))
            intervals.append((max(cursor, start), end))
            full_labels.append(label)
            cursor = end
        duration = len(wave)/SAMPLE_RATE
        if cursor < duration:
            intervals.append((cursor, duration+HOP/SAMPLE_RATE))
            full_labels.append(("<sil>", 0, 0))
        ends = np.array([end for _, end in intervals])
        frame_ids = np.searchsorted(ends, np.arange(len(controls))*HOP/SAMPLE_RATE, side="right")
        frame_ids = np.minimum(frame_ids, len(intervals)-1)
        durations = np.bincount(frame_ids, minlength=len(intervals)).astype(np.int64)
        tokens = np.array([vocab[phone] for phone, _, _ in full_labels], dtype=np.int64)
        tones = np.array([tone for _, tone, _ in full_labels], dtype=np.int64)
        stress = np.array([value for _, _, value in full_labels], dtype=np.int64)
        if not np.isfinite(controls).all() or durations.sum() != len(controls):
            raise ValueError("Invalid feature lengths or numbers")
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, tokens=tokens, tones=tones, stress=stress, durations=durations,
                            controls=controls, waveform=waveform, language=LANGUAGES.index(row["language"]))
        return {**row, "features": str(destination), "frames": len(controls), "phonemes": len(tokens),
                "tone_frames": int((tones[frame_ids] > 0).sum()), "stress_words_matched": stress_matches,
                "words": len(words), "clipped_fraction": clipped_fraction}
    except Exception as error:
        return {"id": row["id"], "language": row["language"], "rejected": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--label", default="debug")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = args.root.resolve()
    vocab = vocabulary(root)
    output = root / "features" / args.label
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "vocabulary.json", vocab)
    jobs, results = [], []
    for language in LANGUAGES:
        for row in json.loads((root / "alignment" / args.label / f"{language}-manifest.json").read_text()):
            aligned = root / "alignment" / args.label / "aligned" / language / (row["id"]+".json")
            if not aligned.exists():
                results.append({"id": row["id"], "language": language, "rejected": "Missing alignment"})
                continue
            jobs.append((row, aligned, output / (row["id"]+".npz"), vocab))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for number, result in enumerate(pool.map(process, jobs)):
            results.append(result)
            if number % 25 == 0:
                print(f"Features {number+1}/{len(jobs)}", flush=True)
    accepted = [row for row in results if "rejected" not in row]
    for language in LANGUAGES:
        for split in ("train", "dev"):
            if not any(row["language"] == language and row["split"] == split for row in accepted):
                atomic_json(output / "failures.json", results)
                raise ValueError(f"No usable {language}/{split} examples; training cannot drop a language")
    (output / "index.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in accepted), encoding="utf-8")
    atomic_json(output / "rejected.json", [row for row in results if "rejected" in row])
    code_hash = hashlib.sha256(b"".join((Path(__file__).parent / file).read_bytes()
                                      for file in ("features.py", "acoustics.py", "phones.py"))).hexdigest()
    atomic_json(output / "summary.json", dict(accepted=len(accepted), rejected=len(results)-len(accepted),
                per_language=dict(Counter(row["language"] for row in accepted)),
                vocabulary=len(vocab), source_hash=code_hash,
                dataset=json.loads((root / "manifests/dataset.json").read_text())["sha256"]))
    print((output / "summary.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
