"""Prepare multilingual MFA corpora and align using compatible IPA dictionaries.

First run is a bounded subset in each of the five languages. Final test data are
excluded. Full alignment uses the same frozen train/dev manifest and dictionaries.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

from xiaotts import LANGUAGES
from xiaotts.fetch import atomic_json

MODELS = {
    "en": ("english_mfa", "english_us_mfa"),
    "zh": ("mandarin_mfa", "mandarin_china_mfa"),
    "es": ("spanish_mfa", "spanish_spain_mfa"),
    "fr": ("french_mfa", "french_mfa"),
    "de": ("german_mfa", "german_mfa"),
}


def dictionary_path(root, language, lexicons=None):
    augmented = (lexicons or root / "lexicons") / f"{language}.dict"
    if augmented.exists():
        return augmented
    return root / "mfa/pretrained_models/dictionary" / (MODELS[language][1]+".dict")


def dictionary_entries(path):
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        word, pronunciation = fields[0], fields[1:]
        probability = 1.0
        if len(pronunciation) > 1:
            try:
                probability = float(pronunciation[0])
                while pronunciation:
                    float(pronunciation[0])
                    pronunciation = pronunciation[1:]
            except ValueError:
                pass
        if pronunciation:
            yield word, probability, pronunciation


def read_dictionary(path):
    result, scores = {}, {}
    for word, probability, pronunciation in dictionary_entries(path):
        if probability > scores.get(word, -1):
            result[word], scores[word] = pronunciation, probability
    return result


def words(text, language, lexicon):
    text = text.lower().replace("’", "'").replace("‘", "'")
    if language == "zh":
        # Deterministic longest dictionary match avoids a learned deployed segmenter.
        chunks = re.findall(r"[\u3400-\u9fff]+|[a-z]+|[0-9]+", text)
        output = []
        for chunk in chunks:
            position = 0
            while position < len(chunk):
                length = min(12, len(chunk)-position)
                while length > 1 and chunk[position:position+length] not in lexicon:
                    length -= 1
                output.append(chunk[position:position+length])
                position += length
        return output
    return re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)*|[0-9]+", text, flags=re.UNICODE)


def prepare(root, limit, label, lexicons=None):
    rows = []
    for split in ("train", "dev"):
        rows += [json.loads(line) for line in (root / "manifests" / f"{split}.jsonl").read_text().splitlines()]
    reports = {}
    for language in LANGUAGES:
        acoustic, dictionary = MODELS[language]
        lexicon = read_dictionary(dictionary_path(root, language, lexicons))
        selected = []
        for split in ("train", "dev"):
            candidates = [row for row in rows if row["language"] == language and row["split"] == split]
            selected += candidates[:limit if split == "train" else max(4, limit//4)] if limit else candidates
        directory = root / "alignment" / label / "corpus" / language
        directory.mkdir(parents=True, exist_ok=True)
        oov, total = defaultdict(int), 0
        for row in selected:
            tokenized = words(row["text"], language, lexicon)
            if not tokenized:
                raise ValueError(f"Empty normalized transcript: {row['id']}")
            total += len(tokenized)
            for word in tokenized:
                if word not in lexicon:
                    oov[word] += 1
            destination = directory / (row["id"]+".wav")
            if not destination.exists():
                destination.symlink_to(Path(row["audio"]).resolve())
            (directory / (row["id"]+".lab")).write_text(" ".join(tokenized), encoding="utf-8")
        reports[language] = dict(clips=len(selected), tokens=total, oov_tokens=sum(oov.values()), oovs=dict(oov))
        atomic_json(root / "alignment" / label / f"{language}-manifest.json", selected)
    atomic_json(root / "alignment" / label / "preparation.json", reports)
    print(json.dumps({lang: {k:v for k,v in value.items() if k != "oovs"}
                      for lang,value in reports.items()}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--limit", type=int, default=32, help="Training clips/language; zero means full train+dev")
    parser.add_argument("--label", default="debug")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--parallel-languages", type=int, default=1)
    parser.add_argument("--lexicons", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    prepare(root, args.limit, args.label, args.lexicons)
    if args.prepare_only:
        return
    environment = dict(os.environ, MFA_ROOT_DIR=str(root / "mfa"), OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    environment["PATH"] = str(Path(".mfa/bin").resolve()) + os.pathsep + environment["PATH"]
    def align_language(language):
        acoustic, dictionary = MODELS[language]
        destination = root / "alignment" / args.label / "aligned" / language
        destination.mkdir(parents=True, exist_ok=True)
        command = [str(Path(".mfa/bin/mfa").resolve()), "align",
                   str(root / "alignment" / args.label / "corpus" / language),
                   str(dictionary_path(root, language, args.lexicons)),
                   str(root / "mfa/pretrained_models/acoustic" / (acoustic+".zip")),
                   str(destination), "--output_format", "json", "--single_speaker", "--num_jobs", str(args.jobs),
                   "--temporary_directory", str(root / "alignment" / args.label / "temporary" / language),
                   "--no_tokenization", "--clean", "--overwrite"]
        print("Aligning", language, flush=True)
        local_environment = dict(environment, MFA_ROOT_DIR=str(root / "mfa-workers" / language))
        subprocess.run(command, env=local_environment, check=True)
    with ThreadPoolExecutor(max_workers=args.parallel_languages) as pool:
        list(pool.map(align_language, LANGUAGES))


if __name__ == "__main__":
    main()
