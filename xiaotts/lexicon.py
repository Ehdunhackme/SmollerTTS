"""Offline G2P for corpus OOV words; exported dictionaries contain pronunciation data.

The G2P models are preprocessing tools and are not shipped as neural inference
dependencies. Final-test texts are excluded from vocabulary augmentation.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
from xiaotts import LANGUAGES
from xiaotts.alignment import MODELS, read_dictionary, words
from xiaotts.fetch import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--subset", help="Alignment manifest label for a separate small debugging lexicon")
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / ("lexicons-"+args.subset if args.subset else "lexicons")
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, MFA_ROOT_DIR=str(root/"mfa"), OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    environment["PATH"] = str(Path(".mfa/bin").resolve()) + os.pathsep + environment["PATH"]
    mfa = str(Path(".mfa/bin/mfa").resolve())
    rows = []
    if args.subset:
        for language in LANGUAGES:
            rows += json.loads((root / "alignment" / args.subset / f"{language}-manifest.json").read_text())
    else:
        for split in ("train", "dev"):
            rows += [json.loads(line) for line in (root / "manifests" / f"{split}.jsonl").read_text().splitlines()]
    report = {}
    for language in LANGUAGES:
        name = MODELS[language][1]
        original = root / "mfa/pretrained_models/dictionary" / f"{name}.dict"
        lexicon = read_dictionary(original)
        missing = sorted({word for row in rows if row["language"] == language
                          for word in words(row["text"], language, lexicon) if word not in lexicon})
        wordlist = output / f"{language}-oov.txt"
        wordlist.write_text("\n".join(missing)+"\n", encoding="utf-8")
        print(language, len(missing), "OOV types", flush=True)
        generated = output / f"{language}-generated.dict"
        if missing:
            subprocess.run([mfa, "model", "download", "g2p", name], env=environment, check=True)
            subprocess.run([mfa, "g2p", str(wordlist),
                            str(root / "mfa/pretrained_models/g2p" / f"{name}.zip"), str(generated),
                            "--num_jobs", "4", "--num_pronunciations", "1", "--clean",
                            "--temporary_directory", str(output / "temporary" / language)],
                           env=environment, check=True)
        else:
            generated.write_text("", encoding="utf-8")
        temporary = output / f"{language}.dict.tmp"
        temporary.write_text(original.read_text(encoding="utf-8")+"\n"+
                             generated.read_text(encoding="utf-8"), encoding="utf-8")
        temporary.replace(output / f"{language}.dict")
        additions = read_dictionary(generated)
        report[language] = dict(oov_types=len(missing), generated=len(additions),
                                unresolved=[word for word in missing if word not in additions])
        atomic_json(output / "report.json", report)


if __name__ == "__main__":
    main()
