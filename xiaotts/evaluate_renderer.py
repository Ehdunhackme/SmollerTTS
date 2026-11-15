"""Calibrated ASR diagnostic of oracle reconstructions; not a TTS benchmark."""
import argparse
import json
from pathlib import Path
import re
import unicodedata
from functools import lru_cache
from faster_whisper import WhisperModel
from jiwer import wer, cer
from xiaotts.fetch import atomic_json


@lru_cache(maxsize=1)
def chinese_converter():
    import opencc
    return opencc.OpenCC("t2s.json")


def normalize(text, language):
    text = unicodedata.normalize("NFKC", text).lower()
    if language == "zh":
        text = chinese_converter().convert(text)
        return "".join(char for char in text if char.isalnum())
    return " ".join(re.findall(r"[^\W_]+(?:'[^\W_]+)*", text, re.UNICODE))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("artifacts/renderer-gate"))
    parser.add_argument("--model", default="small")
    args = parser.parse_args()
    report = json.loads((args.directory / "report.json").read_text())
    recognizer = WhisperModel(args.model, device="cuda", compute_type="float16", cpu_threads=4)
    output = dict(diagnostic="Oracle reconstruction ASR, not TTS or naturalness evidence",
                  recognizer=args.model, normalization="NFKC, lowercase, punctuation removal; Mandarin OpenCC t2s", samples=[])
    for sample in report["samples"]:
        language = sample["language"]
        truth = normalize(sample["text"], language)
        result = dict(language=language, text=sample["text"], metric="CER" if language == "zh" else "WER")
        metric = cer if language == "zh" else wer
        for label in ("reference", "initial", "fitted", "world-reference"):
            segments, _ = recognizer.transcribe(str(args.directory / f"{language}-{label}.wav"),
                                                language=language, beam_size=5, vad_filter=False,
                                                condition_on_previous_text=False)
            transcription = " ".join(segment.text for segment in segments).strip()
            result[label] = dict(text=transcription, error=metric(truth, normalize(transcription, language)))
        output["samples"].append(result)
        atomic_json(args.directory / "asr.json", output)
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
