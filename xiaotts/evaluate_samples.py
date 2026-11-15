"""ASR diagnostic for saved development audio; final test use is a separate action."""
import argparse
import json
from pathlib import Path
from faster_whisper import WhisperModel
from jiwer import wer, cer
from xiaotts.evaluate_renderer import normalize
from xiaotts.fetch import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--model", default="small")
    args = parser.parse_args()
    rows = json.loads((args.samples/"samples.json").read_text())
    model = WhisperModel(args.model, device="cuda", compute_type="float16", cpu_threads=4)
    report = dict(recognizer=args.model, purpose="small development diagnostic, not a benchmark",
                  normalization="NFKC, lowercase, punctuation removal, Mandarin OpenCC t2s", samples=[])
    for row in rows:
        language, name = row["language"], row["id"]
        metric = cer if language == "zh" else wer
        truth = normalize(row["text"], language)
        result = dict(id=name, language=language, metric="CER" if language == "zh" else "WER", reference_text=row["text"])
        for kind in ("reference", "predicted"):
            segments, _ = model.transcribe(str(args.samples/f"{name}-{kind}.wav"), language=language,
                                            beam_size=5, vad_filter=False, condition_on_previous_text=False)
            text = " ".join(segment.text for segment in segments).strip()
            result[kind] = dict(text=text, error=metric(truth, normalize(text, language)))
        report["samples"].append(result)
        atomic_json(args.samples/"asr.json", report)
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
