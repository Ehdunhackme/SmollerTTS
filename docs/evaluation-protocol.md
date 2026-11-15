# Fixed development evaluation

The first frozen panel contains 20 development sentences in each of the five
languages, selected by a fixed hash of their IDs. No samples are removed based on
reference-recognition quality or generated results. The final-test manifest is
not read by this evaluator.

The input is actual text through the dictionary frontend. Ground-truth phonemes,
pitch, durations and aligned acoustic features are not supplied at inference.
Predicted durations and the fixed waveform renderer generate the audio.

The panel records its dataset/vocabulary/dictionary hashes, evaluator source hashes,
the exact cached Whisper-small revision and the recognizer weight hashes. Evaluation
refuses changed sources or assets. Beam size, temperature, text normalization and
recognizer settings are frozen. Candidate model/training code can change separately.
The reference recordings are recognized under the same settings as generated audio.

WER is reported for English, Spanish, French and German; CER for Mandarin, after
fixed OpenCC simplification. These are intelligibility proxies. They do not prove
naturalness, correct lexical tones or native pronunciation. WER can exceed 100%
when the recognizer inserts many words into poor generated audio.

## Provisional promotion rule

The evaluator and `selection.py` enforce all five language results and fewer than
one million learned parameters. A candidate must improve at least one language's
error rate by more than 0.03 while no language worsens by more than 0.02. Invalid
audio must not exceed 5% in any language. CER and WER are never pooled into a single
raw average. No nomination is a final win: matched training budgets, browser/export
checks and repeated seeds are required before selection. Tolerances are declared
before broad search and the small panel's uncertainty remains material.

Invalid output includes numerical failure, very low RMS, more than 1% samples at
or beyond full scale, or generated/reference duration outside [0.3, 3]. Frontend
errors count as failed outputs and empty recognition, rather than being dropped.

## Initial diagnostic: debug-train-v2

This model saw only 32 training recordings per language. It is not the full-data
baseline. The fixed panel produced:

| Language | Metric | Generated error | Original-recording error | Invalid outputs |
| --- | --- | ---: | ---: | ---: |
| English | WER | 64.4% | 5.2% | 0/20 |
| Mandarin | CER | 68.7% | 9.8% | 1/20 |
| Spanish | WER | 60.9% | 4.8% | 0/20 |
| French | WER | 144.2% | 21.8% | 0/20 |
| German | WER | 65.6% | 11.3% | 0/20 |

Speech is poor across the panel, with particularly severe French failures.
Recognition errors on original recordings also limit absolute interpretations.
These numbers are neither a public benchmark result nor evidence of competitive
quality. The protocol is now available for comparison with the proper baseline.

```bash
.venv/bin/python -m xiaotts.benchmark freeze
.venv/bin/python -m xiaotts.benchmark evaluate --checkpoint runs/full-baseline-v1/latest.pt --output runs/full-baseline-dev-v1
```

The first command creates `evaluation/dev-v1.json` once and refuses to overwrite it.
Keep it with the experiment artifacts. All results retain the panel's SHA256 hash.

## Recorded trials

`scripts/run_trial.py` runs one declared hypothesis after a complete full-data
baseline and matching evaluation exist. A specification names one change among
learning rate, width, spectral-loss weight and spectral-loss warmup. It verifies
the incumbent checkpoint/panel hashes, uses random initialization with the same
seed, optimizer-step count and batch composition, then exports and evaluates the
candidate. It preserves both models and appends the outcome to an experiment ledger.

The same step count is a matched data/optimizer budget, not identical GPU cost;
the record includes both measured training times. The first actual trial,
`trial-width144-seed17`, increases shared width from 128 to 144 with 948,164 learned
parameters, preserving the baseline's 50,000 steps, seed and batch composition.
Its outcome is pending; the finished baseline remains the incumbent.
