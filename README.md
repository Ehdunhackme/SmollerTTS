# XiaoTTS

An experimental shared TTS model for English, Mandarin, Spanish, French and German,
trained from random initialization with fewer than **1,000,000 learned inference parameters**.

**Status: all five corpora downloaded; shared prototype training and full-data
preparation are running. The model is not ready for use.** The neural network has
757,828 parameters and the differentiable waveform renderer has zero learned parameters.

The first version targets one recorded voice per language. Mandarin tones and
European-language stress must survive the pronunciation pipeline. All five languages
participate from the first training experiment, including debugging runs.

## Data

- English: [LJSpeech](https://keithito.com/LJ-Speech-Dataset/), public domain in the US.
- Other four languages: [CSS10](https://github.com/Kyubyong/css10), whose linked
  Kaggle releases list CC0. The source listing and archive hashes are saved during download.
- Original audio is preserved. Training, development and final test passages are
  separated before tuning. Actual retained hours will be reported after validation.
- No synthetic speech generation. Pretrained aligners can annotate recordings;
  their weights are not part of the deployed TTS model.

## Server setup

```bash
bash scripts/bootstrap.sh
python3 -m xiaotts.fetch --root data
bash scripts/setup_aligner.sh
```

Use the existing CUDA PyTorch with a separate Python environment. Forced alignment
uses a separate conda-forge environment. Exact installed dependencies are recorded
under `artifacts/`. Downloaded audio, checkpoints and environment files are ignored
by Git. Source files live in the user's D-drive checkout and are synchronized to
the training server.

## Implemented commands

```bash
# Freeze corpus splits after all archives have completed extraction.
.venv/bin/python -m xiaotts.manifest
# Complete missing dictionary entries using offline G2P, then align train/dev.
.venv/bin/python -m xiaotts.lexicon
.venv/bin/python -m xiaotts.alignment --limit 0 --label full --parallel-languages 2 --jobs 6
.venv/bin/python -m xiaotts.features --label full --workers 8
# A run directory must be new; all five languages are required in every batch.
.venv/bin/python -m xiaotts.train --features data/features/full --output runs/baseline-v1 --steps 50000
.venv/bin/python -m xiaotts.export --checkpoint runs/baseline-v1/latest.pt --output artifacts/baseline-export
```

These are the pipeline interfaces, not evidence that the full baseline has finished.
Use `scripts/session_job.py --name <unique-name> --timeout-seconds <seconds> -- <command>`
to record and supervise a job. Each training run stores its actual source snapshot,
configuration, seed, data fingerprint, checkpoints, metrics and five-language samples.

The initial native frontend uses MFA dictionaries and eSpeak NG stress annotations.
It supports new sentences made from known words and reports unknown words explicitly.
Number normalization, arbitrary-word browser pronunciation, browser DSP and the final
benchmark harness are still outstanding. Debug samples generated from alignment
phonemes are labeled as such; they do not establish arbitrary-text performance.

## Research gates

1. Validate the data and pronunciation/alignment pipeline for every language.
2. Reconstruct real audio using fitted synthesis controls before architecture search.
3. Overfit a small five-language subset and verify actual text inference.
4. Train and measure a full-data baseline.
5. Freeze the development evaluator, then run bounded, recorded experiments.
6. Confirm finalists at matched compute with repeated seeds; evaluate locked test data.
7. Export and measure the complete browser CPU path, including frontend and synthesis.

The user removed the initially proposed fixed 12-hour session ceiling. Jobs retain
explicit step budgets, optional individual timeouts, checkpointing and failure logs.
Stopping a process does not change provider billing. Required artifacts are backed up separately.
No public deployment, quality or benchmark leadership is implied by setup success.

Code is MIT licensed; dataset, alignment model and dictionary licenses are separate.
