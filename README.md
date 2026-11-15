# XiaoTTS

An experimental shared TTS model for English, Mandarin, Spanish, French and German,
trained from random initialization with fewer than **1,000,000 learned inference parameters**.

**Status: all five corpora prepared; the 50,000-step full-data baseline is training.
The local browser prototype works; full-baseline speech evaluation is pending.** The neural network has
757,828 parameters and the differentiable waveform renderer has zero learned parameters.

The first version targets one recorded voice per language. Mandarin tones and
European-language stress must survive the pronunciation pipeline. All five languages
participate from the first training experiment, including debugging runs.

## Data

- English: [LJSpeech](https://keithito.com/LJ-Speech-Dataset/), public domain in the US.
- Other four languages: [CSS10](https://github.com/Kyubyong/css10), whose linked
  Kaggle releases list CC0. The source listing and archive hashes are saved during download.
- Original audio is preserved. Training, development and final test passages are
  separated before tuning. After alignment and feature checks, training retains
  40,940 recordings and 84.63 hours; 1,048 development recordings are cached separately.
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
.venv/bin/python -m xiaotts.train --features data/features/full --output runs/baseline-v1 --steps 50000 --workers 4
.venv/bin/python -m xiaotts.export --checkpoint runs/baseline-v1/latest.pt --output artifacts/baseline-export
```

These are the pipeline interfaces, not evidence that the full baseline has finished.
Use `scripts/session_job.py --name <unique-name> --timeout-seconds <seconds> -- <command>`
to record and supervise a job. Each training run stores its actual source snapshot,
configuration, seed, data fingerprint, checkpoints, metrics and five-language samples.

New training checkpoints include optimizer, schedule position, Python/NumPy/PyTorch
random state and exact feature/source identities. Recover an interrupted run into
a new directory with the same recipe and total step budget, adding
`--resume runs/<interrupted-run>/latest.pt`. Inherited steps and training time remain
in the record. Older debug checkpoints lack this recovery state and are rejected
instead of being silently treated as resumable. Milestones are retained every
10,000 steps; the latest checkpoint is also saved at every evaluation.

Training uses deterministic GPU kernels by default. The renderer's reflection and
linear interpolation have equivalent indexing implementations so their CUDA
gradients support this. A 100-step A100 test with real data, waveform loss and two
prefetch workers matched an uninterrupted run exactly after recovery at step 50.
`--nondeterministic` is an explicit opt-out and is recorded in the run.

The initial native frontend uses MFA dictionaries and eSpeak NG stress annotations.
It supports new sentences made from known words and reports unknown words explicitly.
Number normalization, pronunciation of unknown dictionary words and final-test
evaluation are still outstanding. The frozen development evaluator is implemented.
Debug samples generated from alignment
phonemes are labeled as such; they do not establish arbitrary-text performance.

## Local browser demo

The complete text-to-audio path has been tested in Chromium with the GPU disabled,
including an offline reload and all five languages. It uses the shared 10,000-step
full-data checkpoint, a WASM CPU neural runtime and JavaScript DSP. This proves deployment
feasibility, not speech quality. See [browser validation](docs/browser-validation.md).

Prepare matching model and pronunciation assets on the server (choose the intended
checkpoint and its exact feature cache):

```bash
.venv/bin/python -m scripts.package_browser_model --checkpoint runs/full-baseline-v1/latest.pt --features data/features/full --output artifacts/browser-model
```

The packaging command verifies checkpoint/feature identities and records hashes
for every exported asset. Use a finished, selected checkpoint for the final demo;
the current prototype is an early checkpoint. Copy the bundle to local artifacts,
then use `python scripts/install_browser_model.py artifacts/<bundle-directory>`.
The installer checks every file hash and preserves the previous model and browser
evidence before replacing `web/public/model/`. Then:

```bash
cd web
npm ci
npm run build
npm run preview
```

Open the localhost URL printed by Vite. The production build caches all five
languages for offline use. `npm run dev` does not install that offline cache.
`dist/asset-report.json` records asset hashes and raw sizes; the supporting runtime
and dictionaries are substantially larger than the 3.07 MB neural graphs.
No public website or model release has been published.

For reproducible validation, run `python -m scripts.browser_fixture` from the repo
root, followed by `npm --prefix web test`. With the production preview running,
`python -X utf8 scripts/browser_test.py` exercises generation, errors, mobile layout
and offline reload. Browser evidence is saved under `artifacts/browser/`.

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
