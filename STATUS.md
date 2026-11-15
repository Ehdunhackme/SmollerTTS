# Implementation status

Updated 15 November 2026. This is an early research implementation, not a released voice.

## Verified

- Local repository created on the requested D drive and connected to the supplied GitHub origin.
  The initial implementation was pushed as commit `27fe876`; the server checkout is connected too.
- Remote A100 SXM4 40 GB is working with CUDA PyTorch 2.7.0. Training uses a separate environment.
- All five real-audio corpora downloaded, hashed and extracted. Nested CSS10 copies are not counted twice.
- Frozen passage-level train/dev/test splits, with at least 200 final-test clips per language.
  Training audio after initial checks/splits is 84.89 hours; further alignment/quality rejection is pending.
- Shared model: 757,828 learned parameters; fixed differentiable renderer: zero learned parameters.
- Five correctness tests passed on the server, including padding invariance, floating-point WAV ingestion,
  tone preservation, all-language gradients and waveform synthesis gradients.
- Random-weight ONNX encoder/decoder passed native CPU parity over all five language IDs and two new
  sequence lengths. Maximum absolute error was 2.87e-6. This is not a browser test.
- The trained 4,000-step checkpoint also passed native ONNX CPU parity checks.
- Actual browser WASM neural parity passed across all five language IDs and two
  sequence lengths, within atol=2e-5/rtol=2e-4 (largest absolute error 4.06e-5).
- Browser DSP filtering matches PyTorch with maximum waveform error 2.69e-7 on
  identical excitation; the harmonic source also passed a separate parity check.
- Local Chromium generated actual text in all five languages with GPU disabled,
  then repeated all five after an offline reload. No external requests or page errors.
  Short examples generated faster than real time on this PC; see docs/browser-validation.md.
- CPU feature prefetch passed a 600-step server check after fixing retained shared
  tensor file descriptors. Every batch still contains all five languages.
- The compact browser pronunciation format preserves all 487,131 dictionary entries.
  Retesting passed for all five languages online/offline. Application assets fell
  from 58.97 MB to 35.28 MB raw; measured Chromium private memory fell from 683 MB
  to 371 MB in the isolated test (not a portable resident-RAM measurement).
- A frozen 100-sentence text-input development evaluator now hashes its protocol,
  dictionary/vocabulary assets and exact ASR weights. Its first debug-model evaluation
  completed and confirms poor speech; see docs/evaluation-protocol.md for all results.
- A fitted-control renderer diagnostic produced finite audio in all languages; a small ASR panel matched
  the reference-recording error rates after fitting. This is reconstruction evidence, not TTS quality evidence.
- Corrected debug alignment/feature cache has 200 clips, 40 per language: 32 train and 8 dev each.
- Two shared from-scratch debug models completed. The second ran 4,000 steps in 119.8 seconds,
  with 413,132,288 bytes peak PyTorch CUDA allocation. Its source, seed, config and data fingerprint were saved.

## Current limitations

- Speech quality is still poor on unseen development sentences. Five-sentence diagnostics show large
  recognition errors. These tiny panels cannot support benchmark or naturalness claims.
- The first renderer gradient implementation destabilized predicted pitch. The current training path
  supervises pitch directly and detaches it from the oscillator's spectral gradient by default.
- Full-corpus OOV pronunciation generation and full train/dev alignment completed.
  Acoustic feature caching is running. A verified dependent job audits all languages
  and starts the 50,000-step baseline when caching succeeds; training has not begun yet.
- Native and browser pronunciation use dictionaries with explicit stress/tones.
  Unknown words and digits produce explicit errors; broader pronunciation and number handling remain.
- Browser DSP and the local UI work with the debug checkpoint. Final quality evaluation,
  calibrated automatic experiment selection and publication are not complete.
- Final-test data have not been used for tuning or model evaluation.

## Runtime records

On the training server, project files and artifacts are under the XiaoTTS checkout in the user's home.
Supervised jobs record real process IDs, command arguments and exit state under `runs/<job>/process.json`.
Inspect those records together with the live process table; a stale JSON file is not proof of a live job.

- `runs/lexicon-augmentation`: completed full corpus pronunciation preparation.
- `runs/full-data-preparation`: waits on that exact process, then aligns and caches full train/dev data.
- `runs/full-baseline-pipeline`: waits on the exact full-data job, audits it, then trains the baseline.
- `runs/frozen-debug-evaluation`: completed the first fixed text-input evaluation.
- `runs/overfit-five-clips`: stronger memorization diagnostic, one training clip per language.
- `runs/debug-train-v1`, `runs/debug-train-v2`: completed scratch training runs.
- `runs/debug-asr-v1`, `runs/debug-asr-v2`: completed development diagnostics.
- `artifacts/export-smoke-v2`: native ONNX parity report.
- `artifacts/debug-v2-export`: trained checkpoint ONNX parity report.

The latest debug checkpoint, run record and exact source archive are backed up locally under
`artifacts/debug-train-v2/` (ignored by Git). Development ASR remains poor, particularly for
Spanish, French and German. The training-sentence panel is better in English and Spanish but
does not establish generalization; French and German still need substantial work.

The user explicitly removed the proposed fixed 12-hour ceiling. New jobs use individual step/time
budgets and checkpoints; there is no campaign billing cap or provider shutdown requirement.

## Next work

1. Inspect trained-example reconstruction and actual text inference; retain honest per-language failures.
2. Complete full-data alignment and feature auditing, then train and profile the baseline from scratch.
3. Implement and test complete browser CPU inference and pronunciation, counting all deployed assets.
4. Calibrate the fixed development evaluator and provisional promotion rules against the full baseline.
5. Run comparable experiments, repeat finalists, evaluate locked final tests and deliver the verified demo.
