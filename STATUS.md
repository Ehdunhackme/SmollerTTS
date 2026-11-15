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
- Full-corpus OOV pronunciation generation completed; full train/dev alignment is running, with
  acoustic feature caching next. Full-data baseline training has not begun.
- Native pronunciation currently uses dictionaries and an external eSpeak NG stress annotator.
  Unknown words and digits produce explicit errors; browser G2P/number handling remains outstanding.
- Browser DSP, browser UI, final evaluation, calibrated automatic experiment selection and publication
  are not complete. Only the neural graph has passed native ONNX compatibility checks.
- Final-test data have not been used for tuning or model evaluation.

## Runtime records

On the training server, project files and artifacts are under the XiaoTTS checkout in the user's home.
Supervised jobs record real process IDs, command arguments and exit state under `runs/<job>/process.json`.
Inspect those records together with the live process table; a stale JSON file is not proof of a live job.

- `runs/lexicon-augmentation`: completed full corpus pronunciation preparation.
- `runs/full-data-preparation`: waits on that exact process, then aligns and caches full train/dev data.
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
4. Freeze the development evaluator and promotion rules before broader autonomous trials.
5. Run comparable experiments, repeat finalists, evaluate locked final tests and deliver the verified demo.
