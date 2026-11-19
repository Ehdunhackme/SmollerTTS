# Contributing to SmollerTTS

Thanks for helping. The current model has a 1.88M-parameter acoustic network and a frozen 0.93M-parameter HiFi-GAN vocoder. Start with the [training guide](docs/training.md) if you want to train your own acoustic weights, or the [local demo instructions](README.md#run-locally) for browser work.

## Useful contributions

- **Speech quality:** improve pronunciation, durations or prosody without relying only on aligned-duration validation loss. Compare free-running audio on fixed development prompts.
- **Data preparation:** audit LJSpeech alignments and stress labels, improve rejection reports, or document a reproducible preparation environment.
- **Browser performance:** reduce startup time, memory or generation time while checking waveform changes and pronunciation regressions.
- **Evaluation:** add unseen prompts, report intelligibility separately from naturalness, and document voices, runtime and scoring versions. Keep published peer scores distinct from models you rerun.
- **Documentation:** fix a setup step that failed on a fresh machine. Include the command, platform and package versions.

For a substantial architecture or dataset change, open an issue explaining the experiment first so others can coordinate. Small fixes can go straight to a pull request. Custom acoustic architectures are welcome as experiments, but must not silently replace the released browser model or its benchmark scores.

## Checks before a pull request

For Python training changes:

```sh
python -m pip install -e ".[train,export,dev]"
python -m pytest tests/training -q
python -m smollertts.smoke --output runs/my-smoke
```

For frontend changes:

```sh
npm ci
npm run setup
npm test
npm run build
npm start
```

With the server running, use the browser regression test described in the [README](README.md#development). For export changes, also run `python -m smollertts.export` on a trained checkpoint and test the resulting custom browser build. Numerical export agreement is separate from whether the speech sounds good.

## Include enough to reproduce an experiment

Record the Git commit, commands, config and seed, dataset/manifest hashes, package versions, hardware and checkpoint step/hash. Attach comparable development audio with the same text and settings. Explain whether you used recorded or predicted durations/prosody. When reporting performance, separate first-load time, generation time, audio duration, model file size and peak memory.

Keep datasets, checkpoints, generated exports, credentials and local machine paths out of commits. Use an external artifact link for weights or large samples, with a checksum and license. The repo's `runs/`, `data/`, `assets/model/` and `assets/vocoder/` folders are ignored. Do not change frozen benchmark WAVs or reference scores to hide a regression.

Original contributions use this repository's MIT license. Preserve third-party attribution and check the license of any new data, weights or code. Describe the problem, your change and the checks you actually ran in the pull request; mention checks you could not run.
