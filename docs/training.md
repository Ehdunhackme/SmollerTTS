# Train SmollerTTS

This is the training path for the **1,882,807-parameter acoustic model** used in the current release. It starts from random weights. The **925,985-parameter HiFi-GAN LJ_V2** vocoder is pretrained and frozen; the acoustic training loss does not run the vocoder or a discriminator.

The repo includes data preparation, the training loop, checkpoint resume, listening samples and ONNX export. The new exporter has been checked against the original release checkpoint. This is not the older experimental model from the Git history.

## 1. Install

Use Linux or WSL for dataset preparation and GPU training. Python 3.11 is a good starting point. MFA runs in a separate Conda environment. The small CPU training test also works on Windows.

From the cloned repository:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[train,export,dev]"
python -m smollertts.smoke --output runs/smoke
python -m pytest tests/training -q
```

Install a [PyTorch 2.6 build](https://pytorch.org/get-started/previous-versions/#v260) appropriate for your CUDA driver if the default package cannot use your GPU. The training code supports PyTorch 2.6–2.7. Feature extraction also needs [eSpeak NG](https://github.com/espeak-ng/espeak-ng): on Ubuntu, `sudo apt-get install espeak-ng`.

The smoke test runs four updates on synthetic feature arrays. It checks the code path without downloading a dataset; its checkpoint is **not a usable voice**. Tests also compare uninterrupted and resumed training, reject changed manifests, and check the audio feature conventions.

## 2. Get LJSpeech and create the alignment corpus

Download [LJSpeech 1.1](https://keithito.com/LJ-Speech-Dataset/), then extract it under `data/`:

```sh
mkdir -p data
curl --fail --location https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -o data/LJSpeech-1.1.tar.bz2
tar -xjf data/LJSpeech-1.1.tar.bz2 -C data
python -m smollertts.prepare corpus --ljspeech data/LJSpeech-1.1 --output data/corpus
```

This reads the normalized transcripts in `metadata.csv`, writes `.lab` files and links or copies the WAVs. It reserves chapter **LJ043 for development** and **LJ048 for final testing**. Other chapters supply training data. Exact repeated transcripts are removed, with evaluation taking precedence over training. Test WAVs are not read or copied by default.

The released run used 12,134 accepted training recordings (about 22.12 hours), 186 development recordings and 286 reserved test IDs. Its complete accepted-recording list has not been recovered. This contributor pipeline uses the recovered chapter split and explicit validation filters, so counts can differ. It reproduces the architecture and training recipe, **not the exact original data selection or checkpoint bytes**.

## 3. Align the recordings

Install [Montreal Forced Aligner 3.3](https://montreal-forced-aligner.readthedocs.io/en/v3.3.5/user_guide/workflows/alignment.html) separately from the PyTorch environment:

```sh
conda create -n smoller-align -c conda-forge "montreal-forced-aligner=3.3.5"
conda run -n smoller-align mfa model download acoustic english_mfa
conda run -n smoller-align mfa model download dictionary english_us_mfa
conda run -n smoller-align mfa align data/corpus/corpus english_us_mfa english_mfa data/aligned --output_format json --single_speaker --num_jobs 4 --no_tokenization --clean
```

Use the English MFA **IPA** models, not an ARPAbet dictionary. Save the MFA version, downloaded model archives and their checksums with your experiment notes; the model names alone do not pin their contents. Inspect MFA's OOV and alignment reports before proceeding.

## 4. Extract acoustic targets

Back in the PyTorch environment:

```sh
python -m smollertts.prepare features --corpus data/corpus --alignments data/aligned --output data/features --workers 4
```

The preparation code:

- Keeps the released 132-entry phone vocabulary and integer IDs. The English model uses a subset of this inherited inventory; unused entries stay in place for browser compatibility.
- Re-grids aligned phones to native HiFi-GAN frame centres, at sample `128 + t * 256`. A very short phone can occupy zero frames.
- Extracts **magnitude** mel features: 22,050 Hz mono, FFT/window 1024, hop 256, 80 bands from 0–8,000 Hz, reflect padding 384 and natural-log floor `1e-5`. It does not peak-normalize the audio.
- Uses WORLD DIO + StoneMask for recorded pitch and voicing, and recomputes RMS energy on the native windows. eSpeak supplies lexical stress when its syllable count matches the aligned word; match counts are recorded.
- Computes mel mean and standard deviation on **training frames only**. Every manifest, audio source, alignment and feature file has a checksum.

Inspect `data/features/rejected.json` and the per-recording stress match counts. Unknown/noise phones, invalid audio and malformed alignments are rejected. Preparation stops if more than 1% of recordings fail (with a five-recording allowance for small fixtures); it never silently maps missing words to silence.

Each NPZ contains:

| Array | Shape | Convention |
| :--- | :--- | :--- |
| `tokens` | N | int64; 0 padding, 1 silence |
| `stress` | N | int64; 0, 1 or 2 |
| `durations` | N | int64; nonnegative, sum = T |
| `mel` | T × 80 | float32 native log-mel values |
| `prosody` | T × 3 | float32 normalized log-F0, binary voicing, normalized log-RMS |

F0 normalization is `(log(F0) - log(180)) / 0.5`; energy is `(log(RMS) + 3) / 2`. Phone prosody targets are frame averages inside each aligned phone. Feature paths in the manifests are relative, so the prepared directory can be moved to a training server.

For a separate final evaluation, create **new** corpus and feature directories and pass `--include-test` to both preparation commands. Training still reads only `train.jsonl`; routine evaluation reads only `dev.jsonl`. Do not tune on the reserved chapter. The pretrained vocoder may already have seen these LJSpeech recordings, so this is an acoustic holdout, not proof of fully unseen end-to-end data.

## 5. Train and resume

```sh
python -m smollertts.train --data data/features --output runs/acoustic --config configs/acoustic.json --device cuda --workers 4 --verify-data
```

[`configs/acoustic.json`](../configs/acoustic.json) contains the release recipe:

| Setting | Value |
| :--- | :--- |
| Initialization / seed | Random / 2718 |
| Updates | 10,000 |
| Batch | Complete utterances; up to 24 examples, 12,000 padded frames |
| Optimizer | AdamW, betas (0.9, 0.98), weight decay 0.01 |
| Learning rate at step s | `5e-4 * min(1, s/300) * 0.5**(s/20000)` |
| Gradient norm limit | 5 |
| Arithmetic | FP32; CUDA TF32 enabled |
| Decoder conditioning | Recorded prosody during training; predicted prosody during evaluation |

The objective is mel L1 + 0.50 × adjacent-frame mel-difference L1 + 0.50 × log-duration MSE + 0.15 × frame prosody loss + 0.10 × phone prosody loss. Prosody losses combine voiced F0 L1, energy L1 and voicing BCE. Padding is masked. See [`loss.py`](../smollertts/loss.py).

Outputs:

- `latest.pt` and `checkpoints/step-XXXXXX.pt`: acoustic weights, normalization buffers, optimizer, RNG state, recipe and data/source identity.
- `metrics.jsonl`: training losses, learning rate, gradient norm, step time and batch identity.
- `dev-XXXXXX.json`: development losses using **aligned durations and predicted prosody**. These are diagnostic losses, not free-running speech-quality scores.
- `status.json`: current step, state, environment and data summary.

Checkpoints are saved every 1,000 steps and at a clean stop. Ctrl+C or SIGTERM requests a save after the current update; a hard kill or power loss can only recover the last completed save. Files are written atomically. Use a **new output directory** when resuming:

```sh
python -m smollertts.train --data data/features --output runs/acoustic-resumed --config configs/acoustic.json --device cuda --workers 4 --resume runs/acoustic/latest.pt --verify-data
```

For a planned early stop, add `--stop-after 2000`; keep the full recipe unchanged. Resume checks the recipe, manifests, training source, device, thread count and PyTorch/NumPy versions. It refuses a mismatch instead of quietly changing an experiment. The CPU regression test verifies bit-identical resumed weights and optimizer state; that is not a guarantee across different GPU hardware or drivers.

If you need a smaller batch, edit a copy of the config **before starting** the run. Use `--device cpu --threads 4` for debugging, not as an assumed fast route to a trained voice. Ten thousand updates describe the released run, not a guarantee of convergence or quality for every data preparation.

## 6. Listen with predicted timing

Download the original frozen vocoder, then render development examples:

```sh
python -m smollertts.assets --output assets/vocoder
python -m smollertts.infer --checkpoint runs/acoustic/latest.pt --output runs/listening
```

The downloader verifies the original config and generator by byte count and SHA-256. If Google Drive blocks automated access, use the **LJ_V2** folder linked from the [official HiFi-GAN repository](https://github.com/jik876/hifi-gan#pretrained-model), save `config.json` and `generator_v2` in `assets/vocoder/`, and rerun the command to verify them. Expected hashes are in [`assets.py`](../smollertts/assets.py). No account or API key is required.

The six included development prompts are for listening, not benchmark scoring. Inference uses predicted durations and predicted prosody, so it can expose problems hidden by aligned-duration losses. WAVs are accompanied by checkpoint hashes and sample metadata. This native path does not apply the browser's optional punctuation or delivery controls.

To try your own text with the same normalization and pronunciation frontend as the browser:

```sh
npm ci
npm run setup
node scripts/phonemize.mjs --text "Hello, this is a new checkpoint." --output runs/prompt.json
python -m smollertts.infer --checkpoint runs/acoustic/latest.pt --prompts runs/prompt.json --output runs/custom-listening
```

## 7. Export and try it in the browser

```sh
python -m smollertts.export --checkpoint runs/acoustic/latest.pt --output runs/export
npm run build -- --model-dir runs/export
npm start
```

Open **http://127.0.0.1:5175/projects/smollertts/**. The custom build needs the pinned runtime and frontend downloaded by `npm run setup` above. It keeps them fixed and replaces only the three neural graphs. It does not overwrite the published asset manifest or upload anything.

Export checks dynamic phone/frame lengths, duration rounding, mel output and waveform agreement between PyTorch and ONNX Runtime. Graph optimization is disabled to match browser inference. `model.json` is written only after these checks pass. `export.json` records numerical errors, library versions and checksums. A failed parity check should be investigated, not bypassed to publish an export.

Custom builds have a distinct cache identity and a visible local-checkpoint notice. The chart still describes the **published model**, not your new weights. Restore the published demo with `npm run build` without `--model-dir`.

The current browser contract expects this exact architecture and vocabulary. Changes to dimensions, graph inputs or token IDs need corresponding exporter/frontend changes and tests; a parameter-count match alone is not sufficient evidence of compatibility. Test the actual browser output before sharing a new model. The frozen [`browser_smoke.py`](../tests/browser_smoke.py) fixtures apply to the published voice and should not be rewritten just to make a new voice pass.

With the custom build served in another terminal, compare its actual WASM output to your PyTorch checkpoint:

```sh
python -m pip install playwright==1.64.1
python -m playwright install chromium
python tests/export_browser.py --checkpoint runs/acoustic/latest.pt
```

This checks the served checkpoint hash, predicted durations and waveform error on the development prompts, without relying on the old voice's WAV fixtures. It compares the native model path with browser delivery controls disabled.

## What is and is not reproduced

The public `AcousticModel` loads every tensor in the original 10,000-step checkpoint with strict matching. The training objective and native mel conventions are retained. The contributor trainer adds portable manifests, safe checkpoint loading, optimizer/RNG resume and validation around that recipe. Native PyTorch checkpoints and datasets are not bundled in Git; the published inference files remain on Hugging Face.

A complete fresh LJSpeech training run has **not** been performed with this packaged pipeline. Local validation covers synthetic CPU training, exact resume, synthetic WAV feature extraction, the original checkpoint's native/ONNX export parity, and browser waveform regression. Full-corpus alignment and final listening quality remain checks to perform on a new run. Record your environment and data hashes, compare saved checkpoints, and listen before claiming an improvement.

See [CONTRIBUTING.md](../CONTRIBUTING.md) for useful places to help.
