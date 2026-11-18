# Acoustic training

The released acoustic network was trained from scratch on single-speaker LJSpeech recordings. The pretrained HiFi-GAN LJ_V2 generator was frozen. No vocoder distillation or adversarial loss was used to train the released acoustic weights.

The Python package contains the final neural definitions, batching code and objective. It is not a complete raw-audio training pipeline: dataset downloads, forced alignment, feature extraction, checkpoint orchestration and ONNX export are not exposed as a one-command reproduction. The runnable release uses the verified ONNX files on Hugging Face.

## Data and targets

Forced alignments supply phone durations in mel frames. Training examples contain phoneme IDs, stress, durations, 80-band mel features and recorded frame-level pitch/voicing/energy. Train and development utterance IDs are disjoint. The prepared acoustic training split contains approximately 22.12 hours of accepted recordings from LJSpeech.

The published HiFi-GAN generator may have seen the same LJSpeech recordings during its original training. An acoustic development split does not make this an independently held-out vocoder test.

`smollertts.data` expects each manifest row to contain `id`, `split`, `frames` and a `features` path to an NPZ file:

| Array | Shape | Type / convention |
| :--- | :--- | :--- |
| `tokens` | N | int64 phone IDs; 0 padding, 1 silence |
| `stress` | N | int64, 0/1/2 |
| `durations` | N | int64 frame counts, sum = T |
| `mel` | T × 80 | float32 native mel features |
| `prosody` | T × 3 | float32 normalized log-F0, binary voicing, normalized log-RMS |

Mean and standard deviation for each mel band are computed on the training split and stored as model buffers. Phone-level prosody targets are frame averages within each aligned phone. See [model details](model.md) for the prosody normalization formulas.

## Final recipe

- Random initialization, seed 2718; 10,000 acoustic updates.
- Complete utterances grouped by similar length, up to 24 examples and a 12,000-frame padded batch budget.
- AdamW, betas (0.9, 0.98), weight decay 0.01, gradient norm clipped to 5.
- Learning rate at step `s`: `5e-4 * min(1, s / 300) * 0.5**(s / 20000)`.
- FP32 training; CUDA TF32 matrix multiplication was enabled in the training run.
- Recorded prosody conditions the frame decoder during training. Evaluation and browser synthesis use predicted prosody.

The objective in [loss.py](../smollertts/loss.py) is:

```text
L = mel L1
  + 0.50 × adjacent-frame mel-difference L1
  + 0.50 × log-duration MSE
  + 0.15 × frame prosody loss
  + 0.10 × phone prosody loss
```

Prosody loss combines voiced-frame F0 L1, energy L1 and binary-cross-entropy voicing loss. Padding is masked. Training steps alone do not establish convergence or acceptable speech quality; checkpoints must be checked with predicted-duration, predicted-prosody synthesis and listening.

To inspect the architecture locally, install PyTorch appropriate for your machine, then:

```sh
pip install -e .
python -c "from smollertts import AcousticModel; m = AcousticModel(); print(sum(p.numel() for p in m.parameters()))"
```

The acoustic count should be **1,882,807**. Native PyTorch training checkpoints and optimizer state are not bundled with this inference release. The released ONNX graphs and original checkpoint SHA-256 are recorded in [model/config.json](../model/config.json).
