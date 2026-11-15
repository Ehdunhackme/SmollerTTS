# Browser validation, 15 November 2026

Checkpoint: `full-baseline-v1/step-010000.pt`, 757,828 learned parameters.
SHA256: `6c0c9d2ae697c6f8756ef746c322000159d7ca16d404fe25977940ddeb3c9764`.
This is an early checkpoint while the 50,000-step run continues. Its speech quality
has not yet been established; these checks measure deployment, not naturalness.

## Tested

- Playwright Chromium 156.0.8078.4 on a Windows PC with an Intel Core i5-12500H
  (12 cores / 16 logical processors), launched with
  `--disable-gpu`. ONNX sessions explicitly use WASM CPU with one thread.
- Five actual-text examples, one per language. Same encoder/decoder weights reused.
- All five examples repeated after disabling network access and reloading the page.
  No external requests or page errors. No server performs inference.
- Audio was finite and non-silent, with no sample clipping in the ten cases.
- WAV download, explicit number-input errors, and 375px layout without horizontal overflow.
- Native/browser pronunciation tokens, Mandarin tones and stress agreed on the five
  reference examples. This is not an exhaustive pronunciation audit.
- Native PyTorch versus browser WASM encoder/decoder parity for both input lengths
  in every language (ten cases): largest absolute error 2.44e-5, within
  atol=2e-5/rtol=2e-4. Checkpoint identity and all output shapes also match.
- Fixed DSP versus PyTorch with identical excitation: maximum waveform error
  2.69e-7, relative RMS error 8.71e-7. The independent harmonic oscillator's relative
  RMS error was 0.000304. Noise generators differ but implement the same distribution.

## Measured on this PC

These are short examples on one desktop, not cross-device performance guarantees.
Warm model inference excludes initialization; first-language dictionary loading is
included. Audio is generated as a complete sentence before playback; no streaming
latency claim is made.

| Language | Audio duration | Online generation | Offline generation |
| --- | ---: | ---: | ---: |
| English | 2.16 s | 0.231 s | 0.389 s |
| Mandarin | 3.69 s | 0.317 s | 0.542 s |
| Spanish | 2.71 s | 0.274 s | 0.452 s |
| French | 2.77 s | 0.260 s | 0.471 s |
| German | 2.71 s | 0.216 s | 0.356 s |

Neural graphs: 3,073,348 bytes. Versioned application assets: 35,283,940 raw bytes,
excluding the small generated inventory and service-worker script themselves.
Most bytes are pronunciation dictionaries and ONNX Runtime; parameter count alone
does not describe deployment size. Download time over the Internet has not been
measured. The local cold load/cache check is not an Internet download estimate.

All 487,131 pronunciation entries are stored in compact binary records and looked
up without constructing nested JavaScript arrays for the whole dictionary. The
packed files were verified against every source JSON entry, and the complete
five-language online/offline test was rerun after the change.

A separate Windows memory check sampled the isolated Chromium process tree every
50 ms. Peak private committed memory was 366,452,736 bytes (blank browser:
68,005,888 bytes); the peak sum of working sets was 502,652,928 bytes. These include
browser startup, caching, pronunciation, runtime and generation. Private committed
bytes are not resident RAM; summing working sets can double-count shared pages.
The preceding JSON-table debug build reached 683,458,560 private bytes in the same
test procedure. This comparison includes a checkpoint change and is not a precise
isolated estimate of the dictionary-format effect.

The source version and exact generated file hashes are in the saved asset inventory.
Raw test results, screenshots, WAV files and neural parity output are in the ignored
`artifacts/browser/` directory. The browser model assets are also ignored by Git.
The preceding debug model and its validation evidence are preserved under
`artifacts/browser-model-backups/20261115T135544Z-7d31f9/`. Future installations
must pass the checkpoint/vocabulary/file-hash checks before repeating browser QA.

## Remaining limitations

- Unknown dictionary words and digits fail explicitly; users must spell numbers out.
- Earlier debug models had poor generalization. Faster-than-real-time synthesis does not
  establish intelligibility, pleasantness, Mandarin tone accuracy or benchmark rank.
- The ephone wrapper treats isolated function words differently from the native
  annotation executable; a documented stress rule restores the tested cases.
  Broader pronunciation parity remains to be audited.
- Only Chromium was tested. Mobile screenshots are responsive-layout checks on
  desktop Chromium, not performance or audio testing on a physical phone.
