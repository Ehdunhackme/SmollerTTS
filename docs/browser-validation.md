# Browser validation, 15 November 2026

Checkpoint: `debug-train-v2`, 757,828 learned parameters. This is a poor-quality
early voice, not a final model or a naturalness benchmark.

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
- Native PyTorch versus browser WASM encoder/decoder parity for all language IDs
  and two input lengths: largest absolute error 4.06e-5, within atol=2e-5/rtol=2e-4.
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
| English | 2.14 s | 0.248 s | 0.408 s |
| Mandarin | 4.13 s | 0.308 s | 0.560 s |
| Spanish | 2.48 s | 0.239 s | 0.385 s |
| French | 3.07 s | 0.308 s | 0.560 s |
| German | 2.80 s | 0.364 s | 0.384 s |

Neural graphs: 3,073,348 bytes. Versioned application assets: 35,281,833 raw bytes,
excluding the small generated inventory and service-worker script themselves.
Most bytes are pronunciation dictionaries and ONNX Runtime; parameter count alone
does not describe deployment size. Download time over the Internet has not been
measured. The local cold load/cache check is not an Internet download estimate.

All 487,131 pronunciation entries are stored in compact binary records and looked
up without constructing nested JavaScript arrays for the whole dictionary. The
packed files were verified against every source JSON entry, and the complete
five-language online/offline test was rerun after the change.

A separate Windows memory check sampled the isolated Chromium process tree every
50 ms. Peak private committed memory was 370,716,672 bytes (blank browser:
70,717,440 bytes); the peak sum of working sets was 501,440,512 bytes. These include
browser startup, caching, pronunciation, runtime and generation. Private committed
bytes are not resident RAM; summing working sets can double-count shared pages.
The preceding JSON-table build reached 683,458,560 private bytes in the same test.

The source version and exact generated file hashes are in the saved asset inventory.
Raw test results, screenshots, WAV files and neural parity output are in the ignored
`artifacts/browser/` directory. The browser model assets are also ignored by Git.

## Remaining limitations

- Unknown dictionary words and digits fail explicitly; users must spell numbers out.
- Voice quality/generalization is poor. Faster-than-real-time synthesis does not
  establish intelligibility, pleasantness, Mandarin tone accuracy or benchmark rank.
- The ephone wrapper treats isolated function words differently from the native
  annotation executable; a documented stress rule restores the tested cases.
  Broader pronunciation parity remains to be audited.
- Only Chromium was tested. Mobile screenshots are responsive-layout checks on
  desktop Chromium, not performance or audio testing on a physical phone.
