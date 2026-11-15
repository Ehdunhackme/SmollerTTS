# Browser validation, 15 November 2026

Checkpoint: `full-baseline-v1/latest.pt`, finished at 50,000 steps, 757,828 learned parameters.
SHA256: `a8e82c96e175a8f70106a80c5fdd4e29b45bad28db436f0667a78346e8ada065`.
The fixed native development evaluation is in `baseline-results.md`. These browser
checks measure deployment and numerical agreement, not naturalness.

## Tested

- Playwright Chromium 156.0.8078.4 on a Windows PC with an Intel Core i5-12500H
  (12 cores / 16 logical processors), launched with
  `--disable-gpu`. ONNX sessions explicitly use WASM CPU with one thread.
- Five actual-text examples, one per language. Same encoder/decoder weights reused.
- All five examples repeated after disabling network access and reloading the page.
  No external requests or page errors. No server performs inference.
- Audio was finite and non-silent, with no sample clipping in the ten cases.
- WAV download, explicit number-input errors, and 375px layout without horizontal overflow.
- Native/browser pronunciation tokens, Mandarin tones, stress and segmentation
  agree on all 99 supported sentences in the 100-prompt development panel. One
  Mandarin number-normalization failure is retained in both implementations.
- The actual offline browser also rendered the entire 100-prompt panel, preserving
  the same one failure. All 99 successful renders had exact frontend labels, with
  zero page errors or external requests. Its WAVs and complete asset inventory are
  sealed under `artifacts/browser50k-dev-v1/`; browser-audio ASR scoring is pending.
- Native PyTorch versus browser WASM encoder/decoder parity for both input lengths
  in every language (ten cases): largest absolute error 4.39e-5, within
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
| English | 2.08 s | 0.222 s | 0.366 s |
| Mandarin | 3.82 s | 0.328 s | 0.564 s |
| Spanish | 2.72 s | 0.248 s | 0.422 s |
| French | 2.51 s | 0.243 s | 0.429 s |
| German | 2.67 s | 0.221 s | 0.373 s |

Neural graphs: 3,073,348 bytes. Versioned application assets: 33,310,808 raw bytes,
excluding the small generated inventory and service-worker script themselves.
Most bytes are pronunciation dictionaries and ONNX Runtime; parameter count alone
does not describe deployment size. Download time over the Internet has not been
measured. The local cold load/cache check is not an Internet download estimate.

All 487,131 pronunciation entries are stored in compact binary records and looked
up without constructing nested JavaScript arrays for the whole dictionary. The
packed files were verified against every matching native-labeled JSON cache entry,
and the complete five-language online/offline test was rerun after the change.
Stress is compiled offline using the exact native frontend, rather than approximated
in JavaScript. This fixed stress differences in 15 development sentences and removed
the runtime phonemizer. It does not add any learned inference parameters.

A separate Windows memory check sampled the isolated Chromium process tree every
50 ms. Peak private committed memory was 361,652,224 bytes (blank browser:
68,222,976 bytes); the peak sum of working sets was 500,264,960 bytes. These include
browser startup, caching, pronunciation, runtime and generation. Private committed
bytes are not resident RAM; summing working sets can double-count shared pages.
The preceding JSON-table debug build reached 683,458,560 private bytes in the same
test procedure. This comparison includes a checkpoint change and is not a precise
isolated estimate of the dictionary-format effect.

The source version and exact generated file hashes are in the saved asset inventory.
Raw test results, screenshots, WAV files and neural parity output are in the ignored
`artifacts/browser/` directory. The browser model assets are also ignored by Git.
The preceding 10,000-step model and its validation evidence are preserved under
`artifacts/browser-model-backups/20261115T142412Z-04065b/`. Future installations
must pass the checkpoint/vocabulary/file-hash checks before repeating browser QA.

## Remaining limitations

- Unknown dictionary words and digits fail explicitly; users must spell numbers out.
- Earlier debug models had poor generalization. Faster-than-real-time synthesis does not
  establish intelligibility, pleasantness, Mandarin tone accuracy or benchmark rank.
- The pronunciation tables inherit the native dictionary/annotation pipeline's
  limitations. Exact implementation agreement does not prove correct pronunciation
  for every context or Mandarin reading.
- Only Chromium was tested. Mobile screenshots are responsive-layout checks on
  desktop Chromium, not performance or audio testing on a physical phone.
