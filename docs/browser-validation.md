# Browser validation, 15 November 2026

Checkpoint: `debug-train-v2`, 757,828 learned parameters. This is a poor-quality
early voice, not a final model or a naturalness benchmark.

## Tested

- Playwright Chromium 156.0.8078.4 on the development Windows PC, launched with
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
| English | 2.14 s | 0.324 s | 0.468 s |
| Mandarin | 4.13 s | 0.476 s | 0.670 s |
| Spanish | 2.48 s | 0.561 s | 0.552 s |
| French | 3.07 s | 0.775 s | 0.728 s |
| German | 2.80 s | 0.669 s | 0.627 s |

Neural graphs: 3,073,348 bytes. Versioned application assets: 58,968,061 raw bytes,
excluding the small generated inventory and service-worker script themselves.
Most bytes are pronunciation dictionaries and ONNX Runtime; parameter count alone
does not describe deployment size. Download time over the Internet and full browser
memory consumption have not been measured. The local cold load/cache check took
1.01 seconds and is not an Internet download estimate.

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
