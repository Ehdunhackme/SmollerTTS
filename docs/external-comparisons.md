# External comparison protocol

The released [SanoTTS](https://github.com/Ampixa/sanoTTS) browser implementation is
pinned to commit `8ba78d4f9bbd164cc0d3ddc36beba2c5a0c5530e`. It is an external
reference, not a XiaoTTS component or training source. Its code and model files
remain under ignored `artifacts/comparators/`, with upstream licenses preserved.

The chosen released voices are heart-nano for English, Chinese xiaoya, Spanish
tiny, French, and German tiny. This is **five separate models**, whereas XiaoTTS
shares one set of weights across all five languages. The released voices also
use different training data and speakers. Results can compare intelligibility
on the same prompts, but cannot establish which training method is better.

`scripts/fetch_sanotts.py` downloads only the browser dependencies and selected
voices from that exact revision. Every file is verified against its Git blob
hash and recorded with SHA256. `scripts/render_sanotts.py` serves the unchanged
upstream page locally, pins voice loading to local files, blocks external
requests and runs Chromium with the GPU disabled. The page's own frontend and
synthesizer receive all 100 unmodified development prompts. Failure cases remain
in the result. The final-test manifest is never opened.

`scripts/score_comparator.py` checks the same frozen panel and evaluator source
hashes, uses the same exact Whisper weights and decoding settings, and reports
per-language WER/CER and invalid-output rates under the existing protocol.
Original recordings are recognized again as a calibration check. No ASR score
is presented as a listening or naturalness score.

The comparator's reported `render_seconds` comes from its own synthesis timing.
The wrapper also measures the complete call, including text pronunciation and
first-use local asset loading. These measurements are not internet download
times. Browser/runtime conditions and all failure details are saved. A launch
that initializes additional unused upstream runtimes must not be represented
as the minimal memory or download requirement of one selected voice.

```powershell
C:\Python313\python.exe -X utf8 scripts/fetch_sanotts.py
C:\Python313\python.exe -X utf8 scripts/render_sanotts.py
C:\Python313\python.exe -X utf8 scripts/seal_render.py artifacts/comparators/sanotts-dev-v1
```

Copy the resulting render directory to the server and run:

```bash
.venv/bin/python -m scripts.score_comparator --directory artifacts/comparators/sanotts-dev-v1
```

GPU ASR is scheduled after baseline training/review so it does not compete with
the training run during its throughput measurement. Comparison results are
pending; no quality advantage is claimed.

An additional diagnostic, `scripts/render_oracle_panel.py`, supplies acoustic
controls extracted from the original recordings to XiaoTTS's fixed renderer.
It uses the same 100 development prompts/recordings and recognition protocol.
This is explicitly **reference-audio reconstruction, not text-to-speech**. It
helps distinguish renderer limitations from errors made by the text-to-acoustics
model. It is excluded from candidate selection and benchmark claims.
Both rendering directories are sealed with separate SHA256 inventories before
scoring. Scoring verifies each input text, language, audio hash and the report hash.
