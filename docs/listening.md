# Blinded development listening

The user confirmed they can assess English, Mandarin, Spanish, French and German.
Their ratings will be exploratory author feedback, not an independent multi-listener
MOS study. ASR scores and frontend parity cannot establish naturalness or correct tones.

`scripts/listening_panel.py` builds a local comparison from completed evaluations
on the same frozen development panel. It selects the first N preordered prompts in
each language without inspecting their quality, keeps failures visible, and shuffles
anonymous voice letters for each sentence. Original generated audio is copied without
volume normalization or enhancement. The system mapping and audio hashes stay outside
the served directory. There is no final-test material in this panel.

Provide a JSON specification with a seed and two to four named systems, each containing
`report` (a completed evaluation JSON path) and `directory` (its WAV directory):

```bash
python scripts/listening_panel.py build --panel evaluation/dev-v1.json --spec artifacts/listening-spec.json --output artifacts/listening-finalists --count 2
python scripts/listening_panel.py serve artifacts/listening-finalists --port 8766
```

Open the printed localhost address. Rate clarity, naturalness and pronunciation
separately from one to five, note incorrect words or tones, and save. Partial ratings
can be resumed. They are stored in `ratings.json` outside the web root. The panel
identity is checked when saving, so ratings cannot silently be applied to another panel.

An initial baseline/SanoTTS panel with two sentences per language is prepared locally
under `artifacts/listening-baseline-v1`. It has no human ratings yet. A separate
disposable copy is used for the browser UI check; synthetic test ratings are never
written to the human panel. Finalist listening remains pending model selection.
