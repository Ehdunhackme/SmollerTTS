# Evaluation

The interface shows a **rough published overview**, with quality or word errors on the vertical axis and reported parameter count on a logarithmic horizontal axis. These results are useful for exploring size and quality, not for declaring a winner.

## Four-prompt overview

| SmollerTTS metric | Result |
| :--- | ---: |
| Mean UTMOS | 3.5539 |
| Mean prompt WER | 6.6919% |
| Complete prompts | 4 / 4 |

SmollerTTS was rendered on browser CPU at natural speed, pitch and energy, with the optional punctuation steering disabled. The four frozen clips are in [audio](../benchmarks/audio/); [wer.json](../benchmarks/wer.json) includes text, transcripts, normalization, per-prompt scores, audio hashes and recognizer settings. WER uses Whisper large-v3 in CPU FP32 at the checkpoint revision recorded in that file, with full-clip 16 kHz mono preprocessing and the Whisper English normalizer. The reported WER is the arithmetic mean of the four prompt WERs, not a corpus-level score. ASR timing fields describe scoring time, not synthesis speed.

Peer results come from [tts-bench](https://github.com/5uck1ess/tts-bench/tree/1dd3a81a50dd21fb2f4bce89df27eada996ba57f), revision `1dd3a81a50dd21fb2f4bce89df27eada996ba57f`. [overview.json](../benchmarks/overview.json) preserves source links and per-prompt results. The selection prefers a complete default-voice Linux CPU run, otherwise a complete CUDA run; it does not join partial runs. Unknown model sizes and incomplete scores are omitted.

The peers were not rerun here. Voices, runtime settings, parameter-count conventions and some evaluator revisions differ. UTMOS is an automatic predictor, not human MOS. Four prompts give very weak coverage. Prompt ranges show variation across those prompts; they are not confidence intervals. The step line is the best observed score within each size budget in this selected set, not a scaling law.

## Separate 48-sentence comparison

This panel uses eight prompts in each of six categories: everyday speech, narration, questions/dialogue, numbers/units, names/technical terms and long passages. The [prompts](../benchmarks/matched/prompts.json) and [full results](../benchmarks/matched/results.json) include per-sample results, model provenance and bootstrap settings.

| Model / voice | Parameters | UTMOS | Corpus WER |
| :--- | ---: | ---: | ---: |
| SmollerTTS / LJ | 2.81M | 3.7713 | 0.9186% |
| tiny-voca / F2 | 4.54M | 4.4347 | 0.2625% |
| tiny-voca / M2 | 4.54M | 4.3254 | 0.3937% |

All three voices completed 48 renders. This panel uses faster-whisper **small.en**, CPU INT8, and corpus WER: total word edits divided by total reference words. It is a different recognizer, prompt set and aggregation from the overview, so the WER numbers must not be compared across the two tables.

After inspecting interim ASR output, time and unit formatting was normalized symmetrically in every model’s references and hypotheses. That amendment was not part of the initial protocol. The raw normalized results remain in the JSON; SmollerTTS’s pre-amendment WER was 1.7060%. The reported intervals use 10,000 paired, category-stratified bootstrap resamples and omit evaluator and voice-selection uncertainty.

This is a handwritten development panel with one render per voice and sentence, not an independent human listening study. The results do **not** establish that SmollerTTS matches or beats Tiny Voca’s naturalness. The four overview audio files are included in Git; the full 144-render archive is not included in this lightweight repository.
