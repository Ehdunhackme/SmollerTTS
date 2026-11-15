# First full-data baseline

Run `full-baseline-v1`, seed 17, completed 50,000 steps from random initialization
in 2,343.60 seconds on one NVIDIA A100 SXM4 40 GB. Every batch contained one clip
from each of the five languages. The run used 40,940 training recordings, about
84.63 hours. Data-loader waiting totaled 13.01 seconds. Peak PyTorch CUDA tensor
allocation was 630,076,416 bytes; this excludes CUDA context and other GPU overhead.
Some CPU-only preparation overlapped the latter part of training. The measured
wall time is reported as observed, not as an isolated GPU throughput benchmark.

The shared network has 757,828 learned inference parameters. Its fixed renderer
has zero learned parameters. Final checkpoint SHA256:
`a8e82c96e175a8f70106a80c5fdd4e29b45bad28db436f0667a78346e8ada065`.

## Fixed development results

Each row uses the same 20 held-out development sentences and the frozen
Whisper-small protocol. These are corpus WER values, except Mandarin, which uses
CER. They are not naturalness ratings or public benchmark rankings.

| Language | Original recordings | XiaoTTS text input | Released SanoTTS voice | True-control reconstruction |
| --- | ---: | ---: | ---: | ---: |
| English | 5.16% | 11.68% | 5.43% | 5.71% |
| Mandarin | 9.79% | 22.38% | 17.13% | 15.21% |
| Spanish | 4.76% | 12.78% | 8.52% | 8.27% |
| French | 21.82% | 26.75% | 26.49% | 25.97% |
| German | 11.35% | 19.94% | 42.33% | 14.11% |

XiaoTTS had one frontend failure in Mandarin because a compatibility-normalized
footnote becomes a digit; the frontend explicitly lacks number normalization.
That sentence remains in the score as an empty output. All other generated
baseline outputs passed the declared validity checks. Both comparison conditions
had zero invalid outputs. The reconstruction column receives features from the
reference recordings and is diagnostic evidence, never text-to-speech performance.

The released SanoTTS systems are five separate voice models, totaling 4,435,909
learned parameters: English heart-nano, Chinese xiaoya, Spanish tiny, French and
German tiny. XiaoTTS shares one model. The systems use different speakers and
training data, and this small literary development panel can favor the domain
used to train XiaoTTS. The German difference does not establish overall superiority.
See `external-comparisons.md` for the pinned release, source verification and
same-prompt browser/ASR procedure. Original-recording recognition error, especially
in French, limits absolute interpretation.

## Diagnostic evidence for the first trial

| Language | Known phones, predicted durations | Known phones, true durations |
| --- | ---: | ---: |
| English | 9.78% | 9.51% |
| Mandarin | 18.88% | 17.48% |
| Spanish | 11.78% | 11.78% |
| French | 25.45% | 25.97% |
| German | 18.10% | 20.86% |

These conditions receive phonemes from reference alignments, so neither is a TTS
benchmark. Exact durations offer little consistent improvement. Accurate acoustic
controls still offer headroom, particularly in English and German. The first
declared trial therefore changes only shared width from 128 to 144, producing
948,164 parameters. It starts from random weights with the same seed, batch
composition and 50,000-step budget. This is a hypothesis, not a predicted win.
The existing promotion tolerances, final-test separation and repeated-seed
requirement remain in force.

Final-test evaluation and fluent listening evidence are pending. The user has
confirmed they can judge all five languages; a blinded local listening panel is
being prepared. Its ratings will be reported as exploratory author feedback,
not an independent multi-listener MOS study.
