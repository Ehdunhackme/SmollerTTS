# Sources and separate assets

XiaoTTS code is MIT licensed. Training audio, pretrained annotation/evaluation tools,
and pronunciation dictionaries have separate provenance and license obligations.
No pretrained TTS weights or generated teacher audio initialize the speech model.

| Component | Purpose | Source |
| --- | --- | --- |
| LJSpeech 1.1 | Existing English recordings; public domain in the US | https://keithito.com/LJ-Speech-Dataset/ |
| CSS10 | Existing Mandarin, Spanish, French, German recordings; linked Kaggle dataset listings report CC0 | https://github.com/Kyubyong/css10 |
| Montreal Forced Aligner | Offline alignment; version 3.3.10 in the initial environment | https://montreal-forced-aligner.readthedocs.io/ |
| MFA acoustic models, G2P and dictionaries | Offline annotations and pronunciation data; selected MFA model cards list CC BY 4.0 | https://mfa-models.readthedocs.io/ |
| eSpeak NG | External pronunciation/stress annotation executable, GPL-3.0; initial native frontend dependency | https://github.com/espeak-ng/espeak-ng |
| WORLD / PyWORLD | Offline acoustic analysis and oracle comparison, excluded from neural inference | https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder |
| faster-whisper / Whisper small | Offline multilingual intelligibility diagnostic, excluded from TTS inference | https://github.com/SYSTRAN/faster-whisper |

MFA model/dictionary authors: Michael McAuliffe and Morgan Sonderegger. The concrete
downloaded model versions and metadata are preserved in the pretrained model archives,
and source corpus listings/checksums are saved under `data/provenance/`.

Architectural references (independent implementation):

- FastSpeech 2: https://arxiv.org/abs/2006.04558
- DDSP: https://arxiv.org/abs/2001.04643
- Neural source-filter synthesis: https://arxiv.org/abs/1810.11946
- Tiny Voca's published source-filter design: https://github.com/n4ze3m/tiny-voca/blob/main/docs/model.md

Browser dependency ephone 1.0.2 (https://github.com/sjmik/ephone-js) is a modified
eSpeak NG phonemizer under GPL-3.0-or-later. Its bundled license and corresponding
upstream source are linked by the demo. Its rule-based pronunciation code contains
no learned neural parameters. ONNX Runtime Web is MIT licensed; its license is also
bundled. The browser build produces a hash/byte inventory including all dictionaries,
phonemizer packs, runtime binaries, JavaScript and neural graphs. A small neural
parameter count does not describe the size of these supporting assets.
