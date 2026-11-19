import json

import numpy as np
import pytest
import soundfile as sf
import torch

from smollertts import features
from smollertts.prepare import corpus, process


def test_corpus_keeps_test_audio_unread_and_deduplicates(tmp_path):
    source = tmp_path / "raw"
    (source / "wavs").mkdir(parents=True)
    names = ["LJ001-0001", "LJ001-0002", "LJ043-0001", "LJ048-0001"]
    texts = ["Training example.", "Same as test.", "Development example.", "Same as test."]
    (source / "metadata.csv").write_text("\n".join(f"{name}|{text}|{text}" for name, text in zip(names, texts)), encoding="utf-8")
    # Intentionally no test WAV: default preparation must not read it.
    for name in names[:3]:
        sf.write(source / "wavs" / (name + ".wav"), np.zeros(2205), 22050)
    output = tmp_path / "corpus"
    rows = corpus(source, output)
    assert {r["id"] for r in rows} == {"LJ001-0001", "LJ043-0001", "LJ048-0001"}
    assert all(not (output / "corpus" / (name + ".wav")).exists() for name in ("LJ001-0002", "LJ048-0001"))
    assert "audio" not in next(r for r in rows if r["split"] == "test")
    assert (output / "corpus/LJ001-0001.lab").read_text() == "training example"


def test_native_frame_centres_and_zero_duration_phone(monkeypatch):
    monkeypatch.setattr(features, "word_stress", lambda *_: (1,))
    hop = features.HOP / features.SAMPLE_RATE
    alignment = {"tiers": {"phones": {"entries": [[0, hop, "a"], [hop, hop * 1.1, "b"], [hop * 1.1, hop * 4, "c"]]},
                           "words": {"entries": [[0, hop * 4, "word"]]}}}
    tokens, stress, durations, coverage = features.aligned_phones(alignment, 1024, {"a": 3, "b": 4, "c": 5})
    assert tokens.tolist() == [3, 4, 5]
    assert stress.tolist() == [1, 0, 0]
    assert durations.tolist() == [1, 0, 3]
    assert coverage["stress_words_matched"] == 1
    alignment["tiers"]["phones"]["entries"][0][2] = "spn"
    with pytest.raises(ValueError, match="Unknown-word"):
        features.aligned_phones(alignment, 1024, {})


def test_audio_to_native_features(tmp_path, monkeypatch):
    pytest.importorskip("pyworld")
    torch.set_num_threads(1)
    samples = 22050
    wave = .1 * np.sin(2 * np.pi * 180 * np.arange(samples) / 22050)
    audio = tmp_path / "audio.wav"
    sf.write(audio, wave, 22050, subtype="PCM_16")
    alignment = tmp_path / "alignment.json"
    alignment.write_text(json.dumps({"tiers": {"phones": {"entries": [[0, 1, "a"]]}, "words": {"entries": []}}}))
    (tmp_path / "features").mkdir()
    row, total, _ = process(({"id": "fixture", "text": "fixture", "split": "train"}, audio, alignment, tmp_path, {"a": 3}, "unused"))
    assert "error" not in row, row
    with np.load(tmp_path / row["features"]) as saved:
        assert saved["mel"].shape == (samples // 256, 80)
        assert saved["durations"].sum() == len(saved["mel"])
        voiced = saved["prosody"][:, 1] == 1
        assert voiced.mean() > .8
        assert abs(np.median(saved["prosody"][voiced, 0])) < .1
        assert np.isfinite(total).all()
        # Magnitude-mel log energy rises by log(2), not 2*log(2).
        a = features.native_mel(torch.tensor(wave[:22016], dtype=torch.float32)[None])
        b = features.native_mel(torch.tensor(2 * wave[:22016], dtype=torch.float32)[None])
        active = a > -4
        torch.testing.assert_close((b - a)[active], torch.full_like(a[active], np.log(2)), atol=.003, rtol=0)
