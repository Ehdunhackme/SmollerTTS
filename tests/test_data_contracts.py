import json
import numpy as np
import pytest
import soundfile as sf
from xiaotts.manifest import build
from xiaotts.phones import split_phone


def test_mandarin_tones_are_explicit_and_unknown_contours_fail():
    assert split_phone("a˧˥") == ("a", 2, 0)
    assert split_phone("a˨˩˦") == ("a", 3, 0)
    assert split_phone("ˈaː") == ("aː", 0, 1)
    with pytest.raises(ValueError):
        split_phone("a˦˦˦")


def test_float_wav_corpora_remain_usable_and_passage_splits_do_not_overlap(tmp_path, monkeypatch):
    path = tmp_path/"float.wav"
    sf.write(path, np.sin(np.arange(22050)*0.1).astype(np.float32)*0.1, 22050, subtype="FLOAT")
    (tmp_path/"provenance").mkdir()
    def fake_rows(root, language):
        for index in range(500):
            yield dict(id=f"{language}_{index}", language=language, audio=str(path),
                       text=f"{language} sentence {index}", book="one-book", source_index=index)
    monkeypatch.setattr("xiaotts.manifest.source_rows", fake_rows)
    for language in ("en", "zh", "es", "fr", "de"):
        (tmp_path/"provenance"/f"{language}.json").write_text("{}")
    build(tmp_path, minimum_holdout=20)
    rows = [json.loads(line) for line in (tmp_path/"manifests/all.jsonl").read_text().splitlines()]
    for language in ("en", "zh", "es", "fr", "de"):
        groups = [{row["group"] for row in rows if row["language"] == language and row["split"] == split}
                  for split in ("train", "dev", "test")]
        assert all(groups)
        assert groups[0].isdisjoint(groups[1]) and groups[0].isdisjoint(groups[2]) and groups[1].isdisjoint(groups[2])
