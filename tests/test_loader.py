import torch
from xiaotts import LANGUAGES
from xiaotts.train import BalancedSteps


def test_balanced_step_draws_are_order_independent(monkeypatch):
    training = {language: [dict(language=language, id=i) for i in range(20)] for language in LANGUAGES}
    monkeypatch.setattr('xiaotts.train.collate', lambda rows: (rows, [dict(controls=[1], waveform=torch.ones(4))]))
    dataset = BalancedSteps(training, 30, seed=17, per_language=2)
    first, _ = dataset[9]
    dataset[2]
    assert dataset[9][0] == first
    assert len(first) == 10
    assert all(sum(row['language'] == language for row in first) == 2 for language in LANGUAGES)
    assert dataset[10][0] != first
