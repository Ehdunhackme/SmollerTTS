from copy import deepcopy
from xiaotts import LANGUAGES
from xiaotts.selection import compare


def test_one_language_cannot_hide_another_regression():
    baseline = dict(status='finished', panel_sha256='fixed', sources={'evaluator':'fixed'}, parameters=757828,
        per_language={language: dict(metric='CER' if language == 'zh' else 'WER', clips=20,
                                    invalid_fraction=0, predicted_error=0.5) for language in LANGUAGES})
    candidate = deepcopy(baseline)
    candidate['per_language']['en']['predicted_error'] = 0.1
    candidate['per_language']['zh']['predicted_error'] = 0.6
    assert not compare(candidate, baseline)['provisional_promote']
    candidate['per_language']['zh']['predicted_error'] = 0.51
    assert compare(candidate, baseline)['provisional_promote']
    candidate['parameters'] = 1_000_000
    assert not compare(candidate, baseline)['provisional_promote']
