"""Per-language provisional promotion, with no pooled CER/WER ranking."""
from xiaotts import LANGUAGES


def compare(candidate, incumbent, regression_tolerance=0.02, improvement=0.03):
    reasons = []
    if candidate.get('status') != 'finished' or incumbent.get('status') != 'finished':
        reasons.append('Both evaluations must finish')
    if candidate.get('panel_sha256') != incumbent.get('panel_sha256'):
        reasons.append('Evaluation panels differ')
    if candidate.get('sources') != incumbent.get('sources'):
        reasons.append('Evaluator sources differ')
    if not 0 < candidate.get('parameters', 1_000_000) < 1_000_000:
        reasons.append('Parameter ceiling failed')
    improved = []
    for language in LANGUAGES:
        a, b = candidate.get('per_language', {}).get(language), incumbent.get('per_language', {}).get(language)
        if not a or not b:
            reasons.append(f'Missing {language}')
            continue
        if a['metric'] != b['metric'] or a['clips'] != b['clips']:
            reasons.append(f'Incompatible {language} measurements')
        if a['invalid_fraction'] > 0.05:
            reasons.append(f'Too many invalid {language} outputs')
        if a['predicted_error'] > b['predicted_error'] + regression_tolerance:
            reasons.append(f'{language} regressed by more than {regression_tolerance}')
        if a['predicted_error'] < b['predicted_error'] - improvement:
            improved.append(language)
    if not improved:
        reasons.append('No language improved enough to nominate')
    return dict(provisional_promote=not reasons, improved_languages=improved, reasons=reasons,
                requires='Matched training budget, export/CPU checks and repeated seeds before final selection')
