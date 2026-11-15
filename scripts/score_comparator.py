"""Score external comparator WAVs using the unchanged frozen development protocol.

No candidate-promotion claim is made: released voices have different data,
speaker identities and parameter budgets. This is a same-prompt quality reference.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from jiwer import cer, wer
from xiaotts import LANGUAGES
from xiaotts.benchmark import digest, evaluator_sources
from xiaotts.evaluate_renderer import normalize
from xiaotts.fetch import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, default=Path('evaluation/dev-v1.json'))
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    output = args.directory/'scores.json'
    if output.exists():
        raise FileExistsError('Preserve existing scores')
    panel = json.loads(args.panel.read_text())
    rendered = json.loads((args.directory/'render.json').read_text())
    integrity = json.loads((args.directory/'audio-integrity.json').read_text())
    if integrity['render_sha256'] != digest(args.directory/'render.json'):
        raise ValueError('The sealed rendering report changed')
    if rendered['status'] != 'finished' or rendered['panel_sha256'] != digest(args.panel):
        raise ValueError('Incomplete render or different panel')
    if panel['sources'] != evaluator_sources() or any(row['split'] != 'dev' for row in panel['rows']):
        raise ValueError('Changed protocol or non-development input')
    if [item['id'] for item in rendered['samples']] != [row['id'] for row in panel['rows']]:
        raise ValueError('Missing, reordered or extra comparator sentences')
    recognizer_path = Path(panel['recognizer_snapshot']['directory'])
    if panel['recognizer_snapshot']['sha256'] != {name:digest(recognizer_path/name) for name in panel['recognizer_snapshot']['sha256']}:
        raise ValueError('Recognizer weights changed')
    report = dict(status='running', panel_sha256=digest(args.panel), sources=evaluator_sources(),
                  renderer_sha256=digest(args.directory/'render.json'), source=rendered['source'], samples=[],
                  metric_limit='Same-prompt ASR intelligibility reference, no naturalness or matched-training claim')
    recognizer = WhisperModel(str(recognizer_path), device='cuda', compute_type='float16', cpu_threads=4)
    started = time.monotonic()
    for row, rendered_item in zip(panel['rows'], rendered['samples']):
        language = row['language']
        item = dict(rendered_item)
        if any(item[key] != row[key] for key in ('id', 'language', 'text')) or item['reference_seconds'] != row['seconds']:
            raise ValueError('Comparator input differs from its frozen panel')
        if 'failure' in item:
            item.update(invalid=True, predicted_text='')
        else:
            path = args.directory/f"{row['id']}.wav"
            if integrity['audio_sha256'].get(row['id']) != digest(path):
                raise ValueError('Saved comparator audio changed or lacks its hash')
            wave, rate = sf.read(path, dtype='float32')
            if wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all():
                raise ValueError('Malformed saved comparator audio')
            seconds = len(wave)/rate
            rms = float(np.sqrt(np.mean(wave**2)))
            clipped = float((np.abs(wave) >= 1).mean())
            item.update(rms=rms, peak=float(np.abs(wave).max()), clipped_fraction=clipped,
                        duration_ratio=seconds/row['seconds'],
                        invalid=rms < 1e-4 or clipped > 0.01 or not 0.3 <= seconds/row['seconds'] <= 3)
        for kind, path in [('reference', row['audio']), ('predicted', str(args.directory/f"{row['id']}.wav"))]:
            if kind == 'predicted' and 'failure' in item:
                continue
            segments, _ = recognizer.transcribe(path, language=language, beam_size=panel['beam_size'],
                temperature=panel['temperature'], vad_filter=panel['vad_filter'],
                condition_on_previous_text=panel['condition_on_previous_text'])
            item[kind+'_text'] = ' '.join(segment.text for segment in segments).strip()
        metric = cer if language == 'zh' else wer
        truth = normalize(row['text'], language)
        item['reference_error'] = metric(truth, normalize(item['reference_text'], language))
        item['predicted_error'] = metric(truth, normalize(item['predicted_text'], language))
        report['samples'].append(item)
        atomic_json(output, report)
        print(json.dumps({key:item[key] for key in ('id', 'predicted_error', 'reference_error', 'invalid')}), flush=True)
    scores = {}
    for language in LANGUAGES:
        items = [item for item in report['samples'] if item['language'] == language]
        truth = [normalize(item['text'], language) for item in items]
        metric = cer if language == 'zh' else wer
        scores[language] = dict(metric='CER' if language == 'zh' else 'WER', clips=len(items),
            predicted_error=metric(truth, [normalize(item['predicted_text'], language) for item in items]),
            reference_error=metric(truth, [normalize(item['reference_text'], language) for item in items]),
            invalid_fraction=sum(item['invalid'] for item in items)/len(items))
    report.update(status='finished', seconds=time.monotonic()-started, per_language=scores)
    atomic_json(output, report)
    print(json.dumps(scores, indent=2), flush=True)


if __name__ == '__main__':
    main()
