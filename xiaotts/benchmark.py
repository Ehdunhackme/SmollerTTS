"""Frozen, text-input development evaluation. Never reads final-test manifests.

Recognition errors are intelligibility proxies, not naturalness measurements.
The per-language gate in selection.py avoids mixing CER and WER into one score.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch
from xiaotts import LANGUAGES
from xiaotts.alignment import dictionary_path
from xiaotts.dsp import Synthesizer, SAMPLE_RATE
from xiaotts.fetch import atomic_json
from xiaotts.frontend import Frontend
from xiaotts.model import XiaoTTS, ModelConfig, expand_durations


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluator_sources():
    root = Path(__file__).parent
    return {name: digest(root/name) for name in ('benchmark.py', 'evaluate_renderer.py', 'frontend.py',
                                               'phones.py', 'alignment.py', 'selection.py')}


def recognizer_snapshot():
    cache = Path.home()/'.cache/huggingface/hub/models--Systran--faster-whisper-small'
    revision = (cache/'refs/main').read_text().strip()
    if len(revision) != 40 or any(char not in '0123456789abcdef' for char in revision):
        raise ValueError('Invalid cached recognizer revision')
    directory = cache/'snapshots'/revision
    files = {name: digest(directory/name) for name in ('model.bin', 'config.json', 'tokenizer.json', 'vocabulary.txt')}
    return dict(repo='Systran/faster-whisper-small', revision=revision, directory=str(directory), sha256=files)


def freeze(root, vocabulary, output, count):
    if output.exists():
        raise FileExistsError('An evaluation panel cannot be overwritten')
    rows = [json.loads(line) for line in (root/'manifests/dev.jsonl').read_text().splitlines()]
    selected = []
    for language in LANGUAGES:
        candidates = [row for row in rows if row['language'] == language and row['split'] == 'dev']
        candidates.sort(key=lambda row: hashlib.sha256(('xiaotts-dev-v1:'+row['id']).encode()).hexdigest())
        if len(candidates) < count:
            raise ValueError('Not enough development rows')
        selected += candidates[:count]
    atomic_json(output, dict(version=1, purpose='fixed development text-input intelligibility panel',
        recognizer='small', beam_size=5, temperature=0, condition_on_previous_text=False,
        vad_filter=False, normalization='NFKC/lowercase/alphanumeric; Mandarin OpenCC t2s',
        count_per_language=count, dataset=json.loads((root/'manifests/dataset.json').read_text())['sha256'],
        vocabulary_sha256=digest(vocabulary), sources=evaluator_sources(),
        lexicon_sha256={language:digest(dictionary_path(root, language)) for language in LANGUAGES},
        recognizer_snapshot=recognizer_snapshot(),
        selection='SHA256(xiaotts-dev-v1: + id), first N in each language; no quality filtering',
        rows=selected))


def evaluate(root, vocabulary, panel_path, checkpoint, output):
    from faster_whisper import WhisperModel
    from jiwer import wer, cer
    from xiaotts.evaluate_renderer import normalize
    if output.exists():
        raise FileExistsError('Choose a new evaluation output directory')
    panel = json.loads(panel_path.read_text())
    if panel['sources'] != evaluator_sources() or panel['vocabulary_sha256'] != digest(vocabulary):
        raise ValueError('Evaluator or vocabulary changed; do not compare incompatible panels')
    if panel['lexicon_sha256'] != {language:digest(dictionary_path(root, language)) for language in LANGUAGES}:
        raise ValueError('Pronunciation dictionaries changed')
    recognizer_path = Path(panel['recognizer_snapshot']['directory'])
    if panel['recognizer_snapshot']['sha256'] != {name:digest(recognizer_path/name) for name in panel['recognizer_snapshot']['sha256']}:
        raise ValueError('Recognizer weights changed')
    if any(row['split'] != 'dev' for row in panel['rows']):
        raise ValueError('This evaluator accepts development rows only')
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = XiaoTTS(ModelConfig(**state['config'])).eval().cuda()
    model.load_state_dict(state['model'])
    renderer = Synthesizer().cuda()
    vocab = json.loads(vocabulary.read_text())
    frontends = {language: Frontend(root, language, vocab) for language in LANGUAGES}
    report = dict(status='running', checkpoint_sha256=digest(checkpoint), panel_sha256=digest(panel_path),
                  parameters=model.parameter_count, initialization='see training run record',
                  input='actual development text; no aligned phonemes or ground-truth durations',
                  metric_limit='ASR intelligibility proxy, no naturalness claim',
                  sources=evaluator_sources(), samples=[])
    started = time.monotonic()
    for row in panel['rows']:
        item = dict(id=row['id'], language=row['language'], text=row['text'], reference_seconds=row['seconds'])
        try:
            inputs, _ = frontends[row['language']](row['text'])
            inputs = {key:value.cuda() for key,value in inputs.items()}
            with torch.no_grad():
                encoded, duration = model.encode(**inputs)
                duration = duration[0].clamp(0, math.log(101)).expm1().round().long().clamp(1, 100)
                indices, positions = expand_durations(duration)
                if len(indices) > 5000 or len(indices) < 5:
                    raise ValueError('Implausible predicted length')
                controls = model.decode(encoded, indices[None], positions[None], torch.ones(1, len(indices), device='cuda'))
                wave = renderer(controls, seed=1234)[0].cpu().numpy()
            if not np.isfinite(wave).all():
                raise ValueError('Non-finite audio')
            seconds = len(wave)/SAMPLE_RATE
            rms = float(np.sqrt(np.mean(wave**2)))
            clipped = float((np.abs(wave) >= 1).mean())
            item.update(generated_seconds=seconds, duration_ratio=seconds/row['seconds'],
                        rms=rms, peak=float(np.abs(wave).max()), clipped_fraction=clipped,
                        invalid=rms < 1e-4 or clipped > 0.01 or not 0.3 <= seconds/row['seconds'] <= 3)
            sf.write(output/f"{row['id']}.wav", wave, SAMPLE_RATE, subtype='FLOAT')
        except (ValueError, KeyError) as error:
            item.update(invalid=True, failure=str(error))
        report['samples'].append(item)
    del model, renderer
    torch.cuda.empty_cache()
    recognizer = WhisperModel(str(recognizer_path), device='cuda', compute_type='float16', cpu_threads=4)
    for row, item in zip(panel['rows'], report['samples']):
        language = row['language']
        for kind, path in [('reference', row['audio']), ('predicted', str(output/f"{row['id']}.wav"))]:
            if kind == 'predicted' and 'failure' in item:
                item['predicted_text'] = ''
                continue
            segments, _ = recognizer.transcribe(path, language=language, beam_size=panel['beam_size'],
                temperature=panel['temperature'], vad_filter=panel['vad_filter'],
                condition_on_previous_text=panel['condition_on_previous_text'])
            item[kind+'_text'] = ' '.join(segment.text for segment in segments).strip()
        metric = cer if language == 'zh' else wer
        truth = normalize(row['text'], language)
        item['reference_error'] = metric(truth, normalize(item['reference_text'], language))
        item['predicted_error'] = metric(truth, normalize(item['predicted_text'], language))
        atomic_json(output/'report.json', report)
        print(json.dumps({key:item[key] for key in ('id','language','predicted_error','reference_error','invalid')}, ensure_ascii=False), flush=True)
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
    atomic_json(output/'report.json', report)
    print(json.dumps(scores, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'evaluate'))
    parser.add_argument('--root', type=Path, default=Path('data'))
    parser.add_argument('--vocabulary', type=Path, default=Path('data/features/full/vocabulary.json'))
    parser.add_argument('--panel', type=Path, default=Path('evaluation/dev-v1.json'))
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze(args.root, args.vocabulary, args.panel, args.count)
    else:
        if not args.checkpoint or not args.output:
            parser.error('Evaluation needs --checkpoint and --output')
        evaluate(args.root, args.vocabulary, args.panel, args.checkpoint, args.output)


if __name__ == '__main__':
    main()
