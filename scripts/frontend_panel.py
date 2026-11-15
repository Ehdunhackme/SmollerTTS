"""Record native pronunciation for every fixed dev prompt, without model inference."""
import argparse
import json
from pathlib import Path
from xiaotts import LANGUAGES
from xiaotts.benchmark import digest, evaluator_sources
from xiaotts.fetch import atomic_json
from xiaotts.frontend import Frontend

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--panel', type=Path, default=Path('evaluation/dev-v1.json'))
parser.add_argument('--output', type=Path, default=Path('artifacts/frontend-dev-v1.json'))
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError('Preserve previous frontend evidence')
panel = json.loads(args.panel.read_text())
if panel['sources'] != evaluator_sources() or any(row['split'] != 'dev' for row in panel['rows']):
    raise ValueError('Changed protocol or non-development input')
vocabulary = Path('data/features/full/vocabulary.json')
if digest(vocabulary) != panel['vocabulary_sha256']:
    raise ValueError('Different pronunciation vocabulary')
frontends = {lang:Frontend('data', lang, json.loads(vocabulary.read_text())) for lang in LANGUAGES}
report = dict(panel_sha256=digest(args.panel), sources=evaluator_sources(), samples=[])
for row in panel['rows']:
    item = dict(id=row['id'], language=row['language'])
    try:
        inputs, words = frontends[row['language']](row['text'])
        item['input'] = {key:value[0].tolist() for key,value in inputs.items() if key != 'language'}
        item['words'] = [word for word, _ in words]
    except (ValueError, KeyError) as error:
        item['failure'] = str(error)
    report['samples'].append(item)
atomic_json(args.output, report)
print(json.dumps(dict(clips=len(report['samples']), failures=sum('failure' in item for item in report['samples']))))
