"""Fetch a pinned, unmodified SanoTTS browser comparator into ignored artifacts.

Only the five selected voices and browser dependencies are downloaded. These are
external baselines, never training inputs or part of the XiaoTTS parameter budget.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = '8ba78d4f9bbd164cc0d3ddc36beba2c5a0c5530e'
VOICES = {'en': 'heartnano', 'zh': 'chinese-xiaoya', 'es': 'spanish-tiny',
          'fr': 'french', 'de': 'german-tiny'}


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'XiaoTTS-comparison'})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/comparators/sanotts'))
    args = parser.parse_args()
    tree = json.loads(fetch(f'https://api.github.com/repos/Ampixa/sanoTTS/git/trees/{REVISION}?recursive=1'))
    if tree.get('truncated') or tree['sha'] != REVISION:
        raise ValueError('Incomplete or unexpected upstream tree')
    selected = []
    for item in tree['tree']:
        path = item['path']
        if item['type'] != 'blob':
            continue
        if (path in ('LICENSE', 'LICENSE.MIT', 'README.md', 'web/index.html', 'web/trellis_frontend.js')
                or path.startswith('web/snt_')
                or path.startswith('web/g2p-lazy/zh/')
                or any(path.startswith(f'web/voices/{voice}/') for voice in VOICES.values())):
            selected.append(item)

    def download(item):
        path = item['path']
        target = args.output/path
        target.parent.mkdir(parents=True, exist_ok=True)
        def git_hash(content):
            return hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest()
        content = target.read_bytes() if target.exists() else b''
        if git_hash(content) != item['sha']:
            content = fetch(f'https://raw.githubusercontent.com/Ampixa/sanoTTS/{REVISION}/{path}')
            if len(content) != item['size'] or git_hash(content) != item['sha']:
                raise ValueError(f'Upstream blob verification failed: {path}')
            temporary = target.with_suffix(target.suffix+'.partial')
            temporary.write_bytes(content)
            temporary.replace(target)
        return dict(path=path, bytes=len(content), sha256=hashlib.sha256(content).hexdigest(), git_blob=item['sha'])

    with ThreadPoolExecutor(max_workers=6) as pool:
        files = list(pool.map(download, selected))
    report = dict(repository='https://github.com/Ampixa/sanoTTS', revision=REVISION,
                  voices=VOICES, files=files, raw_bytes=sum(item['bytes'] for item in files),
                  note='Five separate released voice models; unchanged upstream browser code. No XiaoTTS training use.')
    (args.output/'source.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(revision=REVISION, files=len(files), raw_bytes=report['raw_bytes'])))


if __name__ == '__main__':
    main()
