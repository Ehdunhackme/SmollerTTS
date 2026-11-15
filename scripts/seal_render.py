"""Add a separate integrity manifest to completed comparator/diagnostic audio."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    path = args.directory/'render.json'
    rendered = json.loads(path.read_text(encoding='utf-8'))
    if rendered['status'] != 'finished':
        raise ValueError('Only completed rendering can be sealed')
    output = args.directory/'audio-integrity.json'
    if output.exists():
        raise FileExistsError('Preserve existing integrity records')
    hashes = {item['id']:digest(args.directory/(item['id']+'.wav'))
              for item in rendered['samples'] if 'failure' not in item}
    output.write_text(json.dumps(dict(render_sha256=digest(path), audio_sha256=hashes,
        timing='Integrity inventory captured after rendering; original report unchanged'), indent=2), encoding='utf-8')
    print(f'Sealed {len(hashes)} audio files')


if __name__ == '__main__':
    main()
