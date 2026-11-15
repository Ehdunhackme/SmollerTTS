"""Build a checkpoint-bound browser asset bundle with its matching pronunciations."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch
from xiaotts.checkpoints import feature_identity, sha256
from xiaotts.fetch import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--features', type=Path, default=Path('data/features/full'))
    parser.add_argument('--root', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new bundle directory')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if state.get('format_version') != 2 or state['feature_identity'] != feature_identity(args.features):
        raise ValueError('The checkpoint and pronunciation/feature cache must match exactly')
    if state['training_identity']['dsp.py'] != sha256('xiaotts/dsp.py'):
        raise ValueError('Renderer code changed; revalidate the candidate renderer before packaging')
    subprocess.run([sys.executable, '-m', 'xiaotts.export', '--checkpoint', str(args.checkpoint),
                    '--output', str(args.output)], check=True)
    subprocess.run([sys.executable, '-m', 'scripts.browser_lexicon', '--root', str(args.root),
                    '--vocabulary', str(args.features/'vocabulary.json'), '--output', str(args.output)], check=True)
    exported = json.loads((args.output/'export.json').read_text())
    frontend = json.loads((args.output/'frontend.json').read_text())
    if frontend['vocabulary_sha256'] != exported['feature_identity']['vocabulary.json']:
        raise ValueError('The frontend vocabulary does not match these weights')
    if not 0 < exported['model_parameters']+exported['renderer_parameters'] < 1_000_000:
        raise ValueError('The complete learned parameter ceiling failed')
    files = {path.name:dict(bytes=path.stat().st_size, sha256=sha256(path)) for path in args.output.iterdir() if path.is_file()}
    atomic_json(args.output/'bundle.json', dict(status='native export and pronunciation identities verified',
        checkpoint_sha256=exported['checkpoint_sha256'], training_step=exported['training_step'],
        learned_parameters=exported['model_parameters']+exported['renderer_parameters'], files=files,
        remaining='Actual browser neural/frontend/DSP parity, full synthesis, memory and offline validation'))
    print('Created matching model and pronunciation bundle:', args.output, flush=True)


if __name__ == '__main__':
    main()
