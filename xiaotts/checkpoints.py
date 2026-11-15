"""Training recovery state; never silently warm-start a new experiment."""
import hashlib
from pathlib import Path
import random
import numpy as np
import torch

RECIPE_KEYS = ('steps', 'seed', 'lr', 'width', 'spectral_start', 'wave_frames', 'wave_weight',
               'wave_pitch_grad', 'per_language', 'train_clips_per_language', 'nondeterministic')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def feature_identity(directory):
    return {name: sha256(Path(directory)/name) for name in ('index.jsonl', 'summary.json', 'vocabulary.json')}


def training_identity():
    return {name: sha256(Path(__file__).parent/name) for name in ('train.py', 'model.py', 'dsp.py', 'checkpoints.py')}


def capture_rng():
    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return dict(python=random.getstate(), numpy=(algorithm, keys.tolist(), position, has_gauss, cached_gaussian),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_rng(state):
    random.setstate(state['python'])
    algorithm, keys, position, has_gauss, cached_gaussian = state['numpy']
    np.random.set_state((algorithm, np.array(keys, dtype=np.uint32), position, has_gauss, cached_gaussian))
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA device count differs from the recovery checkpoint')
        torch.cuda.set_rng_state_all(state['cuda'])
