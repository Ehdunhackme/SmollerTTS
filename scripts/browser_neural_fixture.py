"""Reference tensors for native PyTorch versus actual browser WASM inference."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from xiaotts.model import XiaoTTS, ModelConfig, expand_durations

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', type=Path, default=Path('artifacts/debug-train-v2/latest.pt'))
parser.add_argument('--output', type=Path, default=Path('artifacts/browser-neural-fixture.json'))
args = parser.parse_args()
torch.set_num_threads(2)
state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
model = XiaoTTS(ModelConfig(**state['config'])).eval()
model.load_state_dict(state['model'])
fixtures = []
with torch.no_grad():
    for language, n in ((language, n) for language in range(5) for n in (7, 19)):
        inputs = dict(tokens=torch.arange(3, n+3)[None], language=torch.tensor([language]),
                      tones=(torch.arange(n)%6)[None], stress=(torch.arange(n)%4)[None])
        encoded, duration = model.encode(**inputs)
        indices, positions = expand_durations(torch.arange(n)%5+2)
        mask = torch.ones(1, len(indices))
        controls = model.decode(encoded, indices[None], positions[None], mask)
        tensors = {**inputs, 'frame_indices': indices[None], 'positions': positions[None], 'frame_mask': mask,
                   'encoded': encoded, 'log_duration': duration, 'controls': controls}
        fixtures.append({key: dict(shape=list(value.shape), values=value.flatten().tolist(),
                                  dtype='int64' if value.dtype == torch.long else 'float32') for key,value in tensors.items()})
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(dict(checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
    training_step=state['step'], parameters=model.parameter_count, fixtures=fixtures)))
