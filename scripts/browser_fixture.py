"""Generate a deterministic reference for browser filter and overlap-add parity."""
import json
from pathlib import Path
import torch
from xiaotts.dsp import Synthesizer, HOP

torch.set_num_threads(2)
torch.manual_seed(42)
frames = 27
controls = torch.randn(1, frames, 67) * 0.4
controls[:, :, 0] += 5.2
controls[:, :, 2] -= 3.5
controls[:, :, 3:] -= 2
with torch.no_grad():
    audio, pulse, noise = Synthesizer()(controls, seed=1234, return_excitation=True)
path = Path('artifacts/browser-dsp-fixture.json')
path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(dict(frames=frames, controls=controls[0].flatten().tolist(),
                               pulse=pulse[0].tolist(), noise=noise[0].tolist(), audio=audio[0].tolist())))
print(path)
