"""Compare uninterrupted and recovered A100 training with real five-language data."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=Path('artifacts/gpu-recovery-check'))
args = parser.parse_args()
root = args.output
root.mkdir(parents=True, exist_ok=False)
common = [sys.executable, '-m', 'xiaotts.train', '--features', 'data/features/debug-augmented',
          '--steps', '100', '--eval-every', '100', '--spectral-start', '1', '--workers', '2', '--device', 'cuda']
for name, extra in [('full', []), ('partial', ['--stop-after', '50']),
                    ('resumed', ['--resume', str(root/'partial/latest.pt')])]:
    with (root/f'{name}.log').open('w') as log:
        subprocess.run([*common, '--output', str(root/name), *extra], stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=300)
full = torch.load(root/'full/latest.pt', map_location='cpu', weights_only=True)
resumed = torch.load(root/'resumed/latest.pt', map_location='cpu', weights_only=True)
maximum = 0
for key in full['model']:
    a, b = full['model'][key], resumed['model'][key]
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    maximum = max(maximum, float((a-b).abs().max()))
for parameter in full['optimizer']['state']:
    for key, a in full['optimizer']['state'][parameter].items():
        torch.testing.assert_close(a, resumed['optimizer']['state'][parameter][key], atol=1e-6, rtol=1e-5)
torch.testing.assert_close(full['rng']['torch'], resumed['rng']['torch'])
for a, b in zip(full['rng']['cuda'], resumed['rng']['cuda']):
    torch.testing.assert_close(a, b)
report = dict(status='passed', device=torch.cuda.get_device_name(0), steps=100, interrupted_at=50,
              languages=5, workers=2, spectral_loss=True, maximum_weight_difference=maximum,
              model_and_optimizer_tolerance=dict(atol=1e-6, rtol=1e-5), rng_exact=True)
(root/'report.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
