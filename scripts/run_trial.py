"""Run one declared, bounded scratch experiment against a finished baseline.

This does not choose hypotheses or start an indefinite sweep. Each invocation
records one candidate, preserves its incumbent, and only nominates improvements.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from xiaotts.fetch import atomic_json
from xiaotts.benchmark import digest
from xiaotts.model import ModelConfig, XiaoTTS
from xiaotts.selection import compare

ALLOWED = {'lr', 'width', 'wave_weight', 'spectral_start'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--incumbent-run', type=Path, required=True)
    parser.add_argument('--incumbent-evaluation', type=Path, required=True)
    parser.add_argument('--panel', type=Path, default=Path('evaluation/dev-v1.json'))
    parser.add_argument('--timeout-seconds', type=int, default=7200)
    args = parser.parse_args()
    specification = json.loads(args.spec.read_text())
    name = specification['name']
    if not name or any(character not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for character in name):
        raise ValueError('Use a simple lowercase trial name')
    if not specification.get('hypothesis') or not specification.get('changes'):
        raise ValueError('A hypothesis and explicit change are required')
    changes = specification['changes']
    if set(changes) - ALLOWED or len(changes) != 1:
        raise ValueError('Initial trials change one supported setting at a time')
    incumbent = json.loads((args.incumbent_run/'run.json').read_text())
    reference = json.loads(args.incumbent_evaluation.read_text())
    if incumbent['status'] != 'finished' or reference['status'] != 'finished':
        raise ValueError('Train and evaluate the baseline before proposing trials')
    if reference['checkpoint_sha256'] != digest(args.incumbent_run/'latest.pt') or reference['panel_sha256'] != digest(args.panel):
        raise ValueError('Incumbent evaluation does not match its checkpoint and panel')
    if incumbent['initialization'] != 'random':
        raise ValueError('The baseline must be trained from random initialization')
    if incumbent['arguments'].get('train_clips_per_language', 0):
        raise ValueError('A debugging subset cannot stand in for the full-data baseline')
    config = dict(incumbent['arguments'])
    config.update(changes)
    model = XiaoTTS(ModelConfig(width=int(config['width'])))
    parameters = model.parameter_count
    del model
    output = Path('runs')/name
    if output.exists():
        raise FileExistsError('Never overwrite a previous experiment')
    output.mkdir(parents=True)
    state = dict(status='running', spec=specification, incumbent_run=str(args.incumbent_run),
                 incumbent_evaluation=str(args.incumbent_evaluation), parameters=parameters,
                 seed=incumbent['seed'], steps=incumbent['step'], started=time.time(),
                 budget='Matched optimizer steps, batch composition and seed; actual GPU time also reported',
                 final_claim='Provisional only; repeated seeds and browser checks still required')
    atomic_json(output/'trial.json', state)
    checkpoint = output/'model/latest.pt'
    command = [sys.executable, '-m', 'xiaotts.train', '--features', config['features'],
               '--output', str(output/'model'), '--steps', str(incumbent['step']),
               '--seed', str(incumbent['seed']), '--workers', str(config.get('workers', 4)),
               '--per-language', str(config.get('per_language', 1)),
               '--eval-every', str(config['eval_every']), '--wave-frames', str(config['wave_frames'])]
    for key in ALLOWED:
        command += ['--'+key.replace('_','-'), str(config[key])]
    if config.get('wave_pitch_grad'):
        command.append('--wave-pitch-grad')
    try:
        subprocess.run([sys.executable, 'scripts/session_job.py', '--name', name+'-training',
                        '--timeout-seconds', str(args.timeout_seconds), '--', *command], check=True)
        training = json.loads((output/'model/run.json').read_text())
        if training['status'] != 'finished' or training['step'] != incumbent['step']:
            raise ValueError('Candidate did not complete the matched training budget')
        if training['features'] != incumbent['features']:
            raise ValueError('Candidate features differ from the baseline')
        subprocess.run([sys.executable, '-m', 'xiaotts.export', '--checkpoint', str(checkpoint),
                        '--output', str(output/'export')], check=True)
        subprocess.run([sys.executable, '-m', 'xiaotts.benchmark', 'evaluate', '--panel', str(args.panel),
                        '--vocabulary', str(Path(config['features'])/'vocabulary.json'),
                        '--checkpoint', str(checkpoint), '--output', str(output/'evaluation')], check=True)
        candidate = json.loads((output/'evaluation/report.json').read_text())
        decision = compare(candidate, reference)
        state.update(status='finished', decision=decision, training_seconds=training['elapsed'],
                     incumbent_training_seconds=incumbent['elapsed'],
                     peak_cuda_allocated_bytes=training['peak_cuda_allocated_bytes'])
    except Exception as error:
        state.update(status='failed', error=str(error))
        raise
    finally:
        state['finished'] = time.time()
        atomic_json(output/'trial.json', state)
        with Path('runs/experiment-ledger.jsonl').open('a', encoding='utf-8') as ledger:
            ledger.write(json.dumps(state, ensure_ascii=False)+'\n')
    print(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()
