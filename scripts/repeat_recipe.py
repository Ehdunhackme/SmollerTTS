"""Repeat a finished full-data recipe from scratch at a predeclared new seed."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from xiaotts.benchmark import digest
from xiaotts.checkpoints import RECIPE_KEYS, feature_identity, training_identity
from xiaotts.fetch import atomic_json


def prepare(reference, seed, name):
    if seed not in (41, 73):
        raise ValueError('This confirmation cycle predeclares seeds 41 and 73')
    if not name or any(character not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for character in name):
        raise ValueError('Use a simple lowercase run name')
    record = json.loads((reference/'run.json').read_text(encoding='utf-8'))
    config = dict(record['arguments'])
    if record['status'] != 'finished' or record['step'] != 50000 or record['seed'] != 17:
        raise ValueError('Reference must be a finished 50000-step seed-17 recipe')
    if record['initialization'] != 'random' or config.get('train_clips_per_language', 0):
        raise ValueError('Require a full-data recipe trained from random initialization')
    if record['training_identity'] != training_identity() or record['feature_identity'] != feature_identity(Path(config['features'])):
        raise ValueError('Training source or feature cache changed')
    if not 0 < record['parameters'] + record['renderer_parameters'] < 1_000_000:
        raise ValueError('Complete inference parameter budget failed')
    output = Path('runs')/name
    if output.exists() or Path('runs', name+'-training', 'process.json').exists():
        raise FileExistsError('Preserve existing experiments')
    config['seed'] = seed
    command = [sys.executable, '-m', 'xiaotts.train', '--features', config['features'],
               '--output', str(output/'model'), '--workers', str(config.get('workers', 4)),
               '--eval-every', str(config['eval_every']), '--keep-every', str(config.get('keep_every', 10000))]
    for key in RECIPE_KEYS:
        value = config[key]
        if isinstance(value, bool):
            if value:
                command.append('--'+key.replace('_', '-'))
        else:
            command += ['--'+key.replace('_', '-'), str(value)]
    return record, output, command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--panel', type=Path, default=Path('evaluation/dev-v1.json'))
    parser.add_argument('--timeout-seconds', type=int, default=7200)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    reference, output, command = prepare(args.reference_run, args.seed, args.name)
    if args.dry_run:
        print(json.dumps(dict(status='validated-plan-only', command=command), indent=2))
        return
    output.mkdir(parents=True)
    state = dict(status='running', kind='independent-seed-confirmation', seed=args.seed,
        reference_run=str(args.reference_run), reference_run_sha256=digest(args.reference_run/'run.json'),
        reference_checkpoint_sha256=digest(args.reference_run/'latest.pt'),
        panel_sha256=digest(args.panel), started=time.time(), command=command,
        budget='Same recipe and 50000 steps; new seed only, random initialization')
    atomic_json(output/'confirmation.json', state)
    try:
        subprocess.run([sys.executable, 'scripts/session_job.py', '--name', args.name+'-training',
            '--timeout-seconds', str(args.timeout_seconds), '--', *command], check=True)
        trained = json.loads((output/'model/run.json').read_text(encoding='utf-8'))
        if trained['status'] != 'finished' or trained['step'] != 50000 or trained['seed'] != args.seed:
            raise ValueError('Confirmation training did not finish the declared budget')
        if trained['training_identity'] != reference['training_identity'] or trained['feature_identity'] != reference['feature_identity']:
            raise ValueError('Confirmation source or data identities changed')
        for key in RECIPE_KEYS:
            expected = args.seed if key == 'seed' else reference['arguments'][key]
            if trained['arguments'][key] != expected:
                raise ValueError('Unexpected recipe change: '+key)
        checkpoint = output/'model/latest.pt'
        subprocess.run([sys.executable, '-m', 'xiaotts.export', '--checkpoint', str(checkpoint),
                        '--output', str(output/'export')], check=True)
        subprocess.run([sys.executable, '-m', 'xiaotts.benchmark', 'evaluate', '--panel', str(args.panel),
            '--vocabulary', str(Path(trained['arguments']['features'])/'vocabulary.json'),
            '--checkpoint', str(checkpoint), '--output', str(output/'evaluation')], check=True)
        evaluation = json.loads((output/'evaluation/report.json').read_text(encoding='utf-8'))
        if evaluation['status'] != 'finished' or evaluation['checkpoint_sha256'] != digest(checkpoint):
            raise ValueError('Incomplete or mismatched confirmation evaluation')
        state.update(status='finished', training_seconds=trained['elapsed'],
            peak_cuda_allocated_bytes=trained['peak_cuda_allocated_bytes'],
            checkpoint_sha256=digest(checkpoint), per_language=evaluation['per_language'])
    except Exception as error:
        state.update(status='failed', error=str(error))
        raise
    finally:
        state['finished'] = time.time()
        atomic_json(output/'confirmation.json', state)
        with Path('runs/experiment-ledger.jsonl').open('a', encoding='utf-8') as ledger:
            ledger.write(json.dumps(state, ensure_ascii=False)+'\n')
    print(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()
