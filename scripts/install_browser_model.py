"""Install a verified local asset bundle, preserving the previous demo and evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    source = args.source.resolve()
    bundle = json.loads((source/'bundle.json').read_text())
    expected = {'encoder.onnx', 'decoder.onnx', 'export.json', 'frontend.json',
                'frontend-parity.json', 'lexicon-report.json', 'en.lex', 'zh.lex', 'es.lex', 'fr.lex', 'de.lex'}
    if not expected <= set(bundle['files']) or not 0 < bundle['learned_parameters'] < 1_000_000:
        raise ValueError('Incomplete or oversized browser model')
    for name, record in bundle['files'].items():
        if Path(name).name != name or '/' in name or '\\' in name:
            raise ValueError('Bundle entries must be simple filenames')
        content = (source/name).read_bytes()
        if len(content) != record['bytes'] or hashlib.sha256(content).hexdigest() != record['sha256']:
            raise ValueError('Bundle integrity failed: '+name)
    identifier = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6]
    target = (root/'web/public/model').resolve()
    staging = (root/'web/public'/('.model-install-'+identifier)).resolve()
    backup = (root/'artifacts/browser-model-backups'/identifier).resolve()
    # Verify every directory involved before moving a tree on Windows.
    if any(not path.is_relative_to(root) or path == root for path in (target, staging, backup)):
        raise ValueError('Installation paths escaped the project workspace')
    staging.mkdir()
    for name in [*bundle['files'], 'bundle.json']:
        shutil.copy2(source/name, staging/name)
    backup.mkdir(parents=True)
    for path in (root/'artifacts/browser', root/'artifacts/browser-neural-fixture.json'):
        if path.is_dir():
            shutil.copytree(path, backup/path.name)
        elif path.is_file():
            shutil.copy2(path, backup/path.name)
    old_model = backup/'model'
    if target.exists():
        shutil.move(str(target), str(old_model))
    try:
        shutil.move(str(staging), str(target))
    except Exception:
        if old_model.exists() and not target.exists():
            shutil.move(str(old_model), str(target))
        raise
    (backup/'installation.json').write_text(json.dumps(dict(source=str(source), target=str(target),
        checkpoint_sha256=bundle['checkpoint_sha256'], training_step=bundle['training_step'],
        status='Installed; rebuild and browser validation are still required'), indent=2), encoding='utf-8')
    print('Installed bundle; previous model and validation evidence preserved at', backup)


if __name__ == '__main__':
    main()
