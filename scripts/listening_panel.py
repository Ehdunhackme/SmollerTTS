"""Build or locally serve a blinded development listening panel.

System identities remain outside the served directory. Ratings are exploratory
human evidence, not an independent multi-listener MOS study or final-test scores.
"""
import argparse
from datetime import datetime, timezone
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import random
import shutil


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(args):
    if not 1 <= args.count <= 20:
        raise ValueError('Use one to twenty fixed prompts per language')
    if args.output.exists():
        raise FileExistsError('Preserve existing listening panels and responses')
    panel = json.loads(args.panel.read_text(encoding='utf-8'))
    if any(row['split'] != 'dev' for row in panel['rows']):
        raise ValueError('Listening selection uses development material only')
    specification = json.loads(args.spec.read_text(encoding='utf-8'))
    systems = specification['systems']
    if not 2 <= len(systems) <= 4:
        raise ValueError('Compare two to four systems')
    reports = {}
    for name, system in systems.items():
        report = json.loads(Path(system['report']).read_text(encoding='utf-8'))
        if report['status'] != 'finished' or report['panel_sha256'] != digest(args.panel):
            raise ValueError('Every system must have a completed evaluation on this same panel')
        reports[name] = {item['id']:item for item in report['samples']}
    public = args.output/'public'
    (public/'audio').mkdir(parents=True)
    rng = random.Random(specification.get('seed', 20261115))
    manifest = dict(id=hashlib.sha256(args.spec.read_bytes()+args.panel.read_bytes()+str(args.count).encode()).hexdigest(),
                    purpose='Exploratory blinded development listening', items=[])
    key = dict(panel_sha256=digest(args.panel), specification=specification,
               report_sha256={name:digest(system['report']) for name,system in systems.items()}, items={})
    for language in ('en','zh','es','fr','de'):
        rows = [row for row in panel['rows'] if row['language'] == language][:args.count]
        if len(rows) != args.count:
            raise ValueError('Insufficient rows for a language')
        for row in rows:
            item = dict(id=row['id'], language=language, text=row['text'], voices=[])
            order = list(systems)
            rng.shuffle(order)
            for index, name in enumerate(order):
                identifier = hashlib.sha256((manifest['id']+row['id']+name).encode()).hexdigest()[:24]
                sample = reports[name][row['id']]
                audio = None
                if 'failure' not in sample:
                    source = Path(systems[name]['directory'])/(row['id']+'.wav')
                    audio = 'audio/'+identifier+'.wav'
                    shutil.copy2(source, public/audio)
                    key['items'][identifier] = dict(system=name, sample_id=row['id'], source=str(source), sha256=digest(source))
                else:
                    key['items'][identifier] = dict(system=name, sample_id=row['id'], failure=sample['failure'])
                item['voices'].append(dict(id=identifier, label=chr(65+index), audio=audio))
            manifest['items'].append(item)
    (public/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    (args.output/'private-key.json').write_text(json.dumps(key, indent=2, ensure_ascii=False), encoding='utf-8')
    shutil.copy2(Path(__file__).resolve().parent.parent/'listening/index.html', public/'index.html')
    print('Built blinded listening panel:', args.output)


def serve(args):
    directory = args.directory.resolve()
    manifest = json.loads((directory/'public/manifest.json').read_text(encoding='utf-8'))
    valid_ids = {voice['id'] for item in manifest['items'] for voice in item['voices'] if voice['audio']}
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/api/ratings':
                path = directory/'ratings.json'
                body = path.read_bytes() if path.exists() else b'{}'
                self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers(); self.wfile.write(body)
            else:
                super().do_GET()

        def do_POST(self):
            if self.path != '/api/ratings':
                self.send_error(404); return
            origin = self.headers.get('Origin')
            if origin and origin != f'http://127.0.0.1:{self.server.server_port}':
                self.send_error(403); return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 131072:
                    raise ValueError('Invalid response size')
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict) or not isinstance(data.get('ratings'), dict):
                    raise ValueError('Expected a ratings object')
                if data.get('manifest_id') != manifest['id'] or set(data.get('ratings', {})) - valid_ids:
                    raise ValueError('Response does not match this listening panel')
                if data.get('device') not in ('', 'Headphones', 'Laptop speakers', 'External speakers', 'Other'):
                    raise ValueError('Unknown playback device')
                for rating in data['ratings'].values():
                    if not isinstance(rating, dict):
                        raise ValueError('Each rating must be an object')
                    if set(rating) - {'clarity','naturalness','pronunciation','notes'}:
                        raise ValueError('Unknown rating fields')
                    for field in ('clarity','naturalness','pronunciation'):
                        if rating.get(field) is not None and (type(rating[field]) is not int or not 1 <= rating[field] <= 5):
                            raise ValueError('Ratings must be integers from one to five')
                    if not isinstance(rating.get('notes',''), str) or len(rating.get('notes','')) > 2000:
                        raise ValueError('Invalid notes')
                data['saved_at'] = datetime.now(timezone.utc).isoformat()
                temporary = directory/'ratings.tmp'
                temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
                temporary.replace(directory/'ratings.json')
                self.send_response(200); self.end_headers(); self.wfile.write(b'{}')
            except (ValueError, TypeError, KeyError) as error:
                self.send_error(400, str(error))
    server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(Handler, directory=str(directory/'public')))
    print(f'Listening panel: http://127.0.0.1:{server.server_port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    builder = commands.add_parser('build')
    builder.add_argument('--panel', type=Path, default=Path('evaluation/dev-v1.json'))
    builder.add_argument('--spec', type=Path, required=True)
    builder.add_argument('--output', type=Path, required=True)
    builder.add_argument('--count', type=int, default=2)
    server = commands.add_parser('serve')
    server.add_argument('directory', type=Path)
    server.add_argument('--port', type=int, default=8766)
    arguments = parser.parse_args()
    build(arguments) if arguments.action == 'build' else serve(arguments)
