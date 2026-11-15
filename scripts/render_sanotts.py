"""Run the pinned SanoTTS browser on every frozen development prompt, on CPU.

The unmodified upstream page provides its own pronunciation, weights and renderer.
Local-only requests make the recorded source manifest authoritative. No sentences
are edited or dropped, including sentences that fail or exceed upstream limits.
"""
import argparse
import base64
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Thread
import time

import numpy as np
import soundfile as sf
from playwright.sync_api import sync_playwright


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('artifacts/comparators/sanotts'))
    parser.add_argument('--panel', type=Path, default=Path('artifacts/evaluation/dev-v1.json'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/comparators/sanotts-dev-v1'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing comparison results')
    source = json.loads((args.source/'source.json').read_text())
    for item in source['files']:
        if hashlib.sha256((args.source/item['path']).read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Comparator source changed: '+item['path'])
    panel = json.loads(args.panel.read_text(encoding='utf-8'))
    if any(row['split'] != 'dev' for row in panel['rows']):
        raise ValueError('Only the development panel is allowed')
    args.output.mkdir(parents=True)
    report = dict(status='running', system='SanoTTS', source=source,
                  panel_sha256=hashlib.sha256(args.panel.read_bytes()).hexdigest(),
                  renderer_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  gpu_disabled=True, samples=[], blocked_requests=[], page_errors=[],
                  input='Unmodified actual development text through upstream browser frontend',
                  timing='Total call includes frontend and first-use local asset loading; not an internet download estimate')
    handler = partial(QuietHandler, directory=str((args.source/'web').resolve()))
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    Thread(target=server.serve_forever, daemon=True).start()
    origin = f'http://127.0.0.1:{server.server_port}'
    started = time.monotonic()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, args=['--disable-gpu'])
            report['browser'] = browser.version
            context = browser.new_context()
            def allow_local(route):
                url = route.request.url
                if url.startswith((origin+'/', 'blob:', 'data:')):
                    route.continue_()
                else:
                    report['blocked_requests'].append(url)
                    route.abort()
            context.route('**/*', allow_local)
            page = context.new_page()
            page.on('pageerror', lambda error: report['page_errors'].append(str(error)))
            page.goto(origin+'/index.html?voices=local')
            page.wait_for_function('G2P !== null && Voice !== null && NANO_MODULES.heartnano', timeout=60000)
            for row in panel['rows']:
                item = dict(id=row['id'], language=row['language'], text=row['text'],
                            voice=source['voices'][row['language']], reference_seconds=row['seconds'])
                try:
                    audio = page.evaluate('''async ({voice, text}) => {
                        const started = performance.now();
                        const result = voice === 'heartnano' ? await synthNano(voice, text) : await synthVoice(voice, text);
                        const total = (performance.now() - started) / 1000;
                        const bytes = new Uint8Array(result.pcm.buffer, result.pcm.byteOffset, result.pcm.byteLength);
                        let binary = '';
                        for (let i=0; i<bytes.length; i+=8192)
                            binary += String.fromCharCode(...bytes.subarray(i, i+8192));
                        return {pcm: btoa(binary), rate: result.sr, render_seconds: result.wall, total_seconds: total};
                    }''', dict(voice=item['voice'], text=row['text']))
                    wave = np.frombuffer(base64.b64decode(audio.pop('pcm')), dtype='<f4')
                    if not len(wave) or not np.isfinite(wave).all():
                        raise ValueError('Empty or non-finite comparator audio')
                    path = args.output/f"{row['id']}.wav"
                    sf.write(path, wave, audio['rate'], subtype='FLOAT')
                    item.update(audio, generated_seconds=len(wave)/audio['rate'],
                                wave_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                except Exception as error:
                    item['failure'] = str(error)
                report['samples'].append(item)
                (args.output/'render.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
                print(json.dumps({key:item[key] for key in ('id', 'voice', 'generated_seconds', 'failure') if key in item}), flush=True)
            context.close()
            browser.close()
        report.update(status='finished', seconds=time.monotonic()-started)
    finally:
        server.shutdown()
        server.server_close()
        (args.output/'render.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
