"""Render every fixed dev prompt through the actual offline browser CPU app.

The test captures the worker's float output before WAV quantization, without
changing production code. Product input limits and frontend failures are retained.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, default=Path('artifacts/evaluation/dev-v1.json'))
    parser.add_argument('--frontend', type=Path, default=Path('artifacts/frontend-dev-v1.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous browser evaluations')
    panel = json.loads(args.panel.read_text(encoding='utf-8'))
    native = json.loads(args.frontend.read_text(encoding='utf-8'))
    panel_hash = hashlib.sha256(args.panel.read_bytes()).hexdigest()
    if any(row['split'] != 'dev' for row in panel['rows']) or native['panel_sha256'] != panel_hash:
        raise ValueError('Different frontend panel or non-development input')
    native = {item['id']:item for item in native['samples']}
    exported = json.loads(Path('web/public/model/export.json').read_text())
    report = dict(status='running', system='XiaoTTS browser CPU', panel_sha256=panel_hash,
        source=dict(model=exported, app_assets=json.loads(Path('web/dist/asset-report.json').read_text())),
        gpu_disabled=True, offline_reload=True, samples=[],
        input='Unmodified actual development text through the browser frontend, neural graphs and waveform synthesizer',
        page_errors=[], external_requests=[])
    args.output.mkdir(parents=True)
    started = time.monotonic()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=['--disable-gpu'])
        report['browser'] = browser.version
        context = browser.new_context()
        context.add_init_script('''(() => {
          const OriginalWorker = window.Worker;
          window.Worker = class extends OriginalWorker {
            constructor(...args) { super(...args); this.addEventListener('message', event => {
              if (event.data.type === 'audio') window.__evaluationAudio = event.data.audio;
            }); }
          };
        })();''')
        page = context.new_page()
        page.on('pageerror', lambda error: report['page_errors'].append(str(error)))
        page.on('request', lambda request: report['external_requests'].append(request.url)
                if not request.url.startswith(('http://127.0.0.1:5173/', 'blob:', 'data:')) else None)
        page.goto('http://127.0.0.1:5173/')
        page.wait_for_function("document.documentElement.dataset.offlineReady === 'true' && navigator.serviceWorker.controller !== null")
        context.set_offline(True)
        page.reload()
        page.wait_for_function("!document.querySelector('#generate').disabled")
        if page.evaluate("fetch('/model/export.json').then(r => r.json())") != exported:
            raise ValueError('The browser served stale model metadata')
        for row in panel['rows']:
            item = dict(id=row['id'], language=row['language'], text=row['text'], reference_seconds=row['seconds'])
            try:
                if len(row['text']) > 240:
                    raise ValueError('Exceeds the product input limit of 240 characters')
                page.locator(f'input[value="{row["language"]}"]').check()
                page.locator('#sentence').fill(row['text'])
                page.evaluate('window.__lastSynthesis = null; window.__evaluationAudio = null')
                page.get_by_role('button', name='Generate speech', exact=True).click()
                page.wait_for_function("window.__lastSynthesis !== null || document.querySelector('#status').dataset.error === 'true'", timeout=60000)
                result = page.evaluate('window.__lastSynthesis')
                if result is None:
                    raise ValueError(page.locator('#status').inner_text())
                encoded = page.evaluate('''() => {
                    const wave = window.__evaluationAudio;
                    const bytes = new Uint8Array(wave.buffer, wave.byteOffset, wave.byteLength);
                    let binary = '';
                    for (let i=0; i<bytes.length; i+=8192)
                        binary += String.fromCharCode(...bytes.subarray(i, i+8192));
                    return btoa(binary);
                }''')
                wave = np.frombuffer(base64.b64decode(encoded), dtype='<f4')
                if not len(wave) or not np.isfinite(wave).all():
                    raise ValueError('Empty or non-finite browser audio')
                path = args.output/(row['id']+'.wav')
                sf.write(path, wave, 22050, subtype='FLOAT')
                item.update(rate=22050, generated_seconds=len(wave)/22050, browser_metrics=result['metrics'],
                            browser_input=result['input'], wave_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                reference = native[row['id']]
                item['native_frontend_failure'] = reference.get('failure')
                item['frontend_exact'] = {key:result['input'][key] == reference.get('input', {}).get(key)
                                          for key in ('tokens', 'tones', 'stress')}
                item['native_words'] = reference.get('words')
            except Exception as error:
                item['failure'] = str(error)
                item['native_frontend_failure'] = native[row['id']].get('failure')
            report['samples'].append(item)
            (args.output/'render.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
            print(json.dumps({key:item[key] for key in ('id', 'frontend_exact', 'failure') if key in item}), flush=True)
        browser.close()
    report.update(status='failed' if report['page_errors'] or report['external_requests'] else 'finished',
                  seconds=time.monotonic()-started)
    (args.output/'render.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    if report['page_errors'] or report['external_requests']:
        raise ValueError('Unexpected browser errors or external requests')


if __name__ == '__main__':
    main()
