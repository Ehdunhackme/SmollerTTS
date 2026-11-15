"""Exercise the built app in Chromium with GPU disabled, then reload offline."""
import json
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

destination = Path('artifacts/browser')
destination.mkdir(parents=True, exist_ok=True)
results, errors, external = [], [], []
exported = json.loads(Path('web/public/model/export.json').read_text())
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=['--disable-gpu'])
    context = browser.new_context(viewport=dict(width=1280, height=1050), accept_downloads=True)
    page = context.new_page()
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.on('console', lambda message: print('CONSOLE', message.type, message.text, flush=True) if message.type == 'error' else None)
    page.on('requestfailed', lambda request: print('REQUEST FAILED', request.url, request.failure, flush=True))
    page.on('request', lambda request: external.append(request.url) if not request.url.startswith(('http://127.0.0.1:5173', 'blob:', 'data:')) else None)
    cold_start = time.monotonic()
    page.goto('http://127.0.0.1:5173')
    page.wait_for_load_state('networkidle')
    page.get_by_role('button', name='Generate speech', exact=True).wait_for()
    page.wait_for_function("document.documentElement.dataset.offlineReady === 'true' && navigator.serviceWorker.controller !== null")
    deployed = page.evaluate("fetch('/model/export.json').then(response => response.json())")
    assert deployed == exported, 'Browser is serving stale model metadata'
    cold_seconds = time.monotonic()-cold_start
    for offline in (False, True):
        if offline:
            context.set_offline(True)
            page.reload()
            page.wait_for_load_state('networkidle')
            page.wait_for_function("!document.querySelector('#generate').disabled || document.querySelector('#status').dataset.error === 'true'", timeout=20000)
            print('OFFLINE STATUS', page.locator('#status').inner_text(), flush=True)
            page.screenshot(path=str(destination/'offline-load.png'), full_page=True)
            page.get_by_role('button', name='Generate speech', exact=True).wait_for()
        for language in ('en', 'zh', 'es', 'fr', 'de'):
            page.locator(f'input[value="{language}"]').check()
            page.evaluate('window.__lastSynthesis = null')
            page.get_by_role('button', name='Generate speech', exact=True).click()
            page.wait_for_function("window.__lastSynthesis !== null || document.querySelector('#status').dataset.error === 'true'", timeout=60000)
            result = page.evaluate('window.__lastSynthesis')
            assert result, page.locator('#status').inner_text()
            assert result['metrics']['backend'] == 'wasm-cpu'
            assert result['metrics']['audioSeconds'] > 0.1 and result['metrics']['rms'] > 1e-5
            result['offline'] = offline
            results.append(result)
            if not offline:
                with page.expect_download() as download:
                    page.get_by_role('link', name='Save WAV').click()
                download.value.save_as(destination/f'{language}-browser.wav')
            print(json.dumps(dict(language=language, offline=offline, **result['metrics'])), flush=True)
        page.screenshot(path=str(destination/('offline.png' if offline else 'desktop-audio.png')), full_page=True)
    page.locator('#sentence').fill('12')
    page.get_by_role('button', name='Generate speech', exact=True).click()
    page.wait_for_function("document.querySelector('#status').dataset.error === 'true'")
    assert 'spell numbers' in page.locator('#status').inner_text()
    page.set_viewport_size(dict(width=375, height=812))
    page.screenshot(path=str(destination/'mobile.png'), full_page=True)
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'Mobile horizontal overflow'
    report = dict(browser=browser.version, gpu_disabled=True, offline_reload=True, model=exported,
                  cold_ready_and_cache_seconds=cold_seconds, results=results,
                  page_errors=errors, external_requests=external,
                  assets=json.loads(Path('web/dist/asset-report.json').read_text()))
    (destination/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    assert not errors and not external, (errors, external)
    context.close()
    browser.close()
