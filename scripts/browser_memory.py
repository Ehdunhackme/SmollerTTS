"""Sample private committed memory of this isolated Chromium process tree.

Windows private bytes exclude shared mappings; they are not a portable RAM metric.
"""
import json
from pathlib import Path
import threading
import time
import psutil
from playwright.sync_api import sync_playwright

samples, stop = [], threading.Event()
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=['--disable-gpu'])
    session = browser.new_browser_cdp_session()
    processes = session.send('SystemInfo.getProcessInfo')['processInfo']
    pid = next(item['id'] for item in processes if item['type'] == 'browser')
    root = psutil.Process(pid)
    page = browser.new_page()
    def measure():
        while not stop.is_set():
            private_bytes = working_set = 0
            for process in [root, *root.children(recursive=True)]:
                try:
                    info = process.memory_info()
                    private_bytes += info.private
                    working_set += info.rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            samples.append(dict(time=time.monotonic(), private_bytes=private_bytes, summed_working_set=working_set))
            stop.wait(0.05)
    baseline = sum(process.memory_info().private for process in [root, *root.children(recursive=True)])
    watcher = threading.Thread(target=measure, daemon=True)
    watcher.start()
    page.goto('http://127.0.0.1:5173')
    page.wait_for_load_state('networkidle')
    page.get_by_role('button', name='Generate speech', exact=True).wait_for()
    for language in ('en','zh','es','fr','de'):
        page.locator(f'input[value="{language}"]').check()
        page.evaluate('window.__lastSynthesis = null')
        page.get_by_role('button', name='Generate speech', exact=True).click()
        page.wait_for_function('window.__lastSynthesis !== null')
    stop.set()
    watcher.join()
    result = dict(browser=browser.version, samples=len(samples), sampling_ms=50,
                  model=json.loads(Path('web/public/model/export.json').read_text()),
                  blank_browser_private_bytes=baseline,
                  peak_browser_tree_private_bytes=max(item['private_bytes'] for item in samples),
                  peak_summed_working_set=max(item['summed_working_set'] for item in samples),
                  caveat='Entire isolated Chromium tree, including startup/cache; private committed bytes are not resident RAM. Summed working sets can double-count shared pages.')
    Path('artifacts/browser/memory.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    browser.close()
