"""Exercise listening, saving and validation on an isolated disposable panel copy.

The generated ratings are test inputs and never enter a human listening record.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('panel', type=Path)
    args = parser.parse_args()
    destination = Path('artifacts/listening-ui-check')
    destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.panel/'public/manifest.json').read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory(prefix='listening-ui-', dir='artifacts') as temporary:
        root = Path(temporary)
        shutil.copytree(args.panel/'public', root/'public')
        (root/'private-key.json').write_text('{"test_only":true}')
        process = subprocess.Popen([sys.executable, '-X', 'utf8', 'scripts/listening_panel.py',
            'serve', str(root), '--port', '0'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding='utf-8')
        try:
            line = process.stdout.readline().strip()
            assert line.startswith('Listening panel: http://127.0.0.1:'), line
            url = line.removeprefix('Listening panel: ')
            errors = []
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page(viewport=dict(width=1200, height=1000))
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto(url)
                page.get_by_role('button', name='Save ratings', exact=True).wait_for()
                assert page.request.get(url+'/private-key.json').status == 404
                assert page.request.get(url+'/../private-key.json').status == 404
                for label in ['English', '中文', 'Español', 'Français', 'Deutsch']:
                    page.get_by_role('button', name=label, exact=True).click()
                    assert page.locator('audio').count() == 4
                    page.locator('audio').first.evaluate('(audio) => audio.load()')
                    page.wait_for_function("document.querySelector('audio').readyState >= 2")
                    assert page.locator('audio').first.evaluate('(audio) => audio.duration') > 0
                page.get_by_role('button', name='English', exact=True).click()
                page.locator('#device').select_option('Headphones')
                card = page.locator('.voice').first
                card.get_by_label('Clarity', exact=True).select_option('3')
                card.get_by_label('Naturalness', exact=True).select_option('2')
                card.get_by_label('Pronunciation', exact=True).select_option('4')
                card.get_by_label('Notes / words or tones that sounded wrong', exact=True).fill('AUTOMATED UI TEST ONLY')
                page.get_by_role('button', name='Save ratings', exact=True).click()
                page.wait_for_function("document.querySelector('#status').textContent.startsWith('Saved')")
                page.reload()
                page.get_by_role('button', name='Save ratings', exact=True).wait_for()
                assert page.locator('.voice').first.get_by_label('Clarity', exact=True).input_value() == '3'
                assert page.locator('#device').input_value() == 'Headphones'
                stored = json.loads((root/'ratings.json').read_text())
                voice_id = manifest['items'][0]['voices'][0]['id']
                assert stored['ratings'][voice_id]['notes'] == 'AUTOMATED UI TEST ONLY'
                for malformed in [6, True, '4']:
                    response = page.request.post(url+'/api/ratings', data=dict(manifest_id=manifest['id'],
                        device='Headphones', ratings={voice_id:dict(clarity=malformed)}))
                    assert response.status == 400
                response = page.request.post(url+'/api/ratings', data=dict(manifest_id='wrong-panel', device='', ratings={}))
                assert response.status == 400
                assert json.loads((root/'ratings.json').read_text()) == stored
                page.screenshot(path=str(destination/'desktop.png'), full_page=True)
                page.set_viewport_size(dict(width=375, height=812))
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                page.screenshot(path=str(destination/'mobile.png'), full_page=True)
                assert not errors, errors
                report = dict(status='passed', browser=browser.version,
                    panel_sha256=manifest['id'], languages=5, page_errors=errors,
                    checks=['Audio loads in all five languages', 'Anonymous public manifest',
                        'Private mapping inaccessible', 'Save and reload persist ratings',
                        'Invalid ratings cannot overwrite saved data', '375px layout without overflow'],
                    human_ratings_created=False)
                (destination/'report.json').write_text(json.dumps(report, indent=2))
                print(json.dumps(report, indent=2))
                browser.close()
        finally:
            process.terminate()
            process.wait(timeout=10)


if __name__ == '__main__':
    main()
