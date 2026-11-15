from pathlib import Path
from playwright.sync_api import sync_playwright

destination = Path('artifacts/browser')
destination.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport=dict(width=1280, height=1050))
    page.on('console', lambda message: print('CONSOLE', message.type, message.text))
    page.on('pageerror', lambda error: print('PAGE ERROR', error))
    page.goto('http://127.0.0.1:5173')
    page.wait_for_load_state('networkidle')
    page.screenshot(path=str(destination/'desktop.png'), full_page=True)
    print(page.locator('body').inner_text())
    print('BUTTONS', page.get_by_role('button').all_text_contents())
    print('STATUS', page.locator('#status').inner_text())
    browser.close()
