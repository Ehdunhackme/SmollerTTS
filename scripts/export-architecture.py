"""Export the demo's current architecture to a self-contained SVG and PNG.

Start the built demo, then run with Python and Playwright installed.
"""
import argparse
import base64
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5175/projects/smollertts/")
    args = parser.parse_args()
    output = ROOT / "docs/assets"
    output.mkdir(parents=True, exist_ok=True)
    font = base64.b64encode((ROOT / "web/fonts/geist-latin-wght-normal.woff2").read_bytes()).decode()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1200, "height": 1100}, device_scale_factor=2)
        page.goto(args.url)
        page.locator("#architecture-diagram svg").wait_for()
        page.evaluate("document.fonts.ready")
        page.add_style_tag(content="* { transition: none !important; animation: none !important; }")
        svg = page.evaluate("""font => {
            const source = document.querySelector('#architecture-diagram svg');
            source.querySelectorAll('[aria-pressed]').forEach(el => el.setAttribute('aria-pressed', 'false'));
            const clone = source.cloneNode(true);
            const originals = [source, ...source.querySelectorAll('*')];
            const copies = [clone, ...clone.querySelectorAll('*')];
            const properties = ['fill', 'fill-opacity', 'stroke', 'stroke-width', 'stroke-linecap',
                'stroke-linejoin', 'stroke-dasharray', 'opacity', 'font-family', 'font-size',
                'font-weight', 'letter-spacing', 'text-anchor', 'dominant-baseline'];
            copies.forEach((el, i) => {
                const css = getComputedStyle(originals[i]);
                for (const prop of properties) el.style.setProperty(prop, css.getPropertyValue(prop));
                for (const attr of ['class', 'tabindex', 'aria-pressed', 'aria-describedby', 'data-stage']) el.removeAttribute(attr);
                if (el.getAttribute('role') === 'button') el.removeAttribute('role');
            });
            clone.setAttribute('width', '1000'); clone.setAttribute('height', '890');
            clone.setAttribute('role', 'img');
            clone.querySelector('desc').textContent = 'SmollerTTS inference architecture, read bottom to top: phoneme encoder, duration-regulated acoustic features, and pretrained HiFi-GAN V2 waveform synthesis.';
            const ns = 'http://www.w3.org/2000/svg';
            const bg = document.createElementNS(ns, 'rect');
            bg.setAttribute('width', '1000'); bg.setAttribute('height', '890'); bg.setAttribute('fill', '#fff');
            clone.insertBefore(bg, clone.firstChild);
            const style = document.createElementNS(ns, 'style');
            style.textContent = `@font-face{font-family:Geist;src:url(data:font/woff2;base64,${font}) format('woff2');font-weight:100 900}text{font-family:Geist,Arial,sans-serif!important}`;
            clone.querySelector('defs').append(style);
            return new XMLSerializer().serializeToString(clone);
        }""", font)
        (output / "architecture.svg").write_text(svg, encoding="utf-8")
        page.set_content('<html><body style="margin:0;background:white">' + svg + '</body></html>')
        page.evaluate("document.fonts.ready")
        page.locator("svg").screenshot(path=str(output / "architecture.png"))
        browser.close()
    print(f"Exported architecture.svg and architecture.png to {output}")


if __name__ == "__main__":
    main()
