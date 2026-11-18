"""Test the built browser demo, including waveform regression fixtures.

Install Playwright and Chromium, start npm start, then run:
    python tests/browser_smoke.py --url http://127.0.0.1:5175/projects/smollertts/
"""
import argparse
from array import array
import hashlib
import io
import json
from pathlib import Path
import tempfile
import wave

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def compare_wav(actual, expected):
    if actual == expected:
        return {"byte_identical": True, "max_pcm_difference": 0}
    # Allow two PCM16 units for numerical differences between browser/CPU builds.
    # Length, sample rate, channel count and all samples are still checked.
    samples = []
    for raw in (actual, expected):
        with wave.open(io.BytesIO(raw)) as wav:
            assert wav.getnchannels() == 1 and wav.getsampwidth() == 2
            assert wav.getframerate() == 22050
            samples.append(array("h", wav.readframes(wav.getnframes())))
    assert len(samples[0]) == len(samples[1]), "Waveform length changed"
    maximum = max(abs(a - b) for a, b in zip(*samples))
    assert maximum <= 2, f"Waveform regression: maximum PCM16 difference {maximum}"
    return {"byte_identical": False, "max_pcm_difference": maximum}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5175/projects/smollertts/")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(tempfile.mkdtemp(prefix="smollertts-browser-"))
    output.mkdir(parents=True, exist_ok=True)
    rows = json.loads((ROOT / "benchmarks/wer.json").read_text(encoding="utf-8"))["records"]
    errors, failures, results = [], [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("response", lambda response: failures.append(response.url) if response.status >= 400 else None)
        page.goto(args.url)
        page.wait_for_load_state("networkidle")
        page.wait_for_function("window.ready || window.failure", timeout=180000)
        assert page.evaluate("window.ready"), page.evaluate("window.failure")
        page.locator("#settings-toggle").click()
        page.locator("#stream").uncheck()
        page.locator("#punctuation").uncheck()
        # The frozen overview recordings predate optional punctuation steering.
        for row in rows:
            page.locator("#text").fill(row["text"])
            page.locator("#generate").click()
            page.wait_for_function("window.lastResult || document.querySelector('#status').dataset.error === 'true'", timeout=180000)
            result = page.evaluate("window.lastResult")
            assert result, page.locator("#status").inner_text()
            with page.expect_download() as download:
                page.locator("#download").click()
            path = output / row["wav"]
            download.value.save_as(path)
            expected = (ROOT / "benchmarks/audio" / row["wav"]).read_bytes()
            assert hashlib.sha256(expected).hexdigest() == row["wav_sha256"]
            comparison = compare_wav(path.read_bytes(), expected)
            results.append({"prompt_id": row["prompt_id"], **comparison})

        # The editor's default units example exercises normalization and pauses.
        page.locator("#punctuation").check()
        unit = json.loads((ROOT / "tests/fixtures/units.json").read_text(encoding="utf-8"))
        assert hashlib.sha256((ROOT / "tests/fixtures/units.wav").read_bytes()).hexdigest() == unit["sha256"]
        page.locator("#text").fill(unit["text"])
        page.locator("#generate").click()
        page.wait_for_function("window.lastResult || document.querySelector('#status').dataset.error === 'true'", timeout=180000)
        assert page.evaluate("window.lastResult"), page.locator("#status").inner_text()
        with page.expect_download() as download:
            page.locator("#download").click()
        download.value.save_as(output / "units.wav")
        results.append({"prompt_id": "units", **compare_wav((output / "units.wav").read_bytes(), (ROOT / "tests/fixtures/units.wav").read_bytes())})
        page.locator("#settings-toggle").click()
        page.screenshot(path=str(output / "desktop.png"), full_page=True)

        # Check metric switching, responsive layout and the cached reload path.
        for width in (390, 768, 1440):
            page.set_viewport_size({"width": width, "height": 900})
            for metric in ("utmos", "wer"):
                page.locator(f'[data-metric="{metric}"]').click()
                page.wait_for_function("document.querySelector('#chart-points').children.length > 0")
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1"), "Page overflows"
            if width == 390:
                page.screenshot(path=str(output / "mobile.png"), full_page=True)
        page.reload()
        page.wait_for_function("window.ready || window.failure", timeout=180000)
        assert page.evaluate("window.ready")
        assert page.evaluate("window.loadingReport.cachedBytes > 0 && window.loadingReport.networkBytes === 0")
        page.get_by_role("link", name="Credits", exact=True).click()
        page.wait_for_load_state("networkidle")
        assert page.get_by_role("heading", name="Credits", exact=True).is_visible()
        assert not errors, errors
        assert not failures, failures
        browser.close()
    (output / "report.json").write_text(json.dumps({"waveforms": results, "page_errors": errors, "failed_responses": failures}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"waveforms": results, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
