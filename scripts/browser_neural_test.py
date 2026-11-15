import json
from pathlib import Path
from playwright.sync_api import sync_playwright

fixtures = json.loads(Path('artifacts/browser-neural-fixture.json').read_text())
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=['--disable-gpu'])
    page = browser.new_page()
    page.goto('http://127.0.0.1:5173')
    page.wait_for_load_state('networkidle')
    report = page.evaluate('''async fixtures => {
      const ort = await import('/runtime/ort.wasm.min.mjs');
      ort.env.wasm.wasmPaths = location.origin + '/runtime/';
      ort.env.wasm.numThreads = 1;
      const encoder = await ort.InferenceSession.create('/model/encoder.onnx', { executionProviders: ['wasm'] });
      const decoder = await ort.InferenceSession.create('/model/decoder.onnx', { executionProviders: ['wasm'] });
      const tensor = item => new ort.Tensor(item.dtype,
        item.dtype === 'int64' ? BigInt64Array.from(item.values, BigInt) : Float32Array.from(item.values), item.shape);
      const errors = [];
      for (const fixture of fixtures) {
        const feeds = Object.fromEntries(['tokens','language','tones','stress'].map(key => [key,tensor(fixture[key])]));
        const encoded = await encoder.run(feeds);
        const controls = await decoder.run({encoded: encoded.encoded,
          ...Object.fromEntries(['frame_indices','positions','frame_mask'].map(key => [key,tensor(fixture[key])]))});
        for (const [key, value] of Object.entries({...encoded,...controls})) {
          let maximum = 0, scaledMaximum = 0;
          for (let i = 0; i < value.data.length; i++) {
            const delta = Math.abs(value.data[i] - fixture[key].values[i]);
            maximum = Math.max(maximum,delta);
            scaledMaximum = Math.max(scaledMaximum,delta/(2e-5+2e-4*Math.abs(fixture[key].values[i])));
          }
          errors.push({language: Number(feeds.language.data[0]), tensor:key, maximum, scaledMaximum});
        }
      }
      return {backend:'wasm-cpu', errors};
    }''', fixtures)
    print(json.dumps(report, indent=2))
    assert max(item['scaledMaximum'] for item in report['errors']) < 1
    Path('artifacts/browser/neural-parity.json').write_text(json.dumps(report, indent=2))
    browser.close()
