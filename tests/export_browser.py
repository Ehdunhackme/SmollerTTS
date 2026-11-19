"""Compare a contributor's browser export to its native checkpoint, without audio fixtures."""
import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch
from playwright.sync_api import sync_playwright

from smollertts.assets import load_vocoder
from smollertts.checkpoints import load_acoustic, sha256
from smollertts.export import compare_audio
from smollertts.infer import synthesize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5175/projects/smollertts/")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocoder", type=Path, default=Path("assets/vocoder"))
    parser.add_argument("--prompts", type=Path, default=Path("configs/listening-prompts.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    model, _ = load_acoustic(args.checkpoint)
    model.eval()
    vocoder = load_vocoder(args.vocoder)
    rows = json.loads(args.prompts.read_text(encoding="utf-8"))["samples"]
    reports = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url)
        page.wait_for_function("window.ready || window.failure", timeout=180000)
        assert page.evaluate("window.ready"), page.evaluate("window.failure")
        metadata = page.evaluate("async () => (await fetch('./model.json')).json()")
        assert metadata["checkpoint_sha256"] == sha256(args.checkpoint), "Browser is serving a different checkpoint"
        page.evaluate("""async () => {
          const {loadEngine}=await import('./engine.mjs');
          const models=Object.fromEntries(await Promise.all(['encoder','frames','vocoder'].map(async name=>
            [name,await (await fetch(`./${name}.onnx`)).arrayBuffer()])));
          window.parityEngine=await loadEngine(models);
        }""")
        try:
            for row in rows:
                for chunk in row["chunks"]:
                    native, durations = synthesize(model, vocoder, torch.tensor([chunk["tokens"]]), torch.tensor([chunk["stress"]]))
                    result = page.evaluate("""async chunk => {
                      const {wave,durations}=await window.parityEngine.run(chunk);
                      const bytes=new Uint8Array(wave.buffer,wave.byteOffset,wave.byteLength);
                      let binary='';
                      for(let i=0;i<bytes.length;i+=16384)binary+=String.fromCharCode(...bytes.subarray(i,i+16384));
                      return {pcm:btoa(binary),durations};
                    }""", {"tokens": chunk["tokens"], "stress": chunk["stress"]})
                    np.testing.assert_array_equal(durations.numpy(), result["durations"])
                    actual = np.frombuffer(base64.b64decode(result["pcm"]), dtype="<f4")
                    reports.append({"text": row["text"], **compare_audio(native.numpy(), actual)})
        finally:
            page.evaluate("window.parityEngine.close()")
            browser.close()
        assert not errors, errors
    print(json.dumps({"browser_parity_passed": True, "chunks": len(reports), "cases": reports}, indent=2))


if __name__ == "__main__":
    main()
