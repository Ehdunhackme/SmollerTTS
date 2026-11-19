"""Download and verify the original published HiFi-GAN LJ_V2 generator."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request, urlopen

import torch

from .checkpoints import atomic_json, sha256
from .vocoder import Generator

VOCODER = {
    "config.json": {"url": "https://drive.google.com/uc?export=download&id=1vxXvAdsCtAk8jbYBxAd7KANeqNPVtsTW",
                    "sha256": "fb61019b59922086dc4149ede63332ab21a5deb0aa5b51edc79f9995fc4c9736", "bytes": 832},
    "generator_v2": {"url": "https://drive.google.com/uc?export=download&id=1gfouaWecMbmfqIdWYs-KtsULIdYCveYW",
                     "sha256": "3fac378c5918fb2c102733f21eeaa8e9a4ca6cda24dbfddc55bbb947c78d562f", "bytes": 3758167},
}


def verify_vocoder(folder):
    for name, expected in VOCODER.items():
        path = Path(folder) / name
        if not path.is_file() or path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Missing or mismatched official LJ_V2 file: {path}")


def download_vocoder(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, expected in VOCODER.items():
        path = folder / name
        if path.exists():
            if sha256(path) != expected["sha256"]:
                raise ValueError(f"Refusing to overwrite a different file: {path}")
            continue
        temporary = path.with_suffix(".download")
        try:
            with urlopen(Request(expected["url"], headers={"User-Agent": "SmollerTTS"}), timeout=60) as response, temporary.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
                    if stream.tell() > expected["bytes"]:
                        raise ValueError("Unexpected download size")
            if temporary.stat().st_size != expected["bytes"] or sha256(temporary) != expected["sha256"]:
                raise ValueError("Download checksum mismatch; see docs/training.md for manual acquisition")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    verify_vocoder(folder)
    atomic_json(folder / "provenance.json", {"repository": "https://github.com/jik876/hifi-gan", "variant": "LJ_V2", "files": VOCODER})


def load_vocoder(folder, device="cpu"):
    verify_vocoder(folder)
    config = SimpleNamespace(**json.loads((Path(folder) / "config.json").read_text()))
    model = Generator(config)
    state = torch.load(Path(folder) / "generator_v2", map_location="cpu", weights_only=True)
    model.load_state_dict(state["generator"], strict=True)
    model.remove_weight_norm()
    model.eval().requires_grad_(False)
    if sum(p.numel() for p in model.parameters()) != 925985:
        raise ValueError("Unexpected vocoder architecture")
    return model.to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("assets/vocoder"))
    args = parser.parse_args()
    download_vocoder(args.output)
    print(f"Verified original LJ_V2 generator in {args.output}")
