"""Generate explicitly labeled train/dev diagnostics from an existing checkpoint."""
import argparse
import json
from pathlib import Path
import torch
from xiaotts import LANGUAGES
from xiaotts.model import XiaoTTS, ModelConfig
from xiaotts.dsp import Synthesizer
from xiaotts.train import samples

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--features", type=Path, required=True)
parser.add_argument("--split", choices=("train", "dev"), required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(4)
state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
model = XiaoTTS(ModelConfig(**state["config"])).eval().cuda()
model.load_state_dict(state["model"])
rows = [json.loads(line) for line in (args.features/"index.jsonl").read_text().splitlines()]
selected = [next(row for row in rows if row["language"] == language and row["split"] == args.split)
            for language in LANGUAGES]
samples(model, Synthesizer().cuda(), selected, args.output, "cuda")
