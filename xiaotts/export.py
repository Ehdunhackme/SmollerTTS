"""Export encoder and frame decoder of ONE shared model, checking CPU parity.

Duration indexing and the zero-parameter audio renderer are separate deterministic
operations. Export parity does not establish browser compatibility or voice quality.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from xiaotts.fetch import atomic_json
from xiaotts.model import ModelConfig, XiaoTTS, expand_durations


class Encoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens, language, tones, stress):
        return self.model.encode(tokens, language, tones, stress)


class Decoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, encoded, frame_indices, positions, frame_mask):
        return self.model.decode(encoded, frame_indices, positions, frame_mask)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(19)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model = XiaoTTS(ModelConfig(**state["config"]))
        model.load_state_dict(state["model"])
    else:
        model = XiaoTTS()
    model.eval()
    args.output.mkdir(parents=True, exist_ok=True)
    tokens = torch.tensor([[3, 4, 5, 6, 7]])
    language = torch.tensor([1])
    tones = torch.tensor([[1, 2, 3, 4, 5]])
    stress = torch.tensor([[0, 1, 0, 2, 0]])
    encoded, _ = model.encode(tokens, language, tones, stress)
    indices, positions = expand_durations(torch.tensor([3, 4, 5, 6, 7]))
    mask = torch.ones(1, len(indices))
    torch.onnx.export(Encoder(model).eval(), (tokens, language, tones, stress), args.output/"encoder.onnx",
                      input_names=["tokens", "language", "tones", "stress"],
                      output_names=["encoded", "log_duration"], opset_version=17, dynamo=False,
                      dynamic_axes={key: {1: "phones"} for key in ("tokens", "tones", "stress", "encoded", "log_duration")})
    torch.onnx.export(Decoder(model).eval(), (encoded, indices[None], positions[None], mask), args.output/"decoder.onnx",
                      input_names=["encoded", "frame_indices", "positions", "frame_mask"],
                      output_names=["controls"], opset_version=17, dynamo=False,
                      dynamic_axes={"encoded": {1: "phones"}, **{key: {1: "frames"} for key in
                                                                 ("frame_indices", "positions", "frame_mask", "controls")}})
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    encoder = ort.InferenceSession(str(args.output/"encoder.onnx"), options, providers=["CPUExecutionProvider"])
    decoder = ort.InferenceSession(str(args.output/"decoder.onnx"), options, providers=["CPUExecutionProvider"])
    model.eval()
    maximum = 0
    for n in (7, 19):
        for lang in range(5):
            inputs = dict(tokens=torch.arange(3, 3+n)[None], language=torch.tensor([lang]),
                          tones=(torch.arange(n)%6)[None], stress=(torch.arange(n)%4)[None])
            with torch.no_grad():
                reference = model.encode(**inputs)
            exported = encoder.run(None, {key:value.numpy() for key,value in inputs.items()})
            for actual, expected in zip(exported, reference):
                np.testing.assert_allclose(actual, expected.numpy(), atol=2e-5, rtol=2e-4)
                maximum = max(maximum, float(np.max(np.abs(actual-expected.numpy()))))
            ids, pos = expand_durations(torch.arange(n)%5+2)
            frame_mask = torch.ones(1, len(ids))
            with torch.no_grad():
                expected = model.decode(reference[0], ids[None], pos[None], frame_mask).numpy()
            actual = decoder.run(None, dict(encoded=exported[0], frame_indices=ids[None].numpy(),
                                             positions=pos[None].numpy(), frame_mask=frame_mask.numpy()))[0]
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)
            maximum = max(maximum, float(np.max(np.abs(actual-expected))))
    report = dict(status="native ONNX CPU parity passed; browser DSP integration pending",
                  trained_checkpoint=str(args.checkpoint) if args.checkpoint else None,
                  model_parameters=model.parameter_count, renderer_parameters=0,
                  config=asdict(model.config), maximum_absolute_error=maximum,
                  checked_languages=5, checked_phone_lengths=[7, 19],
                  files={path.name:path.stat().st_size for path in args.output.glob("*.onnx")})
    atomic_json(args.output/"export.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
