"""Export a trained checkpoint to the browser's three dynamic ONNX graphs."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from .assets import load_vocoder, VOCODER
from .blocks import duration_indices
from .checkpoints import atomic_json, load_acoustic, sha256
from .infer import predicted_durations


class Encoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens, stress):
        return self.model.encode(tokens, stress)


class Frames(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, encoded, phone_prosody, indices, position):
        mel, _ = self.model.native_decode(encoded, phone_prosody, indices, position, torch.ones_like(position)[..., None])
        return mel.transpose(1, 2)


def compare_audio(reference, actual):
    if reference.shape != actual.shape or not np.isfinite(actual).all():
        raise ValueError("Export changed waveform shape or produced non-finite audio")
    error = np.asarray(reference, dtype=np.float64) - actual
    peak = float(np.max(np.abs(reference)))
    rms = float(np.sqrt(np.mean(np.square(reference.astype(np.float64)))))
    maximum = float(np.max(np.abs(error)))
    rms_error = float(np.sqrt(np.mean(error ** 2)))
    if maximum > max(1e-6, peak * .001) or rms_error > max(1e-7, rms * .0001):
        raise ValueError(f"Native/ONNX waveform parity failed: max={maximum}, RMS={rms_error}")
    return {"max_error": maximum, "rms_error": rms_error, "reference_peak": peak, "reference_rms": rms}


def export(checkpoint, vocoder_path, prompts, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new export directory")
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    model, saved = load_acoustic(checkpoint)
    model.eval()
    vocoder = load_vocoder(vocoder_path)
    encoder, frames = Encoder(model).eval(), Frames(model).eval()
    tokens = torch.tensor([[1, 22, 12, 69, 52, 92, 1]])
    stress = torch.zeros_like(tokens)
    with torch.no_grad():
        encoded, _, phone = encoder(tokens, stress)
        indices, position = duration_indices(torch.full((7,), 5))
        mel = frames(encoded, phone, indices[None], position[None])
    specs = [
        ("encoder", encoder, (tokens, stress), ["tokens", "stress"], ["encoded", "duration", "phone_prosody"],
         {key: {1: "phones"} for key in ("tokens", "stress", "encoded", "duration", "phone_prosody")}),
        ("frames", frames, (encoded, phone, indices[None], position[None]), ["encoded", "phone_prosody", "indices", "position"], ["mel"],
         {"encoded": {1: "phones"}, "phone_prosody": {1: "phones"}, "indices": {1: "frames"}, "position": {1: "frames"}, "mel": {2: "frames"}}),
        ("vocoder", vocoder, (mel,), ["mel"], ["waveform"], {"mel": {2: "frames"}, "waveform": {2: "samples"}}),
    ]
    output.mkdir(parents=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sessions = {}
    for name, module, inputs, inames, onames, axes in specs:
        path = output / (name + ".onnx")
        torch.onnx.export(module, inputs, str(path), input_names=inames, output_names=onames,
                          dynamic_axes=axes, opset_version=17, dynamo=False)
        onnx.checker.check_model(onnx.load(path))
        sessions[name] = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    samples = json.loads(Path(prompts).read_text(encoding="utf-8"))["samples"]
    if len(samples) < 3:
        raise ValueError("Provide at least three listening prompts for dynamic-length checks")
    cases = list(zip(samples[:3], (1, 4, 8))) + [(row, None) for row in samples[:6]]
    reports = []
    for row, scale in cases:
        chunk = row["chunks"][0]
        tokens, stress = torch.tensor([chunk["tokens"]]), torch.tensor([chunk["stress"]])
        with torch.no_grad():
            encoded, duration, phone = encoder(tokens, stress)
            durations = predicted_durations(tokens[0], duration[0]) if scale is None else torch.full((tokens.shape[1],), scale)
            if not 2 <= int(durations.sum()) <= 6000:
                raise ValueError("Export checkpoint predicts invalid durations")
            indices, position = duration_indices(durations)
            native_mel = frames(encoded, phone, indices[None], position[None])
            native_wave = vocoder(native_mel).numpy()
        en, dn, pn = sessions["encoder"].run(None, {"tokens": tokens.numpy(), "stress": stress.numpy()})
        if scale is None:
            decoded = predicted_durations(tokens[0], torch.from_numpy(dn[0]))
            np.testing.assert_array_equal(decoded.numpy(), durations.numpy())
        mn = sessions["frames"].run(None, {"encoded": en, "phone_prosody": pn, "indices": indices[None].numpy(), "position": position[None].numpy()})[0]
        wn = sessions["vocoder"].run(None, {"mel": mn})[0]
        for expected, actual in [(encoded.numpy(), en), (duration.numpy(), dn), (phone.numpy(), pn), (native_mel.numpy(), mn)]:
            np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
        if wn.shape != (1, 1, len(indices) * 256):
            raise ValueError("Unexpected waveform length")
        reports.append({"text": row["text"], "mode": "predicted" if scale is None else "controlled",
                        "phones": tokens.shape[1], "frames": len(indices), **compare_audio(native_wave, wn)})
    graphs = {name: {"bytes": (output / (name + ".onnx")).stat().st_size, "sha256": sha256(output / (name + ".onnx"))} for name in sessions}
    acoustic_count = sum(p.numel() for p in model.parameters())
    vocoder_count = sum(p.numel() for p in vocoder.parameters())
    metadata = {"name": "SmollerTTS", "parameters": acoustic_count + vocoder_count,
                "acoustic_parameters": acoustic_count, "vocoder_parameters": vocoder_count,
                "sample_rate": 22050, "hop": 256, "precision": "FP32", "graphs": graphs,
                "checkpoint_sha256": sha256(checkpoint), "training_step": saved["step"],
                "architecture": "Duration-regulated acoustic model with HiFi-GAN V2", "language": "en", "speakers": 1,
                "arithmetic": "Unfused PyTorch CPU; ONNX graph optimization disabled",
                "vocoder": {"repository": "https://github.com/jik876/hifi-gan", "variant": "LJ_V2", "pretrained": True,
                            "generator_sha256": VOCODER["generator_v2"]["sha256"]}}
    atomic_json(output / "export.json", {"parity_passed": True, "checkpoint_sha256": sha256(checkpoint), "graphs": graphs,
                "cases": reports, "torch": str(torch.__version__), "onnx": onnx.__version__, "onnxruntime": ort.__version__,
                "scope": "Numerical export compatibility, not a speech-quality claim."})
    # Only publish the model manifest after every parity check succeeds.
    atomic_json(output / "model.json", metadata)
    print(json.dumps({"output": str(output), "parameters": metadata["parameters"], "parity_cases": len(reports)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocoder", type=Path, default=Path("assets/vocoder"))
    parser.add_argument("--prompts", type=Path, default=Path("configs/listening-prompts.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.checkpoint, args.vocoder, args.prompts, args.output)
