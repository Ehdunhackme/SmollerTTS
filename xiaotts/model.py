"""Shared convolutional phoneme-to-acoustic-control network.

Encoder and frame decoder have explicit tensor interfaces for browser export.
Duration expansion is deterministic indexing, with no learned parameters.
"""
from dataclasses import asdict, dataclass
import torch
from torch import nn


@dataclass
class ModelConfig:
    width: int = 128
    vocabulary: int = 384
    encoder_dilations: tuple = (1, 2, 4, 8)
    decoder_dilations: tuple = (1, 2, 4, 1, 2, 4)
    dropout: float = 0.05
    controls: int = 67


class Block(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.norm_conv = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(width, width, 5, padding=2*dilation,
                                   dilation=dilation, groups=width)
        self.norm_ffn = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(),
                                 nn.Linear(2*width, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        y = (self.norm_conv(x) * mask).transpose(1, 2)
        x = (x + self.dropout(self.depthwise(y).transpose(1, 2))) * mask
        return (x + self.dropout(self.ffn(self.norm_ffn(x)))) * mask


class XiaoTTS(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.phonemes = nn.Embedding(c.vocabulary, c.width, padding_idx=0)
        self.languages = nn.Embedding(5, c.width)
        self.voices = nn.Embedding(5, c.width)
        self.tones = nn.Embedding(6, c.width)
        self.stress = nn.Embedding(4, c.width)
        self.encoder = nn.ModuleList(Block(c.width, d, c.dropout) for d in c.encoder_dilations)
        self.duration = nn.Sequential(nn.LayerNorm(c.width), nn.Linear(c.width, c.width//2),
                                      nn.GELU(), nn.Linear(c.width//2, 1))
        self.frame_projection = nn.Linear(c.width, c.width)
        self.frame_position = nn.Linear(2, c.width)
        self.decoder = nn.ModuleList(Block(c.width, d, c.dropout) for d in c.decoder_dilations)
        self.controls = nn.Linear(c.width, c.controls)
        # Physical units at the output: log Hz, voicing logit, log RMS, log envelopes.
        with torch.no_grad():
            self.controls.bias.zero_()
            self.controls.bias[0] = 5.2
            self.controls.bias[2] = -3.5
            self.duration[-1].bias.fill_(2.0)
        if self.parameter_count >= 1_000_000:
            raise ValueError(f"Inference parameter ceiling exceeded: {self.parameter_count}")

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def encode(self, tokens, language, tones, stress):
        mask = (tokens != 0).unsqueeze(-1).to(self.phonemes.weight.dtype)
        x = self.phonemes(tokens) + self.tones(tones) + self.stress(stress)
        x = (x + self.languages(language)[:, None] + self.voices(language)[:, None]) * mask
        for block in self.encoder:
            x = block(x, mask)
        return x, self.duration(x).squeeze(-1) * mask.squeeze(-1)

    def decode(self, encoded, frame_indices, positions, frame_mask):
        mask = frame_mask.unsqueeze(-1).to(encoded.dtype)
        frames = encoded.gather(1, frame_indices.unsqueeze(-1).expand(-1, -1, encoded.shape[-1]))
        x = (self.frame_projection(frames) + self.frame_position(positions)) * mask
        for block in self.decoder:
            x = block(x, mask)
        return self.controls(x) * mask

    def forward(self, tokens, language, tones, stress, frame_indices, positions, frame_mask):
        encoded, duration = self.encode(tokens, language, tones, stress)
        return duration, self.decode(encoded, frame_indices, positions, frame_mask)


def expand_durations(durations):
    """Single utterance; zero-duration phones are allowed after frame quantization."""
    if durations.ndim != 1 or (durations < 0).any() or durations.sum() == 0:
        raise ValueError("Expected nonnegative durations with at least one output frame")
    ids = torch.repeat_interleave(torch.arange(len(durations), device=durations.device), durations)
    starts = durations.cumsum(0) - durations
    lengths = durations[ids].float()
    relative = (torch.arange(len(ids), device=ids.device) - starts[ids] + 0.5) / lengths
    return ids, torch.stack((relative, lengths.log1p()), -1)
