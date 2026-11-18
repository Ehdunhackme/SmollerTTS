"""Attention and depthwise-convolution blocks."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class FrameBlock(nn.Module):
    def __init__(self, width, expansion=2, dilation=1, scale=.25):
        super().__init__()
        self.depthwise = nn.Conv1d(width, width, 7, padding=3*dilation,
                                   dilation=dilation, groups=width)
        self.norm = nn.LayerNorm(width)
        self.up = nn.Linear(width, width*expansion)
        self.down = nn.Linear(width*expansion, width)
        self.scale = nn.Parameter(torch.full((width,), scale))

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask
        h = self.depthwise(x.transpose(1, 2)).transpose(1, 2)
        h = self.down(F.gelu(self.up(self.norm(h))))
        y = x + h*self.scale
        return y if mask is None else y*mask


class TextBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width, self.heads = width, 4
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3*width)
        self.output = nn.Linear(width, width)
        self.ff = FrameBlock(width, expansion=3)

    def forward(self, x, mask):
        b, n, _ = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, n, 3, self.heads, self.width//self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.width//self.heads)
        scores = scores.masked_fill(~mask[:, None, :, 0].bool()[:, :, None, :], -1e4)
        h = (scores.softmax(-1) @ v).transpose(1, 2).reshape(b, n, self.width)
        return self.ff((x + self.output(h))*mask, mask)


def duration_indices(durations):
    """One utterance; exact zero duration is allowed for silence, never for speech."""
    indices = torch.repeat_interleave(torch.arange(len(durations), device=durations.device), durations)
    starts = torch.cumsum(durations, 0)-durations
    position = (torch.arange(len(indices), device=durations.device)-starts[indices]+.5)/durations[indices].clamp_min(1)
    return indices, position
