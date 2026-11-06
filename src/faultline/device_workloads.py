"""Small kernel workloads copied into device-qualification bundles; requires only PyTorch."""

import torch
from torch import nn

DTYPE = torch.float32


def mlp():
    torch.manual_seed(31)
    model = nn.Sequential(nn.Linear(32, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 8))
    return model.to(dtype=DTYPE).eval(), (torch.randn(4, 32, dtype=DTYPE),)


def convolution():
    torch.manual_seed(31)
    model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d((2, 2)))
    return model.to(dtype=DTYPE).eval(), (torch.randn(2, 3, 16, 16, dtype=DTYPE),)


class Attention(nn.Module):
    def forward(self, query, key, value, mask):
        return nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=mask)


def attention():
    torch.manual_seed(31)
    values = tuple(torch.randn(2, 2, 8, 16, dtype=DTYPE) for _ in range(3))
    mask = torch.ones(8, 8, dtype=torch.bool).tril()
    return Attention().eval(), (*values, mask)


class BufferUpdate(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("total", torch.zeros(16, dtype=DTYPE))

    def forward(self, value):
        self.total.add_(value.detach().mean(0))
        return value * 0.5 + self.total


def buffer_update():
    torch.manual_seed(31)
    return BufferUpdate().eval(), [(torch.randn(4, 16, dtype=DTYPE),) for _ in range(3)]


def shape_sequence():
    torch.manual_seed(31)
    model = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 8)).to(dtype=DTYPE).eval()
    return model, [(torch.randn(batch, 32, dtype=DTYPE),) for batch in (2, 5, 2, 7)]
