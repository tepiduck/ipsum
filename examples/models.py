"""Trusted factories for `faultline capture examples/models.py:mlp --output model.json`."""

import torch
from torch import nn
from torch.nn import functional as F


def mlp():
    torch.manual_seed(23)
    model = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.LayerNorm(16), nn.Linear(16, 4))
    return model.eval(), (torch.randn(3, 8),)


def convolution():
    torch.manual_seed(23)
    model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((2, 2)))
    return model.eval(), (torch.randn(2, 3, 8, 8),)


class Attention(nn.Module):
    def forward(self, query, key, value):
        return F.scaled_dot_product_attention(query, key, value, is_causal=True)


def attention():
    torch.manual_seed(23)
    return Attention().eval(), tuple(torch.randn(2, 2, 5, 4) for _ in range(3))


def requests():
    """Use `capture ...:requests --trace`; requests reuse one model during execution."""
    model, _ = mlp()
    return model, [(torch.randn(n, 8),) for n in (5, 2, 7, 3, 6, 2, 8)]
