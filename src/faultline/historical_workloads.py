"""Small independently sized fixtures for documented PyTorch issues.

Issue links and reported environments live in historical.py. These are reduced
operation patterns, not a representative sample of production model failures.
"""

import torch
from torch import nn


class ExpandedIndex(nn.Module):
    def forward(self, values, indices):
        storage = values.new_zeros((1, values.shape[1]))
        return storage.expand(5, -1).index_add(0, indices, values)


def expanded_index():
    values = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)
    return ExpandedIndex().eval(), (values, torch.tensor([0, 2, 0]))


class DiagonalBackward(nn.Module):
    def forward(self, values, replacements):
        return torch.diagonal_scatter(values, replacements).sum()


def diagonal_backward():
    torch.manual_seed(18)
    return DiagonalBackward().eval(), (torch.randn(5, 5), torch.randn(5))


class SliceBackward(nn.Module):
    def forward(self, values, replacements):
        return torch.slice_scatter(values, replacements, dim=1, start=0, end=3).sum(dim=1)


def slice_backward():
    torch.manual_seed(18)
    return SliceBackward().eval(), (torch.randn(2, 7, 9), torch.randn(2, 3, 9))


class CastRoundtrip(nn.Module):
    def forward(self, values):
        rounded = values.to(torch.float16).to(torch.float32)
        return rounded, rounded.sum(dim=1)


def cast_roundtrip():
    values = (torch.arange(42, dtype=torch.float32).reshape(7, 6).t() + 0.3) / 11
    return CastRoundtrip().eval(), (values,)
