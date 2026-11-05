"""Capture an actual PyTorch model; then run the case with the Faultline CLI."""

from pathlib import Path

import torch
from torch import nn

from faultline.capture import capture
from faultline.storage import atomic_json

torch.manual_seed(7)
model = nn.Sequential(nn.Linear(17, 31), nn.LayerNorm(31), nn.ReLU(), nn.Linear(31, 5)).eval()
case = capture(model, (torch.randn(7, 17),), tag="captured-mlp")
target = Path("faultline-runs/mlp-case.json")
atomic_json(target, case.to_dict())
print(target)
