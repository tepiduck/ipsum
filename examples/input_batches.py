"""Ten deterministic, disjoint input batches for a same-version harness control.

faultline model capture examples/input_batches.py:factory --input-batches 10 --output batches
faultline model compare batches --baseline-python /path/to/python --gradients \
    --fp64-advisory --output comparison

This tiny randomly initialized model is a harness control, not a pretrained-model finding.
"""

import torch


def factory():
    torch.manual_seed(160016)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.GELU(), torch.nn.Linear(8, 2)
    ).eval()
    inputs = [(torch.randn(2, 4),) for _ in range(10)]
    return model, inputs
