import json
import subprocess
import sys
from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from faultline.capture import capture
from faultline.evaluate import Settings, evaluate
from faultline.export import export_case
from faultline.ir import Case
from faultline.lowering import build, inputs_for
from faultline.model_source import capture_trace


class Attention(nn.Module):
    def __init__(self, causal=False):
        super().__init__()
        self.causal = causal

    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)


def models():
    torch.manual_seed(19)
    return [
        (nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 2)), (torch.randn(3, 4),)),
        (nn.GELU(approximate="tanh"), (torch.randn(3, 4),)),
        (
            nn.Sequential(
                nn.Conv2d(4, 6, 3, stride=2, padding=2, dilation=2, groups=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((2, 3)),
            ),
            (torch.randn(2, 4, 9, 9),),
        ),
        (Attention(), tuple(torch.randn(2, 2, 4, 3) for _ in range(3))),
        (Attention(True), tuple(torch.randn(2, 2, 4, 3) for _ in range(3))),
        (nn.LayerNorm(4), (torch.randn(3, 4),)),
    ]


@pytest.mark.parametrize("model,args", models())
def test_export_round_trip_matches_original_forward_and_backward(model, args):
    model.eval()
    case = capture(model, args, frontend="export")
    assert case.schema_version == 2
    assert Case.from_dict(case.to_dict()).digest() == case.digest()
    actual = build(case)(*inputs_for(case))
    torch.testing.assert_close(actual, (model(*args),))
    result = evaluate(case, Settings(backend="eager", gradients=True, repeats=2))
    assert result["status"] == "pass", result
    # Independent backward comparison against the original module, not just two lowerings.
    original_args = [x.detach().requires_grad_() for x in args]
    original_parameters = [p for p in model.parameters() if p.requires_grad]
    expected = torch.autograd.grad(
        model(*original_args).square().mean(), original_parameters + original_args
    )
    lowered = list(inputs_for(case))
    selected = [
        i
        for i, spec in enumerate(case.inputs)
        if spec.role == "input" or (spec.role == "parameter" and spec.requires_grad)
    ]
    for i in selected:
        lowered[i].requires_grad_()
    observed = torch.autograd.grad(
        build(case)(*lowered)[0].square().mean(), [lowered[i] for i in selected]
    )
    torch.testing.assert_close(observed, expected)


def test_attention_mask_permutation_and_noncontiguous_input():
    class Model(nn.Module):
        def forward(self, q, k, v, mask):
            return F.scaled_dot_product_attention(q, k, v, mask, scale=0.4).permute(0, 2, 1, 3)

    args = (*[torch.randn(1, 2, 3, 4).transpose(-1, -2) for _ in range(3)], torch.zeros(4, 4))
    case = capture(Model().eval(), args, frontend="export")
    assert case.inputs[0].layout == "transposed"
    torch.testing.assert_close(build(case)(*inputs_for(case)), (Model()(*args),))


def test_capture_rejects_training_unsupported_ops_and_mutation():
    with pytest.raises(ValueError, match="eval"):
        capture(nn.Linear(3, 3), (torch.ones(2, 3),), frontend="export")
    with pytest.raises(ValueError, match="Unsupported exported operator"):
        capture(nn.SiLU().eval(), (torch.ones(2, 3),), frontend="export")

    class Mutable(nn.Module):
        def forward(self, x):
            return x.add_(1)

    with pytest.raises(ValueError, match="mutation|Unsupported exported operator"):
        capture(Mutable().eval(), (torch.ones(2, 3),), frontend="export")


def test_trace_capture_preserves_shared_parameters_and_checks_graph():
    model = nn.Sequential(nn.Linear(4, 6), nn.GELU()).eval()
    trace = capture_trace(model, [(torch.ones(n, 4),) for n in (2, 3, 2)])
    assert len(trace.steps) == 3
    assert evaluate(trace, Settings(backend="eager", gradients=True, repeats=2))["status"] == "pass"
    parameter = trace.inputs[0]
    changed = replace(parameter, values=tuple(x + 1 for x in parameter.values))
    bad = replace(trace.steps[1], inputs=(changed, *trace.steps[1].inputs[1:]))
    with pytest.raises(ValueError, match="stay fixed"):
        replace(trace, steps=(trace.steps[0], bad)).validate()


@pytest.mark.compiler
@pytest.mark.parametrize("model,args", models())
def test_real_inductor_exported_forward_and_backward(model, args):
    case = capture(model.eval(), args, frontend="export")
    result = evaluate(case, Settings(backend="inductor", gradients=True, repeats=2))
    assert result["status"] == "pass", result


@pytest.mark.parametrize("index", [0, 2, 4])
def test_captured_model_standalone_runs_without_package(tmp_path, index):
    model, args = models()[index]
    case = capture(model.eval(), args, frontend="export")
    path = export_case(case, Settings(backend="eager", gradients=True, repeats=2), tmp_path)
    result = subprocess.run(
        [sys.executable, "-I", str(path)], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1])["status"] == "pass"


def test_schema_two_shape_and_role_guards():
    model, args = models()[2]
    case = capture(model.eval(), args, frontend="export")
    conv = next(node for node in case.nodes if node.op == "conv2d")
    for params in (
        {**conv.params, "groups": 0},
        {**conv.params, "stride": [0, 1]},
        {**conv.params, "padding": [-1, 1]},
        {**conv.params, "groups": 3},
    ):
        changed = tuple(replace(n, params=params) if n.name == conv.name else n for n in case.nodes)
        with pytest.raises(ValueError):
            replace(case, nodes=changed).validate()
    with pytest.raises(ValueError, match="schema 2"):
        replace(case, schema_version=1).validate()
    with pytest.raises(ValueError, match="role"):
        replace(case, inputs=(replace(case.inputs[0], role="unknown"), *case.inputs[1:])).validate()
