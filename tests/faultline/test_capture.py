import pytest
import torch
from torch import nn

from faultline.capture import capture
from faultline.lowering import build, inputs_for


def test_captures_mlp_weights_and_layer_norm_without_changing_results():
    torch.manual_seed(3)
    model = nn.Sequential(
        nn.Linear(7, 17), nn.LayerNorm(17), nn.ReLU(), nn.Linear(17, 3), nn.Softmax(dim=-1)
    ).eval()
    inputs = (torch.randn(5, 7),)
    case = capture(model, inputs)
    actual = build(case)(*inputs_for(case))[0]
    torch.testing.assert_close(actual, model(*inputs), rtol=1e-5, atol=1e-6)
    assert all(i.values is not None for i in case.inputs)


def test_captures_function_graph_with_transposed_input():
    class Model(nn.Module):
        def forward(self, x):
            return torch.relu(x + 1).sum(dim=-1, keepdim=True)

    model = Model().eval()
    args = (torch.randn(7, 3).t(),)
    case = capture(model, args)
    assert case.inputs[0].layout == "transposed"
    torch.testing.assert_close(build(case)(*inputs_for(case))[0], model(*args))


def test_training_and_unsupported_modules_are_rejected():
    with pytest.raises(ValueError, match="model.eval"):
        capture(nn.Linear(3, 5), (torch.randn(2, 3),))
    model = nn.Sequential(nn.Conv2d(1, 2, 3)).eval()
    with pytest.raises(ValueError, match="Unsupported module"):
        capture(model, (torch.randn(1, 1, 7, 7),))


def test_root_layer_norm_is_captured_and_invalid_normalized_shape_is_rejected():
    model = nn.LayerNorm(7).eval()
    args = (torch.randn(3, 7),)
    case = capture(model, args)
    torch.testing.assert_close(build(case)(*inputs_for(case))[0], model(*args))
    with pytest.raises(ValueError, match="normalized shape"):
        capture(nn.LayerNorm(5, elementwise_affine=False).eval(), args)
