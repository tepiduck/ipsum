"""Original-model factories for `faultline model capture` (no IR translation).

Weights are randomly initialized; these fixtures validate execution, not accuracy.
Install torchvision for resnet18. The other factories require only PyTorch.
"""

import torch
from torch import nn


def resnet18():
    from torchvision.models import resnet18 as build

    torch.manual_seed(21)
    return build(weights=None).eval(), (torch.randn(1, 3, 224, 224),)


class TokenEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8192, 768)
        layer = nn.TransformerEncoderLayer(768, 12, 3072, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 6, enable_nested_tensor=False)
        self.classifier = nn.Linear(768, 128)

    def forward(self, tokens, padding_mask):
        hidden = self.encoder(self.embedding(tokens), src_key_padding_mask=padding_mask)
        return {"logits": self.classifier(hidden[:, 0]), "hidden": hidden}


def transformer():
    torch.manual_seed(21)
    tokens = torch.randint(0, 8192, (2, 32))
    mask = torch.zeros(2, 32, dtype=torch.bool)
    mask[1, 24:] = True
    return TokenEncoder().eval(), {"tokens": tokens, "padding_mask": mask}


class TransformerIndexing(TokenEncoder):
    """Known #183986 operation pattern embedded in a random-weight transformer.

    The aggregation consumes actual transformer outputs. No graph break or compiler mutant
    is inserted; this is a constructed localization experiment, not a newly discovered bug.
    """

    def forward(self, tokens, padding_mask):
        hidden = self.encoder(self.embedding(tokens), src_key_padding_mask=padding_mask)
        values = self.classifier(hidden)[0, :3, :4]
        storage = values.new_zeros((1, values.shape[1]))
        destinations = torch.tensor([0, 2, 0], device=values.device)
        return storage.expand(5, -1).index_add(0, destinations, values)


def transformer_indexing():
    torch.manual_seed(21)
    tokens = torch.randint(0, 8192, (2, 32))
    mask = torch.zeros(2, 32, dtype=torch.bool)
    mask[1, 24:] = True
    return TransformerIndexing().eval(), {"tokens": tokens, "padding_mask": mask}


class ExpandedIndexAdd(nn.Module):
    """Historical stride-zero indexing regression, PyTorch issue #183986.

    https://github.com/pytorch/pytorch/issues/183986
    This independently sized regression fixture is not a new bug discovery.
    """

    def forward(self, values, destinations):
        storage = values.new_zeros((1, values.shape[1]))
        expanded = storage.expand(5, -1)
        return expanded.index_add(0, destinations, values)


def expanded_index_add():
    values = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)
    return ExpandedIndexAdd().eval(), (values, torch.tensor([0, 2, 0]))


def reduction_demo():
    torch.manual_seed(21)
    model = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Sigmoid(), nn.Tanh()).eval()
    return model, (torch.randn(2, 768),)


class DiagonalScatterReduction(nn.Module):
    """Historical backward regression: https://github.com/pytorch/pytorch/issues/182012."""

    def forward(self, matrix, replacements):
        return torch.diagonal_scatter(matrix, replacements).sum()


def scatter_backward():
    torch.manual_seed(18)
    return DiagonalScatterReduction().eval(), (torch.randn(5, 5), torch.randn(5))
