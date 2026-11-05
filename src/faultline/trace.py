"""Versioned sequences of frozen requests sharing one graph and model state."""

import hashlib
import json
from dataclasses import dataclass

from .ir import MAX_ELEMENTS, Case, numel


@dataclass(frozen=True)
class Trace:
    steps: tuple[Case, ...]
    tag: str = "trace"
    schema_version: int = 1

    @property
    def inputs(self):
        return self.steps[0].inputs

    @property
    def nodes(self):
        return self.steps[0].nodes

    @property
    def outputs(self):
        return self.steps[0].outputs

    @property
    def seed(self):
        return self.steps[0].seed

    def to_dict(self):
        return {
            "kind": "trace",
            "schema_version": self.schema_version,
            "tag": self.tag,
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, data):
        if set(data) != {"kind", "schema_version", "tag", "steps"} or data["kind"] != "trace":
            raise ValueError("invalid trace fields")
        result = cls(
            tuple(Case.from_dict(s) for s in data["steps"]), data["tag"], data["schema_version"]
        )
        result.validate()
        return result

    def digest(self):
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, allow_nan=False).encode()
        ).hexdigest()

    def validate(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported trace schema")
        if not isinstance(self.tag, str) or len(self.tag) > 256 or not 1 <= len(self.steps) <= 32:
            raise ValueError("trace requires a tag and 1..32 requests")
        first = self.steps[0]
        signature = [(i.name, i.dtype, i.role, i.requires_grad) for i in first.inputs]
        for step in self.steps:
            step.validate()
            if (
                step.nodes != first.nodes
                or step.outputs != first.outputs
                or step.schema_version != first.schema_version
            ):
                raise ValueError("trace requests must share the same graph")
            if [(i.name, i.dtype, i.role, i.requires_grad) for i in step.inputs] != signature:
                raise ValueError("trace input identities/dtypes must remain fixed")
            if any(i.values is None for i in step.inputs):
                raise ValueError("trace requests require frozen inputs")
            for a, b in zip(first.inputs, step.inputs, strict=True):
                if a.role != "input" and a != b:
                    raise ValueError(
                        "parameters, buffers and constants must stay fixed across a trace"
                    )
        if sum(numel(i.shape) for s in self.steps for i in s.inputs) > 16 * MAX_ELEMENTS:
            raise ValueError("trace exceeds total input budget")
        return first.validate()

    def complexity(self):
        return (
            len(self.steps),
            len(self.nodes),
            *(sum(s.complexity()[i] for s in self.steps) for i in (1, 2, 3)),
        )


def program_from_dict(data):
    if isinstance(data, dict) and data.get("kind") == "trace":
        try:
            return Trace.from_dict(data)
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"invalid trace: {exc}") from exc
    return Case.from_dict(data)
