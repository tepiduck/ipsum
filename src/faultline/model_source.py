"""Load an explicitly trusted local model factory for data-only capture."""

import importlib.util
import sys
from pathlib import Path

from .capture import capture
from .trace import Trace


def capture_trace(model, requests, *, tag="captured-trace", seed=0, frontend="export"):
    """Capture requests sharing one stateless model; reject changing graph semantics."""
    steps = tuple(
        capture(model, tuple(args), tag=f"{tag}-{i}", seed=seed, frontend=frontend)
        for i, args in enumerate(requests)
    )
    trace = Trace(steps, tag)
    trace.validate()
    return trace


def capture_factory(source, *, trace=False, frontend="export"):
    path, separator, name = source.rpartition(":")
    if not separator or not name.isidentifier():
        raise ValueError("factory must be a local Python file and function: model.py:factory")
    path = Path(path).resolve()
    if not path.is_file() or path.suffix != ".py":
        raise ValueError("factory source must be an existing .py file")
    spec = importlib.util.spec_from_file_location("faultline_user_model", path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(spec.name)
    original_path = sys.path[:]
    try:
        # Match ordinary file execution: sibling imports and decorators can resolve the module.
        sys.modules[spec.name] = module
        sys.path.insert(0, str(path.parent))
        spec.loader.exec_module(module)
        factory = getattr(module, name, None)
        if not callable(factory):
            raise ValueError(f"No callable factory named {name}")  # noqa: TRY004
        model, examples = factory()
        if trace:
            return capture_trace(model, examples, tag=path.stem, frontend=frontend)
        return capture(model, tuple(examples), tag=path.stem, frontend=frontend)
    finally:
        sys.path[:] = original_path
        if previous is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous
