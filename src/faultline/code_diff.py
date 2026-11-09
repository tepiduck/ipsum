"""Read-only summaries of Inductor debug dumps, independent of numerical verdicts."""

import ast
import difflib
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_code(text):
    """Remove debug identities, retaining arithmetic, types, shapes and kernel names.

    A normalized equality does not imply linked-library/header equality. In particular,
    cpp_prefix.h includes ATen vector headers whose implementations can change separately.
    """
    text = text.replace("\r\n", "\n")
    text = re.sub(r"(?m)^# AOT ID:.*\n", "", text)
    text = re.sub(
        r'(#include\s+")[^"\n]*[/\\]cpp_prefix\.h(")',
        r"\1<cache>/cpp_prefix.h\2",
        text,
    )
    return text


def summarize_code(text):
    """Count emitted definitions and call sites, not runtime launches or measured fusion."""
    normalized = normalize_code(text)
    tree = ast.parse(normalized)
    kernels = []
    external = Counter()
    aten = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
            function = ast.unparse(call.func)
            if function.startswith("async_compile."):
                strings = [arg.value for arg in call.args if isinstance(arg, ast.Constant)]
                source = max((x for x in strings if isinstance(x, str)), key=len, default="")
                kernels.append(
                    {
                        "names": [ast.unparse(target) for target in node.targets],
                        "compiler": function,
                        "source_sha256": sha256(source),
                        "reduction_markers": dict(
                            Counter(
                                re.findall(
                                    r"\b(?:Welford\w*|welford\w*|vec_reduce_all|"
                                    r"reduction_(?:sum|max|min|prod)|sum_masked_reduce)\b",
                                    source,
                                )
                            )
                        ),
                        "vector_types": sorted(
                            set(re.findall(r"(?:Vectorized|VectorizedN)<([^>]+)>", source))
                        ),
                        "scalar_types": dict(
                            Counter(re.findall(r"\b(float|double|half|bfloat16)\b", source))
                        ),
                        "explicit_accumulator_types": dict(
                            Counter(
                                re.findall(
                                    r"\b(float|double|int64_t|int32_t|bool)\s+tmp_acc\w*",
                                    source,
                                )
                            )
                        ),
                    }
                )
        if isinstance(node, ast.Call):
            function = ast.unparse(node.func)
            if function.startswith("extern_kernels."):
                external[function] += 1
            if function.startswith(("torch.ops.", "aten.")):
                aten[function] += 1
    names = {name for kernel in kernels for name in kernel["names"]}
    generated_calls = Counter(
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) in names
    )
    return {
        "normalized_sha256": sha256(normalized),
        "kernel_definitions": kernels,
        "kernel_definition_count": len(kernels),
        "generated_kernel_call_sites": dict(sorted(generated_calls.items())),
        "extern_call_sites": dict(sorted(external.items())),
        "operator_call_sites": dict(sorted(aten.items())),
        "scope": "Static emitted code. Counts are definitions/call sites, not runtime launches. "
        "Reduction markers and type tokens are observations, not inferred accumulation semantics. "
        "Equal generated code does not establish equal linked libraries or included headers.",
    }


def summarize_ir(text):
    """Exclude nested child dumps when counting top-level scheduler/fusion groups."""
    text = text.replace("\r\n", "\n")
    headers = dict(re.findall(r"^(op\d+(?:_op\d+)*): ([^\n]+)", text, re.MULTILINE))
    children = set(re.findall(r"\.snodes\[\d+\]\s*=\s*\n(op\d+(?:_op\d+)*):", text))
    top = {name: kind for name, kind in headers.items() if name not in children}
    return {
        "normalized_sha256": sha256(text),
        "status": "parsed" if headers else "unrecognized_format",
        "top_level_node_count": len(top),
        "top_level_node_kinds": dict(sorted(Counter(top.values()).items())),
        "top_level_fusion_groups": {
            name: kind for name, kind in top.items() if "FusedSchedulerNode" in kind
        },
        "nested_child_count": len(children),
        "reduction_signatures": dict(
            sorted(
                Counter(
                    re.findall(
                        r"ops\.reduction\((torch\.\w+, torch\.\w+, ['\"]\w+['\"])",
                        text,
                    )
                ).items()
            )
        ),
    }


def inventory(directory):
    """Inventory only named debug artifacts; never import or execute their contents."""
    directory = Path(directory).resolve()
    files = []
    for path in sorted(directory.rglob("output_code.py")):
        if not path.resolve().is_relative_to(directory):
            raise ValueError("Debug artifact resolves outside the selected directory")
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        companions = {}
        for name in ("ir_pre_fusion.txt", "ir_post_fusion.txt", "fx_graph_transformed.py"):
            companion = path.with_name(name)
            if companion.exists():
                if not companion.resolve().is_relative_to(directory):
                    raise ValueError("Debug companion resolves outside the selected directory")
                companion_raw = companion.read_bytes()
                content = companion_raw.decode("utf-8")
                companions[name] = {
                    "raw_sha256": hashlib.sha256(companion_raw).hexdigest(),
                    "summary": summarize_ir(content)
                    if name.startswith("ir_")
                    else {
                        "normalized_sha256": sha256(normalize_code(content)),
                        "syntax_sha256": sha256(ast.dump(ast.parse(content))),
                    },
                }
        files.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "summary": summarize_code(text),
                "companions": companions,
                "normalized": normalize_code(text),
            }
        )
    if not files:
        raise ValueError("No output_code.py debug artifacts found")
    return files


def compare_dumps(before, after, output):
    """Write an advisory inventory and diff without overwriting earlier evidence."""
    left, right = inventory(before), inventory(after)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # Keep occurrences visible; deduplicate only the content-level comparison.
    sides = []
    for side in (left, right):
        unique = {}
        for row in side:
            unique.setdefault(row["summary"]["normalized_sha256"], row["normalized"])
        sides.append(unique)
    shared = set(sides[0]) & set(sides[1])
    companion_comparison = {}
    for name in ("ir_pre_fusion.txt", "ir_post_fusion.txt", "fx_graph_transformed.py"):
        counts = [sum(name in row["companions"] for row in side) for side in (left, right)]
        complete = counts == [len(left), len(right)]
        hashes = [
            {
                (
                    row["summary"]["normalized_sha256"],
                    row["companions"][name]["summary"]["normalized_sha256"],
                )
                for row in side
                if name in row["companions"]
            }
            for side in (left, right)
        ]
        companion_comparison[name] = {
            "status": "complete" if complete else "incomplete",
            "available": counts,
            "total": [len(left), len(right)],
            "same_content_for_modules": hashes[0] == hashes[1] if complete else None,
        }
        if name == "fx_graph_transformed.py":
            syntax = [
                {
                    (
                        row["summary"]["normalized_sha256"],
                        row["companions"][name]["summary"]["syntax_sha256"],
                    )
                    for row in side
                    if name in row["companions"]
                }
                for side in (left, right)
            ]
            companion_comparison[name]["same_syntax_for_modules"] = (
                syntax[0] == syntax[1] if complete else None
            )
    summary = {
        "schema_version": 1,
        "advisory_only": True,
        "before": [{k: v for k, v in row.items() if k != "normalized"} for row in left],
        "after": [{k: v for k, v in row.items() if k != "normalized"} for row in right],
        "unique_modules": [len(side) for side in sides],
        "shared_normalized_modules": len(shared),
        "same_normalized_modules": set(sides[0]) == set(sides[1]),
        "companions": companion_comparison,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    # Unmatched modules have no presumed one-to-one graph correspondence.
    texts = []
    for side in sides:
        texts.append("\n".join(side[key] for key in sorted(set(side) - shared)))
    diff = "".join(
        difflib.unified_diff(
            texts[0].splitlines(keepends=True),
            texts[1].splitlines(keepends=True),
            fromfile="before/unmatched-modules",
            tofile="after/unmatched-modules",
        )
    )
    (output / "generated-code.diff").write_text(diff, encoding="utf-8")
    return summary
