import json

import pytest

from faultline.code_diff import (
    compare_dumps,
    inventory,
    normalize_code,
    summarize_code,
    summarize_ir,
)


def emitted(root="/temporary/a", dtype="float", operation="addmm"):
    return f'''# AOT ID: ['0_forward']
cpp_fused_sum_0 = async_compile.cpp_pybinding(['const {dtype}*'], \'\'\'
#include "{root}/cpp_prefix.h"
void kernel(const {dtype}* x) {{
    at::vec::Vectorized<{dtype}> v;
    auto z = vec_reduce_all(v);
}}
\'\'\')
def call(args):
    cpp_fused_sum_0(args[0])
    extern_kernels.{operation}(args[0], args[1])
    return torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(*args)
'''


def test_normalization_preserves_material_math_and_types():
    a, b = emitted(), emitted("/temporary/b").replace("0_forward", "5_forward")
    assert normalize_code(a) == normalize_code(b)
    assert normalize_code(a) != normalize_code(emitted(dtype="double"))
    assert normalize_code(a) != normalize_code(emitted(operation="mm"))
    summary = summarize_code(a)
    assert summary["kernel_definition_count"] == 1
    assert summary["generated_kernel_call_sites"] == {"cpp_fused_sum_0": 1}
    assert summary["extern_call_sites"] == {"extern_kernels.addmm": 1}
    assert summary["kernel_definitions"][0]["vector_types"] == ["float"]
    assert summary["kernel_definitions"][0]["reduction_markers"] == {"vec_reduce_all": 1}
    assert summary["operator_call_sites"] == {
        "torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default": 1
    }


def test_dump_comparison_keeps_occurrences_and_preserves_evidence(tmp_path):
    for side, count in (("before", 2), ("after", 1)):
        for index in range(count):
            folder = tmp_path / side / str(index)
            folder.mkdir(parents=True)
            (folder / "output_code.py").write_text(emitted(f"/tmp/{side}/{index}"))
    out = tmp_path / "diff"
    result = compare_dumps(tmp_path / "before", tmp_path / "after", out)
    assert result["same_normalized_modules"]
    assert result["unique_modules"] == [1, 1]
    assert len(result["before"]) == 2
    assert result["companions"]["ir_post_fusion.txt"]["status"] == "incomplete"
    assert result["companions"]["ir_post_fusion.txt"]["same_content_for_modules"] is None
    assert (out / "generated-code.diff").read_text() == ""
    assert json.loads((out / "summary.json").read_text()) == result
    with pytest.raises(FileExistsError):
        compare_dumps(tmp_path / "before", tmp_path / "after", out)
    (tmp_path / "after/0/output_code.py").write_text(emitted(dtype="double"))
    changed = compare_dumps(tmp_path / "before", tmp_path / "after", tmp_path / "diff2")
    assert not changed["same_normalized_modules"]
    assert "double" in (tmp_path / "diff2/generated-code.diff").read_text()


def test_empty_or_invalid_dump_is_not_agreement(tmp_path):
    with pytest.raises(ValueError, match="No output_code"):
        inventory(tmp_path)
    (tmp_path / "output_code.py").write_text("this is not valid python!")
    with pytest.raises(SyntaxError):
        inventory(tmp_path)


def test_ir_fusion_groups_do_not_double_count_nested_children():
    text = """op0_op1: FusedSchedulerNode(SchedulerNode,SchedulerNode)
op0_op1.snodes[0] =
op0: SchedulerNode(ComputedBuffer)
    reduction = ops.reduction(torch.float32, torch.float32, 'sum', tmp0)
op0_op1.snodes[1] =
op1: SchedulerNode(ComputedBuffer)
op2: ExternKernelSchedulerNode(ExternKernelOut)
"""
    result = summarize_ir(text)
    assert result["top_level_node_count"] == 2
    assert result["nested_child_count"] == 2
    assert result["top_level_fusion_groups"] == {
        "op0_op1": "FusedSchedulerNode(SchedulerNode,SchedulerNode)"
    }
    assert result["reduction_signatures"] == {"torch.float32, torch.float32, 'sum'": 1}
    assert summarize_ir("different debug format")["status"] == "unrecognized_format"


def test_identical_code_does_not_hide_changed_ir(tmp_path):
    for side, kind in (("before", "sum"), ("after", "max")):
        folder = tmp_path / side
        folder.mkdir()
        (folder / "output_code.py").write_text(emitted())
        (folder / "ir_post_fusion.txt").write_text(
            "op0: SchedulerNode(ComputedBuffer)\n"
            f"    reduction = ops.reduction(torch.float32, torch.float32, '{kind}', tmp0)\n"
        )
    result = compare_dumps(tmp_path / "before", tmp_path / "after", tmp_path / "diff")
    assert result["same_normalized_modules"]
    companion = result["companions"]["ir_post_fusion.txt"]
    assert companion["status"] == "complete"
    assert companion["available"] == companion["total"] == [1, 1]
    assert companion["same_content_for_modules"] is False


def test_fx_comment_changes_are_distinct_from_arithmetic_changes(tmp_path):
    for side in ("before", "after"):
        folder = tmp_path / side
        folder.mkdir()
        (folder / "output_code.py").write_text(emitted())
        (folder / "fx_graph_transformed.py").write_text(f"# File: {side}/model.py\nx = a + 1\n")
    result = compare_dumps(tmp_path / "before", tmp_path / "after", tmp_path / "diff")
    companion = result["companions"]["fx_graph_transformed.py"]
    assert companion["same_content_for_modules"] is False
    assert companion["same_syntax_for_modules"] is True
    (tmp_path / "after/fx_graph_transformed.py").write_text("x = a + 2\n")
    changed = compare_dumps(tmp_path / "before", tmp_path / "after", tmp_path / "diff2")
    assert changed["companions"]["fx_graph_transformed.py"]["same_syntax_for_modules"] is False
