# CLI and artifact semantics

## Original-model commands

`faultline model` uses binary workload bundles independently of the compact IR commands below.

| Command | Input and output |
|---|---|
| `model capture FILE:FACTORY --output NEW` | Executes trusted Python; saves factory, state dictionary, nested inputs, RNG state, hashes |
| `model run BUNDLE --output NEW` | Runs eager/compiler checks; saves complete workload, logs, report, and standalone reproducer |
| `model compare BUNDLE --baseline-python PYTHON --candidate-python PYTHON --output NEW` | Runs the same snapshot in both environments; classifies the change |
| `model replay RUN --output NEW` | Rechecks saved settings and workload in a fresh process |
| `model reduce RUN --output NEW` | Removes requests, isolates a native graph, attempts reduction, independently confirms |
| `model export RUN --output NEW` | Copies a verified standalone bundle without running it |

Capture accepts `--trace` and repeated `--include RELATIVE_FILE` for explicitly copied source and assets.
Run/capture accept `--python`; compare accepts both interpreter paths. Native reduction and replay
retain saved settings and accept `--python` to select the execution environment. Baseline and
candidate must run on the same operating system; use Linux paths from inside WSL. All output
directories must be new. Source artifacts are not modified.

Run/compare use the execution settings below. For larger models, increase `--timeout` and
`--max-rss-mb` deliberately. The memory watchdog samples aggregate host RSS, including compiler
children, and cannot bound device memory. `--gradients` checks a fixed sum of mean-squared tensor
outputs; it is not a user training loss. `--dynamic` controls compilation, but independent native
graph snapshots concretize captured symbolic integers and are specific to those recorded calls.

`model run` and `model replay` return 0 for agreement, 1 for a stable discrepancy or compiler
exception, and 2 for an inconclusive result. `model compare` uses the classification table below.
`model reduce` returns 0 for a confirmed export and 2 if the saved failure no longer reproduces;
a successful reduction may retain the full workload. Standalone `repro.py` returns 0 for the
expected agreement, 1 for the expected stable failure, and 2 for a changed or inconclusive outcome.
Capture/export return 0 on success. Invalid arguments and artifact-integrity errors are errors.

Native run directories contain `workload/`, `settings.json`, `selection.json`, `evidence.json`,
`run/outcome.json`, `run/graphs.json`, `run/graphs/graph-NNNN/`, `summary.json`, `report.html`,
and `reproducer/`. Graph directories contain readable `forward.py`, structured `graph.json`,
binary `tensors.pt`, and a hashed `workload.json`. Comparison directories additionally retain
baseline/candidate results and reproducers. `--reduce` saves reduction attempts and final evidence.
Bundles contain model weights, exact inputs, source, logs, and environment metadata. Review them
before sharing. Full-model reproducers need the original factory's dependencies and external assets.

`faultline` and `python -m faultline` expose the same commands. `faultline ui` provides a numbered
terminal menu over those commands; it does not start a server. Browser reports are local HTML,
work offline, and load no external scripts or styles.

## Exit codes

| Operation | `0` | `1` | `2` |
|---|---|---|---|
| `run`, `resume` | All completed cases agree | At least one stable discrepancy/compiler exception | Inconclusive result or configuration error |
| `compare` | No new failures and no inconclusive results | New regression or changed failure | Inconclusive result or configuration error |
| `replay` | Not used | Saved stable failure recurs | Failure absent, changed, inconclusive, or invalid input |
| `demo` | Controlled workflow completed | Not used | Workflow or configuration error |
| `generate`, `report`, `reduce`, `export`, `doctor` | Command completed | Not used | Command or probe failed |
| Standalone script with expected finding | Not used | Expected failure family recurs | Expected failure does not recur |
| Standalone script without expected finding | Agreement | Stable discrepancy/compiler exception | Inconclusive or unavailable |

Cancellation returns `130` for a command. The terminal menu remains available after a cancelled
action; Ctrl+C at a prompt or closed input leaves the interface. A run with both stable failures
and inconclusive results returns `1`; consult its counts for the complete picture.

## Execution settings

`run`, `doctor --probe`, and operations on a plain case JSON accept:

| Flag | Default | Meaning |
|---|---|---|
| `--backend` | `inductor` | `inductor`, `aot_eager`, `eager`, or an explicit validation mutant |
| `--device` | `cpu` | CPU or `cuda[:index]`; ROCm uses the same namespace |
| `--repeats` | `3` | Repeated evaluations, from 2 to 20 |
| `--threads` | `1` | PyTorch CPU worker threads |
| `--timeout` | `120` | Wall deadline per worker, in seconds |
| `--max-rss-mb` | `4096` | Sampled aggregate host-process RSS budget |
| `--rtol`, `--atol` | Dtype dependent | Explicit numerical comparison thresholds |
| `--gradients` | Off | Compare scalar loss, input gradients, and trainable parameter gradients |
| `--dynamic` | `static` | `static` (`False`), `auto` (`None`), or `dynamic` (`True`) passed to `torch.compile` |

For finite values, a discrepancy exceeds `atol + rtol * abs(reference)`. Defaults for
`(rtol, atol)` are float64 `(1e-8, 1e-10)`, float32 `(1e-4, 1e-5)`, float16 `(1e-2, 1e-3)`,
and bfloat16 `(5e-2, 5e-3)`. These are engineering thresholds, not universal correctness bounds.
Settings are saved with the evidence. Saved-directory operations reject overrides.

Gradient checking uses `sum(output.square().mean() for output in outputs)`. It tests the derivative
of this loss at the frozen inputs, not the entire Jacobian or a training run. Input tensors become
fresh differentiable leaves on every invocation; parameters are selected according to their saved
trainability. Buffers and constants are excluded. Missing and zero gradients differ. A discrepancy
confined to gradients is `gradient_mismatch`; forward/loss discrepancies keep their numerical or
structural status. Nonfinite reference gradients are inconclusive. Original layouts are retained.

Timeouts, process crashes, memory-limit outcomes, unavailable devices, unstable execution, and
reference failures remain distinct from stable candidate discrepancies. Repeated stability is
required before reduction. CPU float64 comparisons are advisory, not proof of which path is correct.

## Files

Each campaign has a `manifest.json`, `summary.json`, `report.html`, and `cases/` directory. Each
case retains frozen data, execution settings, its result, and worker logs. The report filters by
status or text and expands comparison details. To open an existing report:

```bash
faultline report faultline-runs/compiled --open
```

`resume` checks source, settings, inputs, and relevant environment metadata before continuing.
Completed cases are retained. Use `replay` to intentionally test a saved finding after changing
compiler versions or environments; it records the differences in a new directory.

`reduce` writes each independent attempt, a `reduction.json` decision history, and a `reproducer/`
bundle. Evaluation budgets include the initial check and an uncached final confirmation. Budget
exhaustion is recorded and does not imply local irreducibility.

`export` writes `repro.py`, case/settings/environment JSON, and the saved outcome when supplied.
New bundles include content hashes to catch accidental edits. They are not cryptographic
authenticity signatures. A standalone script requires PyTorch but not Faultline; use `replay`
when process supervision is required, since the standalone script has no wall/RSS watchdog.

## Model capture and traces

`capture path/to/file.py:factory --output model.json` executes a trusted local factory returning
`(model.eval(), tuple_of_example_tensors)`. `--trace` instead expects a list of input tuples.
`--frontend export` is the default. Unsupported operators fail capture with their ATen name.
Capture does not accept arbitrary pickled models. Frozen JSON includes weights and input values.

`trace --case first.json --case second.json --case first.json --output trace.json` assembles existing
cases in exactly the supplied order. Traces allow 1–32 requests, fixed tensor identities, shared
graph structure, and identical parameters/buffers/constants. User input shapes, layouts, and values
may vary. Shape-dependent graph rewrites, such as different concrete reshape parameters, are
rejected rather than combined into a misleading trace. `run`, `reduce`, `replay`, `export`, and
`compare` all accept trace artifacts.

## Comparing environments in CI

```bash
faultline compare --baseline-python .baseline/bin/python --candidate-python .candidate/bin/python --case model.json --gradients --reduce --max-evaluations 64 --output faultline-runs/upgrade
```

Both environments need compatible PyTorch, NumPy, and psutil installations. The coordinator supplies
its own Faultline source to both; it compares compiler environments, not two Faultline versions.
`--backend` applies to both; `--candidate-backend` explicitly selects a different candidate backend.
All other comparison settings remain identical. The per-worker wall/RSS budgets apply independently.

Classification is `agreement` (both pass), `regression` (baseline passes, candidate fails stably),
`not_observed` (only baseline fails), `recurring` (same stable family), `changed_failure` (different
stable families), or `inconclusive`. Existing recurring findings do not fail an upgrade gate;
`not_observed` does not prove a fix. A gate with any regression/changed failure returns 1, even if
other cases are inconclusive. A gate with only agreement/recurring/not-observed results returns 0.

`--reduce` minimizes new regressions using a dual predicate: baseline agreement AND the original
candidate failure. Each evaluation may launch two workers; the budget counts proposal evaluations,
including initial and final checks. Both environments recheck the final case without using the
reducer cache. Failed reduction is recorded without hiding the original regression.

`comparison.json` retains both environments and results; `report.html` links to raw evidence.
Each reduced finding has `baseline-reproducer/`, `candidate-reproducer/`, and `reduction.json`.
Use a fresh output directory. Comparisons currently have no resume command. CI runs real CPU
compiler checks across pinned releases and separately verifies controlled regression detection.

## Backward graph debugging

For native model bundles, `model run` and `model compare` accept `--aot` with `--gradients`.
It records AOT forward/backward graphs and their runtime inputs under `run/graphs/` (or
`baseline/graphs/` and `candidate/graphs/` for comparisons). `aot-graphs.json` maps each graph to
its parent Dynamo graph, stage, execution count, and whether independent export succeeded.
`model replay`, `model reduce`, and `model export` retain this saved policy.

```bash
faultline model run workloads/my-model --gradients --aot --output faultline-runs/backward
faultline model reduce faultline-runs/backward --max-evaluations 32 --output faultline-runs/backward-reduced
python faultline-runs/backward-reduced/reproducer/repro.py
```

For a version comparison, use `model compare ... --gradients --aot --reduce`; the reduction
requires a passing baseline for every accepted proposal and the final confirmation. A standalone
backward graph already contains derivative operations. Its saved cotangents and activations are
ordinary inputs, and its outputs are compared without computing second derivatives.

AOT capture disables the AOTAutograd cache for that diagnostic run, retaining kernel caching.
Unsupported PyTorch hook versions or a missing backward capture are inconclusive (exit 2), not
successful compiler checks. All graph bundles still contain exact tensor data; review before sharing.

## Device checks

```bash
faultline qualify --device cuda:0 --require-rocm --output faultline-runs/rocm
faultline qualify --device cpu --dtype float32 --workload attention --output faultline-runs/cpu-attention
```

`qualify` defaults to `--device cuda:0`, `--backend inductor`, gradient checking, a 300-second
per-worker timeout, and MLP/convolution/attention in float32/float16/bfloat16. Repeated `--dtype`
and `--workload` options select a subset. Standard repeat, thread, tolerance, and memory flags
apply. `--require-rocm` rejects environments without a selected ROCm runtime and GPU device.
It does not install drivers or change PyTorch. The command never falls back to CPU.

`summary.json` and `report.html` link each case to its run, captured graphs, logs, and reproducer.
Exit 0 means every selected case agreed; 1 means a stable discrepancy; 2 means at least one
inconclusive or unavailable check, including when other cases disagreed. These are fixed synthetic
component tests, not a claim of correctness for arbitrary models or GPU performance measurements.

## Useful recipes

Native `model run/compare` additionally accept `--autocast float16|bfloat16` and
`--compile-mode default|reduce-overhead|max-autotune|max-autotune-no-cudagraphs`. These settings
are retained by saved-run operations. They are currently unavailable for compact-IR cases.

`model capture --backward-probe vjp|loss` selects the backward input policy (new captures default
to `vjp`; existing bundles retain their policy). `--fp64-advisory` on native runs, comparisons,
or qualification adds higher-precision numerical evidence without changing the primary verdict.
See [numerics](numerics.md) for the reference semantics and saved-bundle compatibility.

`model diagnose BUNDLE_OR_RUN` executes the four-backend ladder. `benchmark list` and
`benchmark historical` expose the versioned historical corpus. `qualify` additionally accepts
`--require-cuda`, `--no-aot`, and the `buffer_update` / `shape_sequence` workloads. Detailed
semantics, exit codes, and GPU runner setup are in [diagnostics](diagnostics.md).

```bash
# CPU pipeline sanity check without compiling
faultline run --corpus --backend eager --output faultline-runs/eager

# Reproducible generated cases
faultline run --count 20 --seed 100 --depth 12 --dtype float32 --output faultline-runs/generated

# Probe a configured AMD/ROCm or NVIDIA device
faultline doctor --device cuda:0 --backend inductor --probe faultline-runs/gpu-probe

# Continue an interrupted campaign
faultline resume faultline-runs/generated

# Use a separate workspace in the terminal menu
faultline ui --workspace experiments
```

There is no silent fallback to CPU or eager execution. Use fresh output directories. Findings
are symptom families for investigation, not counts of confirmed or unique upstream bugs.
