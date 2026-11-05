# CLI and artifact semantics

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

## Useful recipes

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
