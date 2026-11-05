# CLI and artifact semantics

`faultline` and `python -m faultline` expose the same commands. `faultline ui` provides a numbered
terminal menu over those commands; it does not start a server. Browser reports are local HTML,
work offline, and load no external scripts or styles.

## Exit codes

| Operation | `0` | `1` | `2` |
|---|---|---|---|
| `run`, `resume` | All completed cases agree | At least one stable discrepancy/compiler exception | Inconclusive result or configuration error |
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

For finite values, a discrepancy exceeds `atol + rtol * abs(reference)`. Defaults for
`(rtol, atol)` are float64 `(1e-8, 1e-10)`, float32 `(1e-4, 1e-5)`, float16 `(1e-2, 1e-3)`,
and bfloat16 `(5e-2, 5e-3)`. These are engineering thresholds, not universal correctness bounds.
Settings are saved with the evidence. Saved-directory operations reject overrides.

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
