# Checking an upgrade

Use representative inputs and the checkpoint you intend to deploy. A small built-in example
is useful for checking the installation; it cannot establish safety for a different workload.

## Inputs

`faultline check model.py:factory` calls trusted Python that returns `(model, inputs)`.
Return an eval-mode `torch.nn.Module`. A tuple is positional arguments; a dict is keyword
arguments. Nested tensor inputs use the existing native-capture semantics. Checkpoint files
and sibling modules needed for replay must be listed with repeated `--include relative/path`.
Captured code must load these assets relative to `__file__`, not the current working directory.

For replication return a list of tuples/dicts, one per independent batch. A list of tensors
is ambiguous with a model's nested inputs: explicitly use `--batches N` to select collection
semantics. All batches must share the same model weights, common leading sample dimension,
and contain distinct samples. Capture preserves layouts and freezes a backward probe for each
batch. A duplicated sample is rejected rather than counted as a new replicate.

Alternatively accept `seed` as a keyword argument. Keep model initialization fixed and use
the seed only for selecting/generating the input batch:

```python
def factory(seed=0):
    torch.manual_seed(0)
    model = MyModel().eval()
    model.load_state_dict(torch.load(Path(__file__).with_name("weights.pt"), weights_only=True))
    inputs = representative_batch(seed)
    return model, inputs
```

The default seeds are 0 through N-1. Frozen state hashes must agree across batches.
A factory without a batch list or seed supplies one batch; `--batches` does not invent
replication. Paired standard/thorough checks then report INCONCLUSIVE for missing replication,
or NEEDS_REVIEW if completed accuracy observations trend worse.
A single-environment check can still be SAFE for that batch, without assessing an upgrade.

Built-in `--torchvision NAME` supports torchvision classification models with pretrained weights;
`faultline sweep list` lists the previously tested subset. Other Hugging Face IDs use AutoModel and may have unsupported semantics;
remote custom code is not enabled. Use `--revision COMMIT` to pin a Hub checkpoint. Built-in
text batches are disjoint book excerpts; vision batches are disjoint strips from two photos.
Shared photos/book and weights limit the statistical independence and domain coverage.

## Execution and levels

Without `--baseline`, compiled and eager execute in `--candidate` (default: current Python).
With two interpreters, both execute the same captured inputs, weights and backward probes.
Each interpreter must have PyTorch and all factory dependencies. `env create` can install
these in new venvs; its printed interpreter path and `ready.json` identify the exact environment.
Manually prepared interpreters also need `numpy`, `psutil` and `packaging` for the workers.
They do not need a second Faultline installation: the coordinator supplies its harness code
while dependencies are imported from the selected interpreter.

Quick uses three batches and forward only. Standard uses ten batches and gradients; paired
checks also obtain fp64 errors. Under v2, shared discrepancies do not require explanation;
candidate-only gradient findings still receive the existing explanation checks.
Thorough runs the existing eager controls and AOT/mask diagnostic path for every eligible batch.
`--batches N` changes the requested batch count, but never lowers the ten-complete-batch threshold
for an accuracy shift. Selecting a captured collection uses its existing batches in order.

The default `--jobs auto` measures the first batch alone, then selects up to four workers
with 25% memory headroom and retries memory-limited batches alone at the full budget.
The default budget is 75% of available RAM capped at 32 GiB; the chosen allocation and
retries are printed and retained in the report and evidence JSON.

These 0.18 execution defaults supersede the scheduling section of the preserved 0.17
policy document. Verdict v2's decision rules remain unchanged.

The measured schedule is `max(1, min(cores // 2, 4, floor(budget / (1.25 * peak))))`,
also capped by the planned batch count. Both environments contribute to the first-batch
peak. A missing peak or first-batch memory limit selects one job. Auto diagnostics run
sequentially with the full budget. Explicit `--jobs N` fixes initial concurrency and
divides the total `--max-rss-mb` budget across workers, including initial diagnostics;
a memory-limited attempt still receives a recorded serial retry at the full total budget.
Retries start only after other owned batch workers finish and never increase the total budget.
Generic crashes and timeouts are not treated as memory failures. Parallel jobs require
`--threads 1`; auto with more threads selects one job. `--timeout` applies per supervised worker.
Coordinator memory and RSS sampling overshoot are outside this bound. The memory limit
is host RSS including descendants, not a GPU VRAM limit or security sandbox.
Use `--deterministic` to request the existing deterministic-algorithm and BLAS workspace policy,
including for AMD/ROCm (`--device cuda:0`). Unsupported deterministic operations remain explicit failures.
Compilation caches are isolated per execution slot and environment. A slot runs its
batches sequentially and exclusively owns its caches; diagnostics reuse the same slot.
Capture, first compilation, and optional diagnostics can dominate wall time.

Accuracy diagnostics consume results in frozen batch order. With ten planned batches,
three non-worsening scores make the 80% worsening condition impossible. Accuracy work
can stop there, but all ten primary comparisons still execute. Reports distinguish
planned, attempted, completed and deliberately skipped accuracy batches and retain the
stopping proof. A positive flag still requires ten completed scores. An unavailable
diagnostic is not intentional early stopping. Use `--no-early-stop` to measure all scores.

## Reviewing results

Start with the reasons in `report.html`; `verdict.json` is the stable CI-facing contract.
Use `evidence.json` and individual worker logs for detailed review. SAFE is scoped to the
selected level. A forward-only SAFE says nothing about backward execution.

During execution, the HTML and JSON reports update after each committed batch. Refresh
the HTML to follow progress; `execution_state: running` identifies an unfinished check.
Use the completed report and command exit code for an upgrade decision.

Two-environment checks default to [policy v2](verdict-policy-v2.md), which asks whether
the candidate made the tested workload worse. The report separates pre-existing differences
from change-related evidence. Shared discrepancies, including changed signatures, do not
block SAFE by themselves. A passing baseline with a stable candidate-only failure, or
replicated accuracy degradation, remains REGRESSION. These are candidates for investigation.

NEEDS_REVIEW identifies undecided change-specific evidence:

- `unstable_candidate_difference`: a passing baseline and recorded candidate-only numerical
  differences that are unstable across repeats.
- `replication_unavailable`: required accuracy evidence is incomplete and observed batch scores
  trend worse. Without that trend, unavailable required evidence is INCONCLUSIVE.

Single-environment checks keep [policy v1](verdict-policy.md), as does an explicit
`--verdict-policy v1`. In that policy, contrary or incompletely explained compile/eager
differences need review, even when both environments exhibit them. Primary statuses,
tolerances, signatures and reducer predicates are unchanged in either policy.

A numerical label is evidence, not a proof of correctness. `accuracy_shift` on one batch is
descriptive only. `accuracy_shift_replicated` requires at least ten completed batches, at least
80% worsening, and median growth of at least 1.25x. Targets within a batch are correlated.
See [the v2 policy](verdict-policy-v2.md); no tolerance changes are made to obtain SAFE.

## Reproducers and existing workflows

For an unexplained stable failure, check exports the first failing workload under `reproducer/`.
Run `python repro.py` in that directory with the required PyTorch version and model dependencies.
The export retains fixed weights, input strides, backward vectors, settings, and any injected-fault
label. It does not need Faultline installed. An accuracy-only shift may have no failing primary
workload; use the paired batch evidence and [replicated bisection](bisect.md) in that case.

Exports can contain proprietary code, model weights, and sensitive inputs. Inspect their contents
before sharing. Do not publish large or private bundles automatically. Native graph reduction is
available through `faultline model reduce`; reducer acceptance still preserves the original failure
signature. `check` does not relax that predicate or claim to extend PyTorch's minifier.

Keep completed artifacts. `--resume` verifies identity before reusing completed batches; use a new
output directory after changing the model, source, environment, settings, jobs, policy, or inputs.
Primary commits and diagnostic commits are separate: an interrupted diagnostic does not
rerun already committed primary workers. Auto budget and scheduling decisions are frozen
and verified on resume. A pre-0.18 execution directory is retained as
old evidence; its execution identity is incompatible with the new coordinator.
Resume requires completed preparation. If download or capture was interrupted before the
prepared bundle was committed, retain that attempt and choose a new output directory.
