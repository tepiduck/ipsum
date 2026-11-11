# 0.16 usability validation

The release adds an advisory `check` workflow over the existing primary observations.
The [verdict policy](../verdict-policy.md) and [batch rule](../v016-policy.json) were fixed
before calibration. No numerical tolerances, failure signatures, reducer predicates,
or existing command exit codes were changed.

## Measurement protocol

The CPU measurements use Ubuntu under WSL2 on an AMD Ryzen 7 7700X (8 cores, 16 threads),
with 15.5 GiB available to WSL and 4 GiB swap. Each model worker uses one CPU thread,
a 600-second deadline, and a 12-GiB sampled process-tree RSS budget. Separate compiler
caches for baseline and candidate start empty and warm between batches. Downloads and
capture are excluded from checks against prepared bundles. Other calibration/test work
shares the host; these are operational costs, not isolated performance benchmarks.

The catalog retains all 20 original and nine held-out pretrained models. Ten frozen
batches per model use disjoint source-text chunks or disjoint vertical strips of two
photographs. Samples share model weights and source material; they are not an independent
population sample. No failed model or input is removed from the planned denominator.

Same-version calibration executes separate 2.15.0+cpu baseline and candidate workers.
The upgrade cohort uses 2.15.0+cpu and 2.16.0.dev20261105+cpu. Its recorded full-diagnostic
sweep is summarized retrospectively under the new policy; those workers use two repeats
and AOT capture, rather than the new standard preset's lazy diagnostic scheduling.
Cost-study standard pairs can also supply the matching same-version calibration cell.

For the other same-version cells, the preserved 2.15 upgrade baseline is compared with a
fresh independent 2.15 worker using the same full-diagnostic source, settings, interpreter
and frozen workload. Source/environment equality and workload/commit hashes are checked
before execution. This avoids rerunning an existing baseline; it is not a comparison of
an observation with itself. The two cohorts share those baseline measurements and their
counts are therefore correlated. This scheduling change was declared before starting
the fresh same-version workers and does not change the verdict or replication policy.

Large text-model upgrade outputs and fresh same-version outputs are allocated on a
Windows-mounted evidence volume to avoid filling the WSL disk. Completed evidence was
not moved or deleted. Input bundles and all preset-cost output directories stay on the
Linux filesystem. Storage placement does not change the frozen tensors or worker settings.

Historical-issue fixtures reuse the preserved 0.6 evidence. Controlled injected faults
are new executions with explicit mutant labels. The exp_u20 control reuses the 20-batch
measurements from 0.15: that script recorded accuracy errors, not full primary oracle
observations. Its replicated rule result can be checked, but it cannot establish SAFE.
These cohorts must not be pooled into a false-positive rate.

## Calibration status

The full catalog and preset measurements are in progress. This document is a development
record until the final tables are present; no completed 29-model calibration is claimed.

Completed controls: all three injected faults produce REGRESSION. Of four retained
historical fixtures, two produce REGRESSION and two NEEDS_REVIEW. One missed historical
fixture passes in both retained environments but lacks replicated gradient-accuracy
evidence; the other fails in both. The policy is not retuned to turn these into hits.

The retained exp_u20 data do not trigger the replicated rule for any of its four model
variants. The full verdict for this accuracy-only replay is INCONCLUSIVE because primary
observations are absent. This is distinct from a successful full-model SAFE check.

## Completed memory check

SmolLM2-135M completed all ten single-environment standard batches in 2,067 seconds
(34.5 minutes), at 8,027.7 MiB peak worker-tree RSS. Its verdict was NEEDS_REVIEW; contrary
evidence remains visible and the first unexplained workload is exported. This meets the
12-GiB worker-memory target, but does not establish a universal 30-minute bound.

## Installed-package isolation

The fresh-wheel test exposed a worker import-path bug: prepending the coordinator's
package parent also prepended its entire `site-packages` directory. A candidate requested
as 2.16 could then import the coordinator's 2.15. Workers now load only the Faultline
package explicitly; PyTorch and model dependencies resolve in the selected interpreter.
A rebuilt-wheel check records 2.15.0+cpu in baseline workers and 2.16.0.dev20261105+cpu
in candidate workers. Calibration uses isolated source directories and was not affected.

The [fresh-user transcript](quickstart-test.md) records install and quickstart timing.
Earlier investigation evidence and license metadata are unchanged. No upstream report
was filed and no numerical finding is presented as a confirmed compiler defect.

## CI execution

The [model-upgrade workflow](https://github.com/tepiduck/ipsum/actions/runs/43118691494)
at `4021008` created CPU environments with 2.15.0+cpu and 2.16.0.dev20261111+cpu and
returned SAFE for all ten standard batches of the small example. This is a real workflow
execution, not just a YAML syntax check; the gate fails only on REGRESSION and uploads
the report for other verdicts. Its current-nightly date differs from the fixed November 5
wheel used for catalog calibration. The example tests integration, not pretrained-model
accuracy or general upgrade safety.
