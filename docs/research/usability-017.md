# 0.17 upgrade-verdict validation

The paired verdict now asks whether the candidate environment made the tested workload
worse. This is a deliberate change from v1's compile/eager fidelity-oriented summary,
not a tolerance adjustment. [Policy v2](../verdict-policy-v2.md) was committed as
`aee7d34` before re-scoring or fresh measurements. Single-environment checks and explicit
`--verdict-policy v1` retain v1. Primary statuses, signatures, reducer predicates,
tolerances and exit codes are unchanged.

## Preserved 0.16 evidence: post hoc policy replay

Each cell below lists SAFE / REGRESSION / NEEDS_REVIEW / INCONCLUSIVE. The v2 column is
**post hoc**. No model was rerun for this table, and no early stopping was simulated.
Every retained commit was hash-verified. Recomputed v1 verdicts, reasons, primary counts,
batch-accuracy distributions and evidence hashes match the published 0.16 records.

| Cohort | Planned | v1 | v2 (post hoc) |
|---|---:|---|---|
| Same version | 29 | 4 / 0 / 25 / 0 | 28 / 0 / 0 / 1 |
| 2.15 → retained 2.16 nightly | 29 | 4 / 0 / 25 / 0 | 27 / 0 / 0 / 2 |
| Historical fixtures | 4 | 0 / 2 / 2 / 0 | 1 / 2 / 0 / 1 |
| Controlled injections | 3 | 0 / 3 / 0 / 0 | 0 / 3 / 0 / 0 |
| exp_u20 accuracy-only evidence | 4 | 0 / 0 / 0 / 4 | 0 / 0 / 0 / 4 |

The remaining same-version INCONCLUSIVE is SmolLM2, with no complete paired fp64 scores.
Upgrade GPT-2 has nine of ten scores and SmolLM2 has none. Required missing accuracy
evidence prevents SAFE; none of the measured distributions flags replicated degradation.
The original 0.16 counts and files remain unchanged. These are model/cohort counts,
not a false-positive rate or independent statistical trials.

Both baseline-confirmed historical regressions, #183986 and #182012, retain REGRESSION.
The other fixtures were not regressions under v1: #180164 has passing primaries but only
one accuracy batch and becomes INCONCLUSIVE; #185337 has a shared forward discrepancy
and becomes quick SAFE for change attribution. That SAFE does not assert eager fidelity.
The exp_u20 records lack primary observations and remain INCONCLUSIVE; their batch-level
accuracy rule does not flag. No new historical or exp_u20 model execution is claimed.

[Per-pair JSON](rescore-017.json) and [CSV](rescore-017.csv) retain all 69 entries.
`scripts/rescore_check_v2.py` reproduces the policy-only replay from local evidence.

## Fresh checks

Six checkpoints absent from the 0.16 catalog were fixed before measurement: RegNet-X-400MF,
ShuffleNet-V2-x0.5, SqueezeNet-1.0, a two-layer BERT checkpoint, DistilRoBERTa, and Pythia-70M.
They are new checkpoint variants, not six unseen architecture families. Exact weights,
Hub revisions and frozen inputs are recorded in the machine-readable validation record.
The standard protocol uses ten real-input batches, two primary repeats, one CPU thread,
one job and a 12-GiB per-job budget, comparing 2.15.0+cpu against itself and against
2.16.0.dev20261105+cpu. Input sources and weights induce dependence between batches;
the descriptive batch sign-test does not establish independent sampling.

All planned primary pairs execute. Ordered accuracy diagnostics can stop once every
possible completion has the same replicated-rule result. A negative decision before ten
scores is not reported as ten measurements. Positive flags still require at least ten.
| Cohort | Planned | SAFE / REGRESSION / NEEDS_REVIEW / INCONCLUSIVE |
|---|---:|---|
| same_version | 6 | 6 / 0 / 0 / 0 |
| upgrade | 6 | 6 / 0 / 0 / 0 |

| Model | Cohort | Verdict | Conclusive primary batches | Accuracy scores / planned | Deliberately skipped | Observed median growth |
|---|---|---|---:|---:|---:|---:|
| regnet_x_400mf | same_version | SAFE | 10/10 | 3/10 | 7 | 1.000x |
| regnet_x_400mf | upgrade | SAFE | 10/10 | 3/10 | 7 | 1.000x |
| shufflenet_v2_x0_5 | same_version | SAFE | 10/10 | 3/10 | 7 | 1.000x |
| shufflenet_v2_x0_5 | upgrade | SAFE | 10/10 | 3/10 | 7 | 1.000x |
| squeezenet1_0 | same_version | SAFE | 10/10 | 3/10 | 7 | 1.000x |
| squeezenet1_0 | upgrade | SAFE | 10/10 | 3/10 | 7 | 1.000x |
| bert_tiny | same_version | SAFE | 10/10 | 3/10 | 7 | 1.000x |
| bert_tiny | upgrade | SAFE | 10/10 | 4/10 | 6 | 0.974x |
| distilroberta | same_version | SAFE | 10/10 | 3/10 | 7 | 1.000x |
| distilroberta | upgrade | SAFE | 10/10 | 5/10 | 5 | 1.000x |
| pythia70m | same_version | SAFE | 10/10 | 3/10 | 7 | 1.000x |
| pythia70m | upgrade | SAFE | 10/10 | 4/10 | 6 | 0.933x |

Replicated accuracy flags: none.
Observed medians describe the measured prefix only; deliberately skipped batches have no invented score. All six models remain in both denominators.
The [fresh validation JSON](validation-017.json) retains every measured batch score, stopping certificate, model identity, primary count and evidence hash.

### Standard-level wall time

| Model | 0.16: minutes / verdict | 0.17 jobs=4: minutes / verdict | Primary coverage | Accuracy complete / planned | Complete-comparison wall ratio |
|---|---|---|---:|---:|---:|
| resnet50 | 14.97 / NEEDS_REVIEW | 4.63 / SAFE | 10/10 | 3/10 | 3.23x |
| bert | 33.61 / NEEDS_REVIEW | 1.54 / INCONCLUSIVE | 0/10 | 0/10 | not comparable: incomplete |
| gpt2 | 36.36 / NEEDS_REVIEW | 1.66 / INCONCLUSIVE | 0/10 | 0/10 | not comparable: incomplete |

The ratio compares elapsed commands with different diagnostic work and concurrency; it does not isolate a parallelism speedup. An incomplete run is never reported as a successful speedup.
The final timing queue ran after the fresh cohort and developer test processes had completed. Other host applications were uncontrolled. No new GPU execution is claimed.

## Execution changes and validation method

Each execution slot exclusively owns its baseline and candidate compiler caches. Batches
assigned to a slot run sequentially in fresh supervised workers; slots run concurrently.
Diagnostics consume ascending frozen batch order, independent of worker completion order.
The total RSS budget is split across effective jobs. Serial/parallel parity is tested with
controlled scheduling and actual eager and CPU Inductor workers. Memory-limited execution
remains distinct and cannot be described as a successful speedup.

An initial fresh attempt used cold caches per batch. It was retained and interrupted
after observing unnecessary repeated compilation, before a completed fresh verdict.
The final series uses new directories and the same already-frozen inputs with exclusive
reusable slot caches. This changes execution cost, not the policy or numerical thresholds.
Prepared checkpoint factories retain their original source bytes.

The timing series uses the original 0.16 frozen ResNet-50, BERT and GPT-2 collections,
paired 2.15 interpreters, standard level and four jobs under a total 12-GiB budget
(3072 MiB per worker). Each cell starts with empty slot caches. The previous series used
one worker at a time with 12 GiB each. Report coverage alongside elapsed time: a smaller
budget that cannot execute the workload is not an optimization result. These are CPU
operational costs on the same WSL host, not isolated performance benchmarks.

## Executed developer checks

| Check | Result |
|---|---|
| Linux portable suite | 381 passed; 13 compiler tests deselected |
| Windows portable suite | 380 passed, one skipped; 13 compiler tests deselected |
| Actual CPU Inductor suite | 13 passed, including serial/parallel primary-result parity |
| Compiler validation campaign | All 14 checks passed; 17 generated/corpus passes and one unconfirmed numerical discrepancy |
| Legacy suite (`asyncio_mode=auto`) | 31 passed |
| Ruff lint and requested formatting | Passed |
| Historical preservation | 37 prior data files, all 24 prior validation entries, and prior RESULTS content verified |
| Installed 0.17 wheel | Paired standard v2 and explicit quick v1 SAFE; both JSON schemas verified |

Fresh reruns of the three explicitly injected controls returned REGRESSION and retained
their injected-fault labels. Policy tests cover changed shared signatures, incomplete or
unstable change evidence, conservative stopping bounds, missing primary execution and a
candidate-only failure on the last batch after accuracy stopping. Lifecycle tests cover
separate primary checkpoints, diagnostic interruption/resume, settings incompatibility,
memory splitting, worker environment isolation and serial/parallel decision parity.

The installed-wheel standard check used ten frozen small-model batches, two jobs and
independent 2.15.0+cpu / 2.16.0.dev20261105+cpu interpreters. It completed in 99.56 seconds;
this is a packaging/workflow check, not a pretrained-model timing claim. The tested
checkout/frozen-source hash is `43fc70267291fee131123c50b2135f0cdd1426c1ef66f64524b509918c7956d8`.
The installed Git-normalized package hash is
`72e84c2e13648d1bad721c3a89c7091995012a6d9a0c5b829566c910dbd914df`;
the difference is line endings in five existing files. Normalized source bytes match.

Nothing is filed upstream. Numerical findings remain investigation candidates.
