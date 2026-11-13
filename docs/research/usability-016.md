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
capture are excluded from checks against prepared bundles. The earlier series shared the
host with calibration/test jobs. After a host OOM interruption, the final series runs one
Faultline measurement coordinator at a time. The host is otherwise uncontrolled; these
are operational costs, not isolated performance benchmarks.

The catalog retains all 20 original and nine held-out pretrained models. Ten frozen
batches per model use disjoint source-text chunks or disjoint vertical strips of two
photographs. Samples share model weights and source material; they are not an independent
population sample. No failed model or input is removed from the planned denominator.

The cost study uses pretrained float32 models. Each ResNet-50 batch contains two
224-by-224 RGB images. BERT, GPT-2 and SmolLM2 use two sequences of 64 tokens, returning
`last_hidden_state`; gradient checks use the frozen VJP vectors. Quick selects the first
three prepared batches, while standard and thorough use all ten. These sizes define the
scope of the reported costs.

The paired cost command is `faultline check PREPARED_MODEL_BUNDLE --baseline PYTHON
--candidate PYTHON --level LEVEL --output NEW_DIRECTORY`, with the same 2.15 CPU
interpreter selected for both sides. The two workers execute independently. The SmolLM2
cell omits `--baseline`. Each cell uses a new output directory and its own empty compiler
caches; frozen preparation is shared between levels.

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
Fresh controls ran after both the timing study and upgrade sweep finished. The earlier
plan to overlap small controls was superseded before any control workers started.
This resource-only scheduling change does not remove any model.

Text models and seven not-yet-started vision models write upgrade evidence to a
Windows-mounted volume to avoid filling the WSL disk. Fresh same-version outputs use
that volume too. This routing left existing evidence, including the interrupted ViT
attempt, in place. Input bundles and all preset-cost output directories stay on the Linux
filesystem. Storage placement does not change the frozen tensors or worker settings.
The Windows upgrade and fresh same-version output directories use transparent NTFS
compression. The latter inherits compression from the existing evidence storage pool;
this placement was set before those controls began. Tensor contents are unchanged, and
the preset-cost measurements remain on the Linux filesystem without NTFS compression.

After each completed catalog case, large captured tensor copies can be stored in a local
lossless archive to keep the remaining calibration within available disk space. Every
decompressed file is checked against its original SHA256, and the originals are checked
again before removing their extracted copies. The archive retains original paths and a
restore tool. Frozen input bundles, model weights, result JSON and commit records stay in
place; all result-JSON hashes must remain unchanged. Graph export or reduction of these
local validation artifacts requires restoring their tensor copies first. This is storage
management for the release study, not a change to normal `faultline check` output.

The archive helper passed twelve lifecycle checks, including corrupt archives, changed source
files, path traversal, conflicting restore destinations and exact round trips. A separate
real-data check reconstructed all 52 files from a 21.2-GiB completed tensor set; its archive
was 388 MiB. Archival runs after the preset timing study, with a separate memory guard,
and does not change the recorded cost measurements or numerical-worker settings.

Historical-issue fixtures reuse the preserved 0.6 evidence. Controlled injected faults
are new executions with explicit mutant labels. The exp_u20 control reuses the 20-batch
measurements from 0.15: that script recorded accuracy errors, not full primary oracle
observations. Its replicated rule result can be checked, but it cannot establish SAFE.
These cohorts must not be pooled into a false-positive rate.

Windows suspended the host during upgrade calibration on November 12, from about
09:54 to 17:47 UTC. The workers and committed evidence survived. The preset-cost study
had already completed; no reported preset timing spans this pause.

A subsequent WSL restart interrupted DistilBERT after 180 upgrade pairs had committed.
All 180 commit records passed hash verification before resuming the frozen campaign.
The supervisor now runs through a persistent background client with durable logs and
checks both the Linux boot ID and process start ticks. Lightweight monitoring reads only
the required summary metadata; it does not retain the large diagnostic report in memory.

## Completed calibration

All 29 same-version pairs and 29 upgrade pairs have ten committed input batches each.
The fixed policy was applied without retuning. Completed resource-limited or otherwise
inconclusive attempts remain in the denominator; completion of the campaign does not
mean every diagnostic completed or every verdict is SAFE.

| Cohort | Planned | SAFE | REGRESSION | NEEDS_REVIEW | INCONCLUSIVE |
|---|---:|---:|---:|---:|---:|
| same_version | 29 | 4 | 0 | 25 | 0 |
| upgrade | 29 | 4 | 0 | 25 | 0 |
| historical | 4 | 0 | 2 | 2 | 0 |
| controlled_injection | 3 | 0 | 3 | 0 | 0 |
| exp_u20 | 4 | 0 | 0 | 0 | 4 |

The same-version SAFE target was missed by 25 of 29 models. Each model's verdict is
listed below. The [JSON evidence](calibration-016.json) and [CSV table](calibration-016.csv)
retain the recorded reasons and batch-level values.
These misses must not be presented as a false-positive rate. Same-version workers can
agree with one another while both retain unexplained eager-versus-compiled findings.

Of the 25 same-version misses, 22 contain contrary numerical evidence and five contain
unexplained failures; two models belong to both groups. SmolLM2 also lacks replicated
accuracy coverage. The same-version cohort has no replicated accuracy flags. This
policy therefore still requires manual review for most of this catalog, even when the
environment is unchanged. The completed workflow is not a general automatic upgrade
approval gate.

Replicated accuracy flags in the upgrade cohort: none.
A REGRESSION verdict can also arise from a stable candidate-only primary failure;
the JSON distinguishes those reasons from replicated accuracy degradation.
With fewer than ten usable batch scores, a false replicated flag means insufficient
coverage to apply the rule, not evidence that an accuracy shift is absent.

| Model | Same version | Upgrade | Upgrade accuracy batches | Worse | Median growth | Replicated flag |
|---|---|---|---:|---:|---:|---|
| resnet18 | NEEDS_REVIEW | NEEDS_REVIEW | 10/10 | 0 | 1.000x | False |
| resnet34 | NEEDS_REVIEW | NEEDS_REVIEW | 10/10 | 0 | 1.000x | False |
| resnet50 | NEEDS_REVIEW | NEEDS_REVIEW | 10/10 | 0 | 1.000x | False |
| resnet101 | NEEDS_REVIEW | NEEDS_REVIEW | 10/10 | 0 | 1.000x | False |
| resnext50_32x4d | SAFE | SAFE | 10/10 | 0 | 1.000x | False |
| efficientnet_b0 | NEEDS_REVIEW | NEEDS_REVIEW | 10/10 | 2 | 1.000x | False |
| efficientnet_b1 | NEEDS_REVIEW | NEEDS_REVIEW | 10/10 | 2 | 1.000x | False |
| efficientnet_b2 | NEEDS_REVIEW | NEEDS_REVIEW | 10/10 | 3 | 1.000x | False |
| efficientnet_b3 | NEEDS_REVIEW | NEEDS_REVIEW | 10/10 | 3 | 1.000x | False |
| efficientnet_v2_s | SAFE | SAFE | 10/10 | 0 | 1.000x | False |
| mobilenet_v2 | NEEDS_REVIEW | NEEDS_REVIEW | 10/10 | 0 | 1.000x | False |
| mobilenet_v3_small | NEEDS_REVIEW | NEEDS_REVIEW | 10/10 | 0 | 1.000x | False |
| mobilenet_v3_large | NEEDS_REVIEW | NEEDS_REVIEW | 10/10 | 0 | 1.000x | False |
| vit_b_16 | NEEDS_REVIEW | NEEDS_REVIEW | 10/10 | 9 | 1.004x | False |
| swin_t | NEEDS_REVIEW | NEEDS_REVIEW | 10/10 | 4 | 1.000x | False |
| convnext_tiny | NEEDS_REVIEW | NEEDS_REVIEW | 10/10 | 5 | 1.000x | False |
| bert | NEEDS_REVIEW | NEEDS_REVIEW | 10/10 | 7 | 1.016x | False |
| gpt2 | NEEDS_REVIEW | NEEDS_REVIEW | 9/10 | 4 | 0.969x | False |
| distilbert | SAFE | SAFE | 10/10 | 7 | 1.011x | False |
| smollm2 | NEEDS_REVIEW | NEEDS_REVIEW | 0/10 | 0 | missing | False |
| regnet_y_400mf | NEEDS_REVIEW | NEEDS_REVIEW | 10/10 | 0 | 1.000x | False |
| densenet121 | NEEDS_REVIEW | NEEDS_REVIEW | 10/10 | 0 | 1.000x | False |
| shufflenet_v2_x1_0 | NEEDS_REVIEW | NEEDS_REVIEW | 10/10 | 0 | 1.000x | False |
| squeezenet1_1 | NEEDS_REVIEW | NEEDS_REVIEW | 10/10 | 0 | 1.000x | False |
| mnasnet1_0 | NEEDS_REVIEW | NEEDS_REVIEW | 10/10 | 0 | 1.000x | False |
| t5_small_encoder | NEEDS_REVIEW | NEEDS_REVIEW | 10/10 | 0 | 1.000x | False |
| roberta_base | SAFE | SAFE | 10/10 | 5 | 0.998x | False |
| opt_125m | NEEDS_REVIEW | NEEDS_REVIEW | 10/10 | 6 | 1.002x | False |
| pythia_160m | NEEDS_REVIEW | NEEDS_REVIEW | 10/10 | 6 | 1.011x | False |

All three explicitly injected faults produce REGRESSION. Of four retained historical
fixtures, two produce REGRESSION and two NEEDS_REVIEW. One missed fixture passes in both
retained environments but lacks replicated gradient-accuracy evidence; the other fails
in both. These are calibration misses, not policy changes or new discoveries.

The retained exp_u20 evidence triggers no replicated flag for any of its four variants.
Their full verdict is INCONCLUSIVE because the 0.15 script recorded accuracy errors but
not primary oracle observations. This is not a successful full-model SAFE check.

The three paired standard-cost cases and the other 26 retrospective full-diagnostic
controls use the distinct execution protocols described above. The upgrade cohort and
those 26 controls share preserved baseline observations. Input batches share their source
material. These results describe this catalog, these frozen inputs and these settings;
they are not independent estimates of deployment risk.

For a retained legacy worker that exceeded its resource budget during diagnostics, the
policy replay uses its already-saved primary checkpoint and records the diagnostic gap.
The raw resource-limit outcome is unchanged. SmolLM2 illustrates the distinction: all
twenty upgrade workers retained primary comparisons but reached the diagnostic memory
limit without retaining a completed fp64 report, leaving 0/10 paired accuracy scores.
The ten fresh same-version SmolLM2 workers also retained their primary comparisons
before reaching the diagnostic memory limit. These full-AOT cohorts are separate from
the new standard-preset SmolLM2 run that completed within 8,015.6 MiB.

## Measured costs

All ten timing cells completed their planned primary batches within the 12-GiB worker
budget. These wall times include interpreter startup and exclude frozen-input preparation.
The [machine-readable evidence](cost-016.json) records settings, workload shapes, source
and verdict hashes, diagnostic counts, and both internal and external wall clocks.

| Model | Level | Batches | Verdict | Wall minutes | Peak worker-tree RSS (MiB) |
|---|---|---:|---|---:|---:|
| ResNet-50 | quick | 3 | SAFE | 1.46 | 1,145.5 |
| ResNet-50 | standard | 10 | NEEDS_REVIEW | 15.00 | 1,945.2 |
| ResNet-50 | thorough | 10 | NEEDS_REVIEW | 27.42 | 3,149.4 |
| BERT | quick | 3 | NEEDS_REVIEW | 2.28 | 3,449.4 |
| BERT | standard | 10 | NEEDS_REVIEW | 33.67 | 6,743.6 |
| BERT | thorough | 10 | NEEDS_REVIEW | 78.43 | 10,252.3 |
| GPT-2 | quick | 3 | SAFE | 2.51 | 3,881.5 |
| GPT-2 | standard | 10 | NEEDS_REVIEW | 36.44 | 7,428.1 |
| GPT-2 | thorough | 10 | NEEDS_REVIEW | 85.86 | 11,466.1 |
| SmolLM2-135M | standard | 10 | NEEDS_REVIEW | 17.86 | 8,015.6 |

The approximate 30-minute standard target is met for the measured ResNet-50 and SmolLM2
workloads. BERT and GPT-2 miss it. Thorough takes over an hour for both text models;
it is an investigation preset, not a short CI check. RSS is sampled worker-tree memory,
not the coordinator, other host processes, or total system memory.

These paired timings compare the same PyTorch version in independent workers. BERT quick
retains forward mismatches in both environments. Matching failures do not establish SAFE,
and contrary diagnostics remain NEEDS_REVIEW. The timings do not imply upgrade safety.

## Measurements before the performance changes

SmolLM2-135M completed all ten single-environment standard batches in 2,067 seconds
(34.5 minutes), at 8,027.7 MiB peak worker-tree RSS. Its verdict was NEEDS_REVIEW; contrary
evidence remains visible and the first unexplained workload is exported. This meets the
12-GiB worker-memory target, but does not establish a universal 30-minute bound.

The six completed cells below precede the exact-match and diagnostic-storage changes
described next. That timing series was stopped after two BERT thorough batches; all
48 committed batches and the incomplete artifacts were preserved. The final series reran
all ten cost cells from separate empty caches. ResNet-50 and BERT use separate baseline
and candidate workers in the same PyTorch version; SmolLM2 uses one environment.

| Model | Level | Batches | Verdict | Wall minutes | Peak worker-tree RSS (MiB) |
|---|---|---:|---|---:|---:|
| ResNet-50 | quick | 3 | SAFE | 2.09 | 1,271.4 |
| ResNet-50 | standard | 10 | NEEDS_REVIEW | 24.49 | 1,938.1 |
| ResNet-50 | thorough | 10 | NEEDS_REVIEW | 45.07 | 3,140.5 |
| BERT | quick | 3 | NEEDS_REVIEW | 4.73 | 4,526.2 |
| BERT | standard | 10 | NEEDS_REVIEW | 61.86 | 6,721.4 |
| SmolLM2-135M | standard | 10 | NEEDS_REVIEW | 34.45 | 8,027.7 |

RSS excludes the coordinator and other active jobs. BERT's quick result contains forward
mismatches in both environments; matching failures do not establish SAFE. The longer
presets are not represented as meeting a universal 30-minute target.

The replacement SmolLM2-135M standard check completed all ten batches in 1,071.87 seconds
from process launch to exit (17.86 minutes), with 8,015.6 MiB peak worker-tree RSS. The
CLI's internal command timer recorded 1,067.71 seconds. It returned NEEDS_REVIEW, retained
all contrary evidence and exported the first unexplained workload. All ten primary
comparisons, statuses, signatures, fp64 advisories and numerical contexts exactly match
the preoptimization check. This satisfies the tested 12-GiB memory and approximately
30-minute targets for this workload. This is not a universal latency bound or an isolated
optimization speedup estimate.

The replacement BERT standard check took 2,020.22 seconds (33.67 minutes), at 6,743.6 MiB
peak worker-tree RSS, and returned NEEDS_REVIEW. It still misses the approximate
30-minute goal. Its separate fp64 and eager-control workers repeat primary verification
before using their diagnostics. All primary comparisons and numerical diagnostics agree
with the earlier ten-batch BERT result.

## Performance changes and equivalence checks

Exact finite floating-point matches now return the same zero-error comparison without
allocating full-size error buffers or copying both tensors to CPU float64. Other cases
use the existing comparison. A differential check against the earlier implementation
matched every report field and exception in 2,238 cases, including strided tensors,
nonfinites, empty tensors, structural mismatches and invalid tolerances.

Compact fp64/control replays also stop writing redundant graph tensor dumps. The primary
worker still retains compiler inputs for export, and thorough AOT capture is unchanged.
These changes do not reuse primary observations, change tolerances or modify verdict rules.

Two fresh BERT workers using the same frozen second batch and copied warm compiler caches
verified the changes against the earlier artifacts:

| Check | Before seconds | After seconds | Evidence equality |
|---|---:|---:|---|
| Primary worker | 58.57 | 37.27 | Full comparison, status, signature, workload and settings |
| fp64 diagnostic | 69.10 | 43.63 | Full fp64 advisory, context and replay verification |

The diagnostic directory decreased from 439,112,174 to 654,167 bytes. Concurrent host
work differed between runs; these checks establish result parity and are not full-preset
speedup measurements. A separate one-million-element exact-match microbenchmark decreased
from 59.02 to 2.22 milliseconds median over seven calls, including the input clone.
The complete preset timing series uses the external process wall clock, including
interpreter startup; both time fields are retained.
The six cells with completed before/after measurements contain 82 worker observations.
Their statuses, workload hashes, signatures, full comparisons, fp64 advisories and numerical
contexts are identical. Timing and resource fields are measured separately.

The first optimized timing attempt was interrupted by a kernel global OOM while it
overlapped the upgrade sweep. Hash verification retained all 130 committed upgrade
pairs and nine completed SmolLM2 batches. The interrupted SmolLM2 attempt is not a
completed timing cell. The final ten-cell timing series started with empty caches and ran
sequentially; the original upgrade campaign then resumed with its original source,
settings and committed evidence. The same-version controls followed. No result is
reclassified as a model/compiler failure because its host environment was interrupted.

The changed implementation passed 366 Linux portable tests, 365 Windows tests with one
skip, all 12 CPU Inductor tests, and all 14 compiler-validation checks. Six new behavior
tests cover exact-match handling and diagnostic replay without duplicate tensor dumps.

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

The [model-upgrade workflow](https://github.com/tepiduck/ipsum/actions/runs/43132883347)
at `2f7874e` created CPU environments with 2.15.0+cpu and 2.16.0.dev20261111+cpu and
returned SAFE for all ten standard batches of the small example. This is a real workflow
execution, not just a YAML syntax check; the gate fails only on REGRESSION and uploads
the report for other verdicts. Its current-nightly date differs from the fixed November 5
wheel used for catalog calibration. The example tests integration, not pretrained-model
accuracy or general upgrade safety.
All six jobs passed, including the [validation workflow](https://github.com/tepiduck/ipsum/actions/runs/43132883301).
The earlier six-job run at `4021008` also passed and remains in the historical CI record.
