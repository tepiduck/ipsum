# 0.20 nightly watch and compilation structure

This release adds compilation telemetry and a scheduled nightly workflow. Verdict
policy v2, primary comparison rules, tolerances, failure signatures, reducer predicates
and existing command exit codes are unchanged. Compilation advisories are notes, not
regression decisions. Prior evidence has not been re-scored.

## Seed protocol

The CPU seed selects the seven available nightly wheels dated November 8–14, 2026,
with November 7 as the preceding anchor. For each nightly, five pretrained frozen
workloads are compared against both PyTorch 2.15.0 and the previous available nightly:
ResNet-50, MobileNetV3-Small, BERT, GPT-2 and ViT-B/16. The planned denominator is
70 standard-level checks, each with ten primary input-batch pairs.

The input collections are the preserved 0.16 collections. Every comparison uses
standard level, fixed VJP probes, two primary repeats, one tensor thread, `--jobs auto`,
an explicit 11,000-MiB total worker budget and the existing accuracy early-stopping
rule. A positive replicated accuracy flag requires ten completed batch scores;
fewer scores can complete a negative decision only when the unchanged rule proves
that remaining batches cannot change it. All planned primary executions still run.

The installed model dependencies are pinned to the seed's stable environment.
Transformers is 5.23.0; torchvision uses compatible wheels from the respective
PyTorch index. These are comparisons of environments, including torchvision.
They are not an assertion that only the torch package changed, nor an exact
repetition of the 0.19 environment. Versions, upstream commits, frozen-bundle
hashes, per-pair coverage and measurements are retained in the portable JSON.

The compilation policy was saved before execution. New graph-break reasons and
increased recompilation counts require complete paired observations. A compile-time
note requires both at least 1.5× and at least ten seconds more per batch, averaged
over all planned paired primary workers with complete timing coverage. See the
[policy](../compilation-policy-v1.json) and [collection limits](../compilation-structure.md).

## Seven-nightly results

All 70 planned comparisons completed with a v2 SAFE verdict. Every model completed
all 140 of its planned primary batch pairs across the fourteen comparisons, giving
700/700 primary pairs overall. The recorded accuracy diagnostics completed
273 paired batch scores; the unchanged early-stopping rule accounted
for the remaining scores. No replicated accuracy shift was flagged.

| Model | Stable comparison | Previous-nightly comparison |
|---|---|---|
| resnet50 | 7 SAFE | 7 SAFE |
| mobilenet_v3_small | 7 SAFE | 7 SAFE |
| bert | 7 SAFE | 7 SAFE |
| gpt2 | 7 SAFE | 7 SAFE |
| vit_b_16 | 7 SAFE | 7 SAFE |

Compilation timing coverage is complete in all 70 checks. There are no
`new_graph_breaks`, `recompile_increase` or `compile_time_increase` notes in this seed.
The deliberately changed adapter controls below establish that the first two labels
are observable. Unit tests exercise the exact compile-time threshold boundaries.

No real nightly REGRESSION occurred, so this seed did not trigger automatic
bisection. Dispatch, replication and boundary handling were checked with controlled
lifecycle tests; this release does not claim to have localized a new nightly regression.
SAFE applies only to these frozen inputs and the existing v2 policy, not general
model correctness or a measured false-positive rate.

The observed command costs across the fourteen checks per model were:

| Model | Checks | Median minutes | Range, minutes | Auto jobs |
|---|---:|---:|---:|---|
| resnet50 | 14 | 6.97 | 6.66–7.28 | 4 |
| mobilenet_v3_small | 14 | 6.19 | 6.00–6.60 | 4 |
| bert | 14 | 14.24 | 10.99–20.22 | 1, 2 |
| gpt2 | 14 | 14.97 | 12.55–16.60 | 1, 2 |
| vit_b_16 | 14 | 12.90 | 10.54–15.34 | 2, 3 |

The sum of check command times is 12.83 hours. These costs
include primary runs and required diagnostics, but exclude environment installation,
input preparation and the local artifact hardlink pass. Host load and cache state
were uncontrolled. They are operational costs, not isolated performance benchmarks.
Per-pair values, exact wheel versions, compilation counts and accuracy coverage are in
[the portable seed evidence](watch-020.json).

An installed 0.20 wheel resumed the completed history with exit code zero, reused all
70 checks and preserved the hashes of all committed JSON/HTML files. It launched no
new check supervisors. The installed runtime hash matches the source used for the seed.

## Adapter controls

`scripts/validate_compilation.py` executed real CPU Inductor forward and backward
passes on 2.15.0+cpu and 2.16.0.dev20261113+cpu. A controlled candidate adds an explicit
graph break and changes dynamic-shape compilation to shape specialization. Both
environments report the following structure while preserving identical outputs
and input gradients:

| Execution | Frame attempts | Graphs | Break events | Recompiles |
|---|---:|---:|---:|---:|
| Dynamic baseline | 1 | 1 | 0 | 0 |
| Deliberately changed candidate | 4 | 4 | 2 | 2 |

The adapter records the inserted break's source location and the size-guard failure
for the two-to-four-row transition. Both controls produce `new_graph_breaks` and
`recompile_increase`. These are injected structural changes, not nightly findings.
Backward compilation events are present. Cache-dependent timings from these controls
are not used to make a performance claim.

## Reproducibility and limits

The watch freezes its selected dates before executing checks and verifies source,
settings, input and environment identities on resume. Committed rows hash their
JSON and HTML evidence. Full local reports and tensor artifacts are preserved;
large checkpoints, environments and compiler caches are excluded from Git.
The local seed uses NTFS compression and byte-verified hardlinks for identical
new tensor artifacts, with compiler caches on the Linux filesystem.
The hardlinks were created by the local measurement harness. The watch CLI
retains complete worker artifacts and does not prune or deduplicate histories.

Compile wall time comes from the union of PyTorch's reported forward/backward
compilation intervals. Host load, cache state and worker scheduling affect it.
Separate fresh batch workers do not measure recompilation across a persistent
serving session. Dynamic requests within one workload can measure such recompiles.
Unavailable telemetry remains a coverage gap rather than a zero count.

The CPU scheduled workflow and local WSL/AMD scheduling examples are documented
in [watch setup](../watch.md). No local timer is installed. This release's new
measurements cover CPU only; they do not establish AMD or NVIDIA execution results,
a false-positive rate, or a compiler defect. No upstream report is filed.

## Executed checks

The Linux portable suite passed 418 tests, with 14 compiler tests deselected.
The separate CPU Inductor suite passed all 14 tests. Windows passed 417 portable
tests with one platform skip. An export-artifact test passed in a separate focused invocation
on both platforms, giving 419 Linux and 418 Windows passing portable
cases across these invocations. It exports all ten real frozen batches, checks
their hashes, runs the exported driver's validation, and verifies rejection of
a deliberately modified snapshot.

A fresh default-catalog preparation smoke captured pretrained MobileNetV3-Small
into ten verified batches in 37.20 seconds. A second preparation call reused the
same capture and manifest hash without creating another attempt. This validates
preparation and reuse separately from the seed's preserved input collections.

All 31 legacy tests and all 14 compiler validation-campaign checks passed. The
legacy suite requires its existing asynchronous fixtures to run with
`-o asyncio_mode=auto`; an earlier invocation without that setting failed its
async tests, and its log is preserved. Ruff lint and required formatting passed.

Lifecycle unit tests with simulated check outcomes cover interrupted-plan resume, committed-evidence
tampering, stable and previous-nightly tracks, automatic bisection dispatch,
independent boundary confirmation, inconclusive intermediate builds, selected-plan
exit codes and HTML escaping. Compilation tests cover exact advisory thresholds,
unavailable adapters, exception propagation, overlapping compile intervals and
unchanged primary fingerprints. These controls are distinct from the nightly seed.
