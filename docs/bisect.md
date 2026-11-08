# CPU nightly accuracy bisection

`faultline bisect` compares a captured workload with CPU TorchInductor across dated PyTorch
nightly wheels. It measures compiled error against each build's repeated eager fp64 reference.
It does not bisect the primary pass/fail verdict, change tolerances, or file upstream issues.

```bash
faultline bisect workloads/gpt2 --baseline-python .stable/bin/python --start 2026-10-05 --end 2026-11-05 --requirements model-dependencies.txt --output faultline-runs/gpt2-bisect
# After an interruption, repeat that exact command with --resume.
```

Use a Python version and platform with compatible CPU nightly wheels. The coordinator selects
the wheel matching its own interpreter and platform; the baseline interpreter must support the
same workload. CPU Inductor still needs a C++ toolchain and Python headers. There is no GPU or
eager fallback. The workflow runs trusted captured model code, as other native commands do.
The interpreter's `venv`/`ensurepip` support is also required; on Debian/Ubuntu this is provided
by the matching `python3-venv` package. Installation failures are recorded separately from accuracy.

`model-dependencies.txt` accepts exact `name==version` requirements, without URLs, environment
markers, or pip options. Pin the model libraries as well as their dependencies to reduce
confounding. Torch, torchvision, torchaudio, and Faultline cannot be pinned through this file:
the workflow manages torch and supplies the same Faultline source to workers. Models importing
torchvision are currently outside this CPU-only installer workflow because a matching torchvision
nightly is not installed. Without the file, only the coordinator's numpy and psutil versions
are pinned. Every environment records the full resolved package set, torch build information,
and `torch.version.git_version`; inspect dependency changes before interpreting a boundary.

The default predicate is fixed before searching:

- Run the stable baseline in three fresh workers. Freeze the per-target median compiled/fp64
  RMSE for gradient targets with finite, strictly positive errors in every baseline observation.
- Run each selected nightly in three fresh workers. Each worker reports the median log2 error
  growth over those same targets. A missing or zero candidate error is inconclusive under this
  strictly positive-error predicate; it is not silently dropped.
- All three scores at least `log2(1.4)` means `shifted`; all below means `not_shifted`.
  Mixed observations, failed execution, or missing references mean `inconclusive`.
- Binary search requires an observed non-shifted first build and shifted last build. Inconclusive
  intermediate builds do not close the interval. Independently rerun both boundary builds in
  fresh workers before reporting a localized interval.

`--growth` and `--observations` define a new experiment. They cannot change when resuming.
The default worker budget is 600 seconds and 12 GiB host RSS; `--timeout`, `--max-rss-mb`, and
`--install-timeout` make these budgets explicit. Installer commands are separately supervised,
with a 4 GiB process-tree RSS budget. The watchdog stops only owned processes and descendants;
it is not a security sandbox.

Each nightly has a separate virtual environment and download/install/probe logs. Wheel downloads
come from the official CPU index. When the index supplies a wheel SHA-256, pip verifies it;
otherwise the downloaded wheel's SHA-256 is recorded locally, without claiming publisher-supplied
hash verification. Completed environment identities and downloaded wheel bytes are rechecked on
reuse. Completed results are committed with hashes. Resume checks the workload, harness source,
settings, baseline environment, dependency lock, and completed evidence. Interrupted attempts
remain on disk. After fixing an installation prerequisite or network problem, use
`--resume --retry-installations` to retry failed installs in new attempt directories without
rerunning completed model observations. Measured inconclusive trials remain evidence; use a new
campaign to investigate them under a changed setup.

`summary.json` and `report.html` retain all tested dates, unavailable dates, worker observations,
primary statuses, and independent confirmations. The boundary includes the observed last-good
and first-bad nightly dates **under this accuracy predicate**, plus an upstream commit comparison
link. PyTorch may put a nightly release snapshot SHA in `torch.version.git_version`; the release
commit message then identifies the underlying main-branch source SHA. Record both when inspecting
individual upstream changes, since two release snapshots can encompass many source commits.
Binary search assumes a monotone shift over available builds; it does not prove there were
no earlier temporary shifts or localize a single causal commit. Missing calendar dates remain
visible even when the neighboring available builds are localized.

This new command returns 0 for an independently confirmed boundary and 2 for an inconclusive
search. An accuracy shift does not return a defect verdict. Existing commands keep their exit
codes. An upstream change may intentionally trade accuracy for another property; inspect the
code and absolute errors before attributing the observed shift.

## Recorded GPT-2 workload

The 0.11 experiment used the built-in pretrained GPT-2 factory, a batch of two real 64-token sequences,
frozen weights, and fixed backward probes. To prepare the same kind of workload:

```bash
faultline sweep prepare --model gpt2 --output workloads/gpt2
faultline bisect workloads/gpt2/models/gpt2/bundle --baseline-python .stable/bin/python --start 2026-10-05 --end 2026-11-05 --requirements model-dependencies.txt --growth 1.4 --observations 3 --timeout 900 --max-rss-mb 12288 --output faultline-runs/gpt2-bisect
```

The dependency pins and exact workload/checkpoint identities are recorded in
`accuracy_bisection_validation` in [validation.json](validation.json). Check them before treating
a new preparation as an exact reproduction: preparation resolves a checkpoint revision at that
time, while replay and bisection use only the frozen local workload. Large weights and environments
are not included in the repository. [RESULTS](../RESULTS.md#version-011-cpu-nightly-bisection)
records the smaller-workload attempts, independently confirmed date boundary, and the negative
broadcast-heuristic control.
