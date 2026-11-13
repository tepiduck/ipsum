# Upgrade verdict policy, version 2

Declared before 0.17 re-scoring or new measurements. This is a deliberate change in
the question answered by two-environment `check`: did the candidate environment make
the tested workload worse? Version 1 instead required compile/eager discrepancies
to be explained even when they already occurred in the baseline. The 0.16 policy,
evidence, verdict counts and numerical thresholds remain unchanged and available.

`check` selects v2 by default for two environments. `--verdict-policy v1` selects
the original policy. Single-environment checks always use v1, including when v2 is
requested: one environment cannot establish an upgrade regression. Primary statuses,
tolerances, signatures, reducer acceptance and command exit codes are unchanged.
SAFE is limited to the tested inputs and settings, and is not a claim of absolute
compile/eager fidelity, model quality, or a confirmed absence of compiler defects.

## Decisions

Precedence is REGRESSION, INCONCLUSIVE, NEEDS_REVIEW, SAFE.

1. **REGRESSION / 1:** the existing passing-baseline, stable reducible candidate-only
   failure rule, excluding a completely explained gradient-only mismatch, or the
   unchanged `accuracy_shift_replicated` rule. A failure reproducible on one frozen
   batch suffices; passing different inputs does not make that failure unstable.
   These remain regression candidates, not compiler-defect claims.
2. **INCONCLUSIVE / 3:** missing or inconclusive required primary execution, including
   reference errors, unavailable devices, crashes, timeouts and resource limits.
   Preserve these distinctions. The specific unstable numerical candidate case below
   is NEEDS_REVIEW. Required accuracy evidence that is unavailable without an observed
   worsening trend is also INCONCLUSIVE: absence of evidence cannot establish SAFE.
3. **NEEDS_REVIEW / 2:** candidate-only unstable numerical/output evidence with a
   passing, stable baseline, or incomplete required replication trending worse.
   The unstable case requires recorded cold/repeated numerical comparisons; a generic
   unstable result or an unstable baseline remains INCONCLUSIVE. A worsening trend
   means at least one valid batch score and median log2 growth greater than zero;
   this is a review reason, not an alternative regression threshold.
4. **SAFE / 0:** every planned primary comparison is conclusive, no candidate-only
   stable failure or replicated accuracy flag exists, and the required accuracy
   decision is complete. Forward-only quick checks do not request gradient accuracy.
   A required decision is complete after all planned scores (at least ten) succeed,
   or after a validated mathematical stopping certificate rules the flag out.

Conclusive discrepancies present in both environments, including changed failure
signatures, are nonblocking **pre-existing compile/eager differences, not attributable
to this change**. Preserve both observations, signatures, target counts, diagnostics
and evidence links. Report baseline-only discrepancies separately as absent in the
candidate. Compile/eager contrary evidence alone does not show cross-environment
worsening. A changed shared signature does not independently block SAFE; the replicated
accuracy rule determines an accuracy regression. Both-environment compile/runtime
failures remain visible even in a quick SAFE verdict: SAFE assesses the change, not
whether either environment can successfully execute the model.

## Replication and stopping

The 0.16 rule is unchanged: at least ten completed input-batch scores, at least 80%
worsening, and median batch growth at least 1.25x. Batch scores and zero-error handling
remain those in [v016-policy.json](v016-policy.json). Target-level sign tests remain
descriptive. No positive flag can be produced from fewer than ten scores.

Consume accuracy scores in ascending frozen batch order. For every possible number
of additional completed scores, bound the final fraction and median using maximally
worsening or maximally improving scores. Stop accuracy evaluation as soon as every
possible completion gives the same flag outcome. A negative certificate can occur
before ten scores: three non-worsening scores out of ten planned batches already
make 80% worsening impossible. A positive certificate still needs ten measured scores.
Do not certify incomplete/failed observed diagnostics as intentional early stopping.
Record planned, attempted, completed and deliberately skipped accuracy batches, the
ordered observed prefix, decision bounds, and the stop reason. Missing measured scores
remain missing; a negative certificate is not ten successful fp64 measurements.

Stopping accuracy work does not stop required primary checks. All planned primaries
must run before SAFE, because a later batch can contain a new candidate-only failure.
Candidate-only gradient discrepancies still receive the explanation checks needed by
the existing failure rule. `--no-early-stop` evaluates all requested diagnostics.

## Parallel execution and memory

`--jobs N` runs independent, single-threaded primary batch workers concurrently;
the default is `max(1, min(4, available CPU cores // 2))`, capped by planned batches.
Accuracy evaluation consumes the ordered primary results sequentially so serial and
parallel runs use the same stopping prefix. Each side runs in a separate supervised
process. Baseline and candidate within a batch run sequentially. Compiler caches and
worker environment settings must not race through shared process environment mutation.

The configured RSS budget is a total worker budget divided equally across effective
jobs, including diagnostics. Host/coordinator memory and sampling overshoot are not
covered by that bound. Do not silently enlarge budgets or fall back to fewer jobs.
Resource-limited runs remain visible; numerical/verdict equivalence to serial execution
is conditional on successful executions within both budgets. Runtime, RSS and scheduling
metadata are expected to differ. Preserve completed batch commits on interruption;
resume checks policy, jobs, budgets, inputs, source and environment identities.

## Validation fixed before re-scoring

- Re-score all 29 same-version and 29 upgrade pairs from retained 0.16 evidence, plus
  the retained historical, explicitly injected and exp_u20 cohorts. Publish v1 and v2
  side by side, label v2 post hoc, and preserve the original files byte-for-byte.
  The two baseline-confirmed historical regressions and all three injected faults must
  retain REGRESSION. The other two historical fixtures did not meet that target under
  v1; report their v2 outcomes honestly rather than overriding the policy. Accuracy-only
  exp_u20 evidence must not become REGRESSION or SAFE without primary evidence.
- Fresh validation uses six checkpoints absent from the 0.16 catalog:
  torchvision `regnet_x_400mf`, `shufflenet_v2_x0_5`, `squeezenet1_0`; Hugging Face
  `google/bert_uncased_L-2_H-128_A-2`, `distilbert/distilroberta-base`, and
  `EleutherAI/pythia-70m`. These are new checkpoint/model variants, not six unseen
  architecture families. Resolve and record exact weight revisions before execution.
  Use ten frozen real-input batches, standard level, one job and a 12-GiB budget for
  same-version 2.15 and 2.15 versus the retained 2.16.0.dev20261105 CPU environment.
  No fresh outcomes are used to design or change v2. Keep failures in the denominator.
- Measure ResNet-50, BERT and GPT-2 on the same frozen 0.16 bundles and paired 2.15
  interpreters, standard level, early stopping enabled and four jobs, with a 12-GiB
  total worker budget. Compare against the preserved 0.16 costs, including coverage
  and verdict differences. A faster incomplete run is not a successful speedup.
  Use fresh cache/output directories. Report host contention and memory limitations.
- Add behavior tests for policy differences, mixed signatures, missing/unstable
  evidence, early-stop bounds, a late candidate-only failure, serial/parallel parity,
  memory splitting, environment isolation, interruption/resume and both schemas.
  Run the full developer checklist. No license change or upstream filing is included.
