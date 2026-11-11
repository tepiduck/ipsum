# Check verdict policy, version 1

This policy is fixed before the 0.16 verdict calibration. Calibration misses are reported;
the policy is not fitted to make the model catalog pass. `check` summarizes recorded evidence.
It does not replace primary outcomes, tolerances, failure signatures or reducer predicates.
SAFE means no blocking evidence in the selected workloads and level, not a proof about unseen
inputs, model quality, performance, training, or all compiler configurations.

## Decisions and precedence

The decision order is REGRESSION, INCONCLUSIVE, NEEDS_REVIEW, SAFE. Every verdict retains
counts and evidence paths for all planned batches, including failures and skipped diagnostics.

1. **REGRESSION (exit 1):** a two-environment comparison has a passing baseline and a stable
   reducible candidate failure, except a gradient-only mismatch fully explained below; or the
   fixed `accuracy_shift_replicated` rule flags a cross-environment accuracy change. These are
   regression candidates, never compiler-defect claims. Single-environment mismatches cannot
   establish an upgrade regression. An established candidate-only primary failure takes
   precedence over a contrary diagnostic label; contrary evidence alone is not a regression.
2. **INCONCLUSIVE (exit 3):** required preparation or primary execution is incomplete, unavailable,
   unstable, timed out, crashed, or exceeded memory, and no regression was established. A
   missing baseline must never be treated as a passing baseline. Harness/setup errors remain
   distinct from model/compiler failures in the reason codes and underlying outcomes.
3. **NEEDS_REVIEW (exit 2):** completed evidence has contrary numerical diagnostics, an unexplained
   mismatch, a recurring or changed failure, or required accuracy evidence is unavailable.
   Partial explanations are not contrary evidence, but cannot support SAFE. Standard/thorough
   two-environment checks requesting replication require at least ten completed batch scores;
   a factory supplying one batch reports replication unavailable and needs review, rather
   than asserting an accuracy regression. Optional mask diagnostics are unnecessary once a
   complete explanation is already established.
4. **SAFE (exit 0):** every planned primary execution is conclusive and passes or is a fully
   explained gradient-only mismatch, with no contrary evidence or replicated accuracy shift,
   and the requested level's required accuracy evidence is complete. Forward-only quick checks
   explicitly limit SAFE to forward execution. A single-environment SAFE does not assess an
   upgrade. Matching failures in both environments do not alone establish SAFE.

Only gradient-only mismatches can become nonblocking notes. The existing numerical triage
must classify them as `explained`, with complete target coverage through fp64 comparability
within the existing noise floor, eager control, or discontinuity evidence. A label string alone,
an incomplete explanation, or an unmatched diagnostic replay is insufficient. Verified contrary
evidence overrides reassuring labels. Passing primary results with contrary evidence also need
review. The underlying gradient_mismatch status is never changed.

## Replication and diagnostics

The frozen rule in [v016-policy.json](v016-policy.json) requires at least ten completed input
batches, candidate error worsening in at least 80% of them, and median batch growth at least
1.25x. Each batch contributes one median log2 compiled/fp64 gradient RMSE ratio. Missing batches
remain in coverage. Target-level `accuracy_shift` is descriptive only and never independently
causes REGRESSION. Previous flags and the original 0.11 evidence remain unchanged.

Primary workers and diagnostic workers have separate budgets and evidence directories.
Diagnostic resource limits are reported as `diagnostic_skipped_memory` (or timeout/error),
without replacing a completed primary observation. Incomplete required accuracy or explanations
produce NEEDS_REVIEW; they never produce SAFE by omission. Diagnostic replays must match the
recorded primary status and failure signature before their explanations are used.

## Presets fixed before measurement

| Level | Default batches | Primary | Accuracy | Explanations |
|---|---:|---|---|---|
| quick | 3 | forward, two repeats | not requested | none |
| standard | 10 | forward and gradients, two repeats | paired fp64, batch replication | lazy fp64/control, then masks only if unresolved |
| thorough | 10 | forward and gradients, three repeats | paired fp64, batch replication | full eager-control and AOT/mask diagnostics |

The defaults are CPU, one thread, 600 seconds and 12 GiB per worker. Users can choose smaller
budgets; no budget changes tolerances. A supplied single bundle remains one batch. Captured
collections keep their frozen inputs; selecting a prefix for quick checks is recorded explicitly.
Compiled and eager are both executed in the selected interpreter in single-environment mode.
Cross-environment comparisons use the same frozen inputs and independently recorded settings.

## Calibration plan

Report all 29 catalog models for same-version 2.15 comparisons and 2.15 versus the installed
2.16 nightly, with the planned denominator retained. Targets are SAFE for same-version checks,
REGRESSION for the baseline-confirmed historical corpus and clearly labeled injected faults,
and not REGRESSION for the exp_u20 GPT-2 replicated evidence. These targets are expectations,
not policy inputs. Separate fresh executions from retrospective summaries of retained evidence.
Publish wall time and peak worker-tree RSS for all three levels on ResNet-50, BERT and GPT-2,
and a standard SmolLM2-135M run under a 12-GiB worker budget. Report inconclusive runs and misses.

No license change or upstream filing is part of this release.
