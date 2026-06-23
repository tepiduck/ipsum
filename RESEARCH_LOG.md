# Research log

Append-only. One dated entry per experiment (RESEARCH.md §4). If an entry has no
metric and no control, it is not an experiment — don't log it.

Template:

```
## YYYY-MM-DD — <Card A/B/C or "instrument"> — <one-line claim>
- Hypothesis: I expect ___ to improve ___ over ___ because ___.
- Setup: testbed knobs / dataset / control.
- Result: the number(s). Plot path if any.
- Decision: keep / kill / iterate, and why.
- Next: the single next question.
```

---

<!-- newest entries on top -->

## 2026-06-23 — Card I — instrument self-check passes positive and negative controls
- Hypothesis: I expect the synth harness to show an ipsum-vs-data-matched slope gap only when ipsum has the admitted abstraction store enabled; if the store is disabled and ipsum is equivalent to the data-matched control, the gap should be zero.
- Setup: synth only; `n_files=96`, `n_tests=48`, `n_clusters=8`, `cycles=520`, `eval_interval=40`, `eval_window=80`, `selection_rate_cap=0.33`, `p_hit=0.9`, `p_flaky=0.01`, no delay, no drift; positive mode used ipsum with abstraction admission, negative mode used an ipsum selector with abstraction store disabled and data-matched control behavior. Verification was run under Python 3.10.20.
- Result: positive run `experiments/runs/20260623-003931-instrument-poscontrol-synth` had ipsum slope `0.001054`, data-matched slope `0.0000659`, slope gap `0.000988`, early gap `0.0`, final gap `0.403`, max SelectionRate `0.3125`; negative run `experiments/runs/20260623-003931-instrument-negcontrol-synth` had ipsum slope equal to data-matched slope, slope gap `0.0`, final gap `0.0`, max SelectionRate `0.3125`.
- Decision: PASS for the instrument; the harness detects a planted structural edge and does not fabricate a gap when ipsum is byte-equivalent to the control.
- Next: proceed to Card B eviction using this harness with drift enabled.

## 2026-06-23 — instrument — refreshed synth slope after harder admission
- Hypothesis: I expect the synth compounding harness to remain positive after the stronger Card A admission rule because abstractions should still add selection signal beyond cumulative raw failure rates.
- Setup: synth only; same instrument config as the prior run, after raising cluster-correlated commit signal and changing admission to a one-standard-error lower-bound rule with real complexity cost.
- Result: run `experiments/runs/20260623-002320-instrument-synth`; final TestRecall `0.857` for ipsum vs `0.458` data-matched and `0.447` weekly-retrain; slope gap vs data-matched `0.001007`; final selection rate `0.333`.
- Decision: keep the synth instrument as a smoke-level slope check; this is not Card A evidence and not a real-data thesis result.
- Next: run the RTPTorrent path once real project CSVs and changed-file metadata are available.

## 2026-06-23 — Card A — confidence-bound admission passes granularity gate
- Hypothesis: I expect a one-standard-error lower bound on held-out LL gain, minus a per-file complexity cost, to recover true clusters better than admit-everything at every granularity because noisy or redundant candidates should fail the confidence-adjusted cost test.
- Setup: synth only; `n_files=96`, `n_tests=48`, `n_clusters_sweep=(4, 8, 12, 16)`, `train_cycles=480`, `heldout_cycles=160`, `p_hit=0.9`, `p_flaky=0.01`, no delay, no drift; admission used `gain_confidence_z=1.0`, `complexity_per_file=0.0015`, `cochange_threshold=0.18`, `min_support=3`, `max_candidates=256`; controls were no-abstraction and naive admit-everything.
- Result: run `experiments/runs/20260623-002149-A-synth`; mean cluster F1 `0.783` vs admit-everything `0.209`; held-out LL gain `0.1116`; average rejection fraction `0.256`; per-granularity F1 margins vs admit-everything were `0.367` at 4 clusters, `0.669` at 8, `0.679` at 12, and `0.584` at 16.
- Decision: keep for Card A on synth; this supersedes the earlier weaker run because the complexity term now rejects a meaningful fraction and the granularity gate passes.
- Next: use this gate as the regression target before Card B eviction work.

## 2026-06-22 — instrument — synth slope harness with data-matched control
- Hypothesis: I expect the synth compounding harness to show ipsum's TestRecall slope above the data-matched abstraction-off control because admitted co-change abstractions add decision signal beyond cumulative per-test failure rates.
- Setup: synth only; `n_files=96`, `n_tests=48`, `n_clusters=8`, `cycles=520`, `eval_interval=40`, `eval_window=80`, `selection_rate_cap=0.33`, `p_hit=0.9`, `p_flaky=0.01`; comparison lines were weekly-retrain raw rates, cumulative data-matched abstraction-off raw rates, and ipsum with online abstraction admission.
- Result: run `experiments/runs/20260622-233842-instrument-synth`; final TestRecall `0.808` for ipsum vs `0.458` for data-matched and weekly-retrain; slope gap vs data-matched `0.000948`; final selection rate `0.333`.
- Decision: keep the instrument for synth; it emits the required `meta.json`, `slope.json`, `metrics.json`, and `abstractions.json`, and it verifies the thesis comparison shape before RTPTorrent.
- Next: implement an RTPTorrent loader that yields the same cycle/commit/outcome shape, then wire the weekly-retrain baseline to real features instead of raw-rate synth features.

## 2026-06-22 — Card A — held-out LL admission recovers synth clusters
- Hypothesis: I expect held-out predictive log-likelihood gain minus complexity to improve cluster recovery over admit-everything and improve held-out LL over no-abstraction because useful co-change abstractions should predict repeated test failures while spurious candidates should not earn their cost.
- Setup: synth only; `n_files=96`, `n_tests=48`, `n_clusters_sweep=(4, 8, 12)`, `train_cycles=240`, `heldout_cycles=120`, `p_hit=0.9`, `p_flaky=0.01`, no delay, no drift; controls were no-abstraction and naive admit-everything.
- Result: run `experiments/runs/20260622-233108-A-synth`; cluster F1 `0.572` vs admit-everything `0.366`; cluster precision `0.921`, recall `0.478`; held-out LL gain `0.0461` per observation over no-abstraction.
- Decision: keep for Card A on synth; the representation clears the isolating metric, though recall drops on the hardest `n_clusters=12` setting (`0.315` F1), so later iterations should improve candidate granularity rather than treat this as final mechanism quality.
- Next: build the experiment instrument/data-matched abstraction-off control, then use the same artifact path for the RTPTorrent-facing harness.
