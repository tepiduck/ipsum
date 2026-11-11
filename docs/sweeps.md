# Pretrained model sweeps

The `sweep` commands prepare pretrained models and compare two execution environments.
The original default catalog contains 16 torchvision classifiers and four Hugging Face base models:
BERT, GPT-2, DistilBERT, and SmolLM2-135M (Llama architecture). Text workloads return the final hidden
states; task-specific prediction heads and generation caches are not exercised.

The nine-model `--set held-out` catalog adds RegNet-Y-400MF, DenseNet-121, ShuffleNet-V2-x1.0,
SqueezeNet-1.1, MNASNet-1.0, the T5-small encoder, RoBERTa-base, OPT-125M, and Pythia-160M.
These were selected before the 0.11 held-out validation. `--set original` selects the existing
20; `--set all` selects all 29. `--model` and `--set` are mutually exclusive. The default stays
the original catalog so existing prepared directories remain usable.

T5 loads the encoder only. RoBERTa omits the unused pooler rather than initializing a missing
pooler randomly. All remaining model parameters must come from the pinned checkpoint.
OPT-125M uses the existing [SFconvertbot safetensors conversion](https://huggingface.co/facebook/opt-125m/discussions/51)
at commit `1f9886ce095904096e22b0f4d9e7ba932fa7df2a`; its main revision has only pickle weights.
The loader first resolves its declared embedding/head weight tie through `AutoModelForCausalLM`,
then extracts the base model. Direct base-model loading would miss the embedding stored under
the head's name. Missing or mismatched weights still fail preparation.
Preparation requires safetensors at the selected revision and never requests an automatic
upstream conversion. The resolved revision and its source are recorded in the workload manifest.

```bash
# Install the matching PyTorch/torchvision build in each environment first.
python -m pip install -e '.[sweep]'
faultline sweep list
faultline sweep prepare --output workloads/pretrained
faultline sweep run workloads/pretrained --baseline-python .stable/bin/python --candidate-python .nightly/bin/python --device cpu --output faultline-runs/pretrained-cpu
faultline sweep run workloads/pretrained --baseline-python .rocm-stable/bin/python --candidate-python .rocm-nightly/bin/python --device cuda:0 --output faultline-runs/pretrained-amd
faultline sweep prepare --set held-out --output workloads/held-out
faultline sweep run workloads/held-out --set held-out --baseline-python .rocm-stable/bin/python --candidate-python .rocm-nightly/bin/python --device cuda:0 --deterministic --output faultline-runs/held-out-amd
```

Use `--model NAME` repeatedly to prepare or run a subset. Each interpreter needs the model libraries
and their dependencies. The run defaults to Inductor, AOT capture, fixed VJP gradients, numerical
diagnostics, two repeats, one thread, 600 seconds and 12 GiB of host RSS per worker. Override the
execution budgets explicitly when needed. GPU memory has no separate watchdog.
Use `--deterministic` to request the saved deterministic execution policy in both environments.
This changes the experiment; results must identify the policy when compared with a default run.

Preparation downloads actual pretrained weights, two photographs from official PyTorch example
repositories, and two paragraphs from Project Gutenberg's public-domain *Alice's Adventures in
Wonderland*. Vision inputs use the resolved weight enum's inference transforms. Text inputs use
the checkpoint's tokenizer, two padded/truncated 64-token sequences, and SDPA attention. All model
parameters use float32 storage. Input tensors, fixed backward vectors, model state, provider
versions, checkpoint revisions/URLs, transforms, and SHA-256 hashes are retained. Hugging Face
downloads use safetensors and reject remote model code. Missing checkpoint weights are an error;
unused task-head weights are recorded. Worker replay reads local assets and makes no model download.

This input set checks concrete real-input execution paths. It is small, contains two dog photographs
and two literary passages, and is not representative coverage of ImageNet, language tasks, or
production traffic. Weights and inputs stay in local artifacts rather than being committed to the
repository. Review model/dataset terms and artifact contents before redistributing them.

Every selected model stays in the denominator. Preparation failures, unavailable devices, compiler
errors, timeouts, memory limits, unstable comparisons, and incomplete diagnostics remain visible.
Both interpreters execute the same frozen workload. There is no CPU/backend substitution, tolerance
relaxation, automatic model replacement, or retry that erases a completed failure.

The report contains per-model baseline/candidate statuses, environment-gate classification,
whole-finding labels, matched/total target counts, control policy, and mask coverage. Summary counts
separate full agreement, accuracy advisories, explained findings, contrary-evidence findings,
missing-evidence findings, and inconclusive pairs. Label counts can
overlap: a model/device pair counts once for a label present in either version. A finding remains
unexplained if either failing version lacks sufficient diagnostic evidence. Contrary evidence in
either version takes precedence over missing evidence. A failed mask diagnostic alone does not
override a complete eager-control explanation. Same-input backward uses its fp64-relative judgment
while retaining the elementwise result. The [numerical evidence guide](numerics.md) defines these
criteria and the fixed cross-version accuracy threshold. The HTML table shows the advisory and
qualifying/total counts even when both versions already fail. A passing baseline with a candidate failure is a regression candidate;
changes in model-library and vendor builds must be considered before attributing it to PyTorch.

These are disagreement and diagnostic-coverage measurements. They are **not a false-positive rate**:
that would require independently adjudicated correctness labels. Neither fp64 proximity nor an
eager-control discrepancy proves that an observed compiler difference is harmless.

To continue after an interruption, repeat the same run command with `--resume`. Completed case
documents are verified by their recorded hashes. Source, interpreters, packages, device/runtime,
settings, and prepared input identity must match. An unfinished attempt is retained and a new
attempt directory is used. Completed inconclusive results remain completed evidence; use a new
campaign with an explicit changed policy to investigate them. Run campaigns sequentially when
they share host RAM or the same GPU, so resource contention does not dominate the measurements.

## Multiple inputs: 0.15 design (historical)

A model's output and parameter gradients from one input batch are correlated targets, not
independent repetitions. The current sweep's target-level `accuracy_shift` sign test is
descriptive. Do not interpret its target count as an input sample size.

An optional multi-input extension should prepare a fixed list of disjoint input batches before
execution and keep model weights, tokenizer, VJP construction and compiler settings fixed. Run
each batch in fresh workers in both environments, store its existing primary result unchanged,
then reduce its paired gradient errors to one median log2 growth score. Apply a separate
batch-level sign test; retain missing batches in the planned denominator. Resume must validate
each batch's input/probe hashes as well as the shared model and environment identities.

The [0.15 investigation](input-accuracy.md) implements that analysis as a standalone example and
summary script. It is not a new `sweep` CLI option. Its predeclared twenty-batch decision rule is
specific to that experiment; existing sweep thresholds and previous labels remain unchanged.

## 0.16: input batches are the replication unit

The design above is now available through `sweep prepare --input-batches N`:

```bash
faultline sweep prepare --set all --input-batches 10 --output prepared-016
faultline sweep run prepared-016 --set all --baseline-python /path/stable/bin/python \
  --candidate-python /path/nightly/bin/python --device cpu --output sweep-016
# Continue with the same arguments and --resume.
```

Preparation fixes all batches before comparison. Text batches contain pairs of disjoint
200-word passages after the original two paragraphs, tokenized to 64 tokens. Vision batches
contain disjoint vertical strips of the two original photographs, resized by the checkpoint's
usual transform. These are **not independent photographs or a representative image dataset**;
the two source images and one book remain shared sources. The input manifest records source
regions and hashes. No sample is replaced after observing a failure.

Each batch has an independently saved VJP probe, generated with seed equal to its batch index,
and the same model weights. The collection hashes every workload, input and probe. Its
workers run each batch separately in both environments. The batch results retain all primary
statuses and the old `accuracy_shift`, explicitly marked **single_input, descriptive only**.
No new label changes the primary gate, reduction predicates or exit codes.

The policy was declared in [v016-policy.json](v016-policy.json) before the new run. Each completed
batch contributes the median log2 candidate/baseline compiled-to-fp64 RMSE growth over paired,
present parameter gradients. The new `accuracy_shift_replicated` flags when at least ten batches
complete, at least 80% of completed batches worsen, and their median growth is at least 1.25x.
This agrees with the earlier 16/20 threshold; historical 0.15 classifications remain unchanged.
The exact one-sided binomial tail uses worsening versus improving batches, excluding exact ties.
It is reported descriptively, not used as another flag threshold.

Missing or unusable batches remain in the planned denominator and coverage. The flag's fraction
uses completed batches, so a flagged partial campaign must still be read with its coverage.
Zero/zero target errors tie; zero/nonzero gives positive infinity; nonzero/zero gives negative
infinity. An indeterminate median or changed gradient set makes a batch incomplete.

The JSON/HTML report contains the full score distribution, sign counts, p-value and coverage.
Resume checks source, settings, interpreters, packages, workload and probe identity, and committed
result hashes. Interrupted attempts are retained; completed inconclusive results are not replaced.
Batch comparisons reference their prepared bundles rather than duplicating large weights into
two standalone exports per batch. Retain the prepared directory alongside the results.

For custom workloads, use `model capture factory.py:factory --input-batches N --output batches`.
The factory returns `(eval_model, sequence_of_N_input_batches)`. Each input batch has the same
format as an ordinary factory's inputs, with a common leading sample dimension. Capture rejects
repeated tensor-valued samples across batches and changes to the frozen weights. It does not
infer whether distinct values represent overlapping real-world examples; the caller must supply
disjoint source examples. `--trace` and scalar-loss probes are not supported for this mode.
`model compare batches ... --resume` uses the same batch coordinator and advisory.

### Annotation of the 0.11 flags

All original 0.11 `accuracy_shift` labels and measurements remain intact. They are
**single_input, descriptive only**: their target counts are not input-replication counts.
The [exp_u20 case study](accuracy-investigation.md#case-study-from-a-sweep-flag-to-input-replication)
shows why localizing a real numerical change is not sufficient to call it a systematic regression.
