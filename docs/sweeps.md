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
