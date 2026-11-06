# Pretrained model sweeps

The `sweep` commands prepare a fixed catalog of 20 pretrained models and compare two execution
environments. The catalog contains 16 torchvision classifiers and four Hugging Face base models:
BERT, GPT-2, DistilBERT, and SmolLM2-135M (Llama architecture). Text workloads return the final hidden
states; task-specific prediction heads and generation caches are not exercised.

```bash
# Install the matching PyTorch/torchvision build in each environment first.
python -m pip install -e '.[sweep]'
faultline sweep list
faultline sweep prepare --output workloads/pretrained
faultline sweep run workloads/pretrained --baseline-python .stable/bin/python --candidate-python .nightly/bin/python --device cpu --output faultline-runs/pretrained-cpu
faultline sweep run workloads/pretrained --baseline-python .rocm-stable/bin/python --candidate-python .rocm-nightly/bin/python --device cuda:0 --output faultline-runs/pretrained-amd
```

Use `--model NAME` repeatedly to prepare or run a subset. Each interpreter needs the model libraries
and their dependencies. The run defaults to Inductor, AOT capture, fixed VJP gradients, numerical
diagnostics, two repeats, one thread, 600 seconds and 12 GiB of host RSS per worker. Override the
execution budgets explicitly when needed. GPU memory has no separate watchdog.

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
separate full agreement, findings, inconclusive pairs, and unexplained findings. Label counts can
overlap: a model/device pair counts once for a label present in either version. A finding remains
unexplained if either failing version lacks sufficient diagnostic evidence. `discontinuity_present`
or a disagreeing same-input backward remains unexplained for triage,
even alongside another label. A passing baseline with a candidate failure is a regression candidate;
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
