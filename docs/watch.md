# Nightly watch

`faultline watch` runs standard-level checks against a pinned stable environment and
the immediately preceding available nightly. It runs once and exits; cron, systemd,
or GitHub Actions supplies the schedule. The default catalog is pretrained ResNet-50,
MobileNetV3-Small, BERT, GPT-2 and ViT-B/16. Every model uses ten frozen input batches.

```bash
faultline watch --output faultline-runs/watch-cpu
faultline watch --output faultline-runs/watch-cpu
```

The second invocation verifies the existing campaign and skips committed comparisons.
It resumes interrupted checks or appends a newly available nightly. Stable defaults to
`2.15.0`; `--stable VERSION` pins another release. `--baseline-python /path/to/python`
uses an existing interpreter, whose PyTorch version must match `--stable`.
Use the same output directory and options when resuming. A changed Faultline source,
catalog, stable environment, frozen inputs or execution policy needs a new history directory.
Old history stays readable. Keep the virtual environments and frozen inputs with a local campaign.
An incomplete selection is frozen in `plans/`; a newer wheel cannot displace unfinished
work. Resume with the original `--seed-nightlies` and `--through` options. Once that plan
finishes, a subsequent invocation can discover and append newer nightlies.

Open `index.html` for per-model HTML timelines or read `history.json`. Each row links to
the standard check report and retains the exact candidate wheel version, upstream commit,
reference environment, verdict, replication coverage, resource allocation and compilation
notes. `records/` contains immutable committed rows; the timeline is a derived view.
Interrupted work remains pending in the planned denominator. Setup or installation failures
retain logs and a resumable incomplete state rather than becoming a passing result.

## Catalog and resources

A catalog is a JSON list. Use official pretrained model names or an existing collection
of exactly ten frozen input batches:

```json
[
  {"id": "resnet50", "torchvision": "resnet50"},
  {"id": "bert", "hf": "google-bert/bert-base-uncased"},
  {"id": "production", "bundle": "/data/frozen-production-batches"}
]
```

```bash
faultline watch --catalog models.json --output faultline-runs/production-watch
faultline watch --seed-nightlies 7 --through 2026-11-14 --output faultline-runs/seed
```

Seeding seven nightlies also installs the preceding anchor, so five models produce
70 standard comparisons, not seven checks. This can take many hours and retain substantial
model/graph data. `--max-wall-s` defaults to 21,600 seconds per invocation and bounds check
subprocesses; installation and preparation also have per-stage timeouts. Run again to
continue unfinished work. `--jobs auto`, memory calibration and serial retries use the
existing check scheduler; `--max-rss-mb`, `--jobs` and `--timeout` retain their check meanings.
The chosen RAM budget is frozen for resume and printed before work starts.

The installer reuses isolated environments and pip's wheel cache. `--environment-cache`
can share ready environments between compatible watches and bisections. NumPy, psutil,
packaging, transformers, safetensors, Pillow, tokenizers and huggingface-hub are pinned to
the stable environment when creating nightlies. A cached environment with different
versions is rejected. Torchvision uses compatible wheels from the selected PyTorch index;
its version is recorded separately. These are environment comparisons, including torchvision,
not a claim that only one upstream package changed. Do not modify environments mid-campaign.

## Regression handling

Only a v2 `REGRESSION` triggers automatic investigation. Compilation advisories cannot
trigger it. Watch exports the complete frozen batch collection and locates the last observed
SAFE nightly for the same model and comparison track. It runs `bisect --predicate verdict-v2`
between that nightly and the first observed regression, using a fixed reference environment,
then confirms both boundary builds with new check executions. Stable comparisons retain the
pinned stable reference; previous-nightly comparisons use the last SAFE nightly as reference.
Ten batches and the existing repeated primary executions provide replication; numerical
accuracy uses the unchanged batch-level rule. No new tolerance or failure predicate is introduced.

If there is no observed SAFE nightly, the report says that no bracket is available. An
inconclusive intermediate build cannot be called good or bad; a boundary containing it
is explicitly nonadjacent. Binary search assumes monotonicity and does not prove that there
were no intermediate recoveries. The independent endpoint checks remain visible.

```bash
faultline bisect frozen/batches --predicate verdict-v2 \
  --baseline-python /path/to/fixed-reference/python \
  --start 2026-11-07 --end 2026-11-13 --output faultline-runs/bisect
```

The exported `replicated-reproducer/repro_check.py` includes all ten frozen batches and
needs Faultline plus the factory dependencies. Supply baseline/candidate interpreter paths
and a new output directory. Existing per-batch `repro.py` exports still run without Faultline,
but one batch alone cannot reproduce a replicated accuracy decision. Nothing is filed upstream.

## CPU CI

[The scheduled workflow](../.github/workflows/watch.yml) checks ResNet-50 and MobileNetV3-Small
in separate CPU jobs each day at 09:43 UTC. It caches pip wheels and compatible history,
allows two hours of watch work within a 150-minute job cap, and uploads evidence for 30 days.
It fails visibly for any non-SAFE/incomplete result; a failed workflow is not automatically
a compiler-regression finding. The full five-model default can be used on a larger runner.

CI caches contain frozen inputs and committed JSON/HTML history; full worker artifacts are
in that run's downloadable artifact, not guaranteed to survive cache eviction. Interrupted
checks may need replay on the next runner. Caches are scoped by Faultline source and runner
image/Python identity; an incompatible image starts a new history rather than changing an old
campaign's environment. Download retained artifacts before their retention period ends.

## WSL and AMD scheduling

Keep the checkout, environments and compiler caches on the Linux filesystem. For an AMD
watch, choose the nightly index matching the installed ROCm runtime. Faultline validates
the wheel's device build and never substitutes CPU execution.

```bash
faultline watch --device cuda:0 --deterministic \
  --index-url https://download.pytorch.org/whl/nightly/rocm7.0 \
  --output /data/faultline-watch-amd
```

`rocm7.0` is an example index, not automatic driver detection; use the index for your
installation. GPU execution still uses PyTorch's `cuda:0` device spelling.

An example systemd user service (`~/.config/systemd/user/faultline-watch.service`):

```ini
[Unit]
Description=Faultline nightly model watch

[Service]
Type=oneshot
WorkingDirectory=%h/projects/ipsum
ExecStart=%h/projects/ipsum/.venv/bin/faultline watch --output %h/faultline-watch --max-wall-s 18000
TimeoutStartSec=6h
```

And `faultline-watch.timer`:

```ini
[Unit]
Description=Run Faultline daily

[Timer]
OnCalendar=*-*-* 03:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now faultline-watch.timer
```

Alternatively, add this to `crontab -e`, adjusting paths:

```cron
0 3 * * * cd "$HOME/projects/ipsum" && .venv/bin/faultline watch --output "$HOME/faultline-watch" >> "$HOME/faultline-watch.log" 2>&1
```

Campaign locks prevent overlapping watches. WSL must be running for Linux scheduling
to execute; these examples do not start WSL from a powered-off Windows session. The timer
uses the Linux system's local timezone. No local scheduler is installed by the CLI.
