# Dataset scouting — milestone 1

**Bottom line:** don't build a CI scraper first. Two curated datasets already provide
per-test pass/fail parsed from real build logs, which is the part that's hard to get
from raw public CI (GitHub Actions only exposes job-level status unless a repo uploads
JUnit artifacts). Replay these offline to measure the slope.

## Primary: RTPTorrent (MSR 2020, Mattis et al.)
- **What:** 20 open-source **Java** GitHub projects, **100,000+ Travis CI build logs**, **9 years** of history.
- **Granularity:** per build job AND **per test case**: total/failed/errored/skipped test methods, original run **index** (order), and JUnit **duration**. Plus version-control metadata. Baseline approaches included for comparison.
- **Why it fits ipsum:** the 9-year per-project chronology is exactly what you need to **replay commits in order and measure dQ/dt** (the slope). Per-test outcomes give clean labels. Baselines included = less reimplementation.
- **Access:** Zenodo (records 3712290 / 4046180); paper PDF on author site.
- **Caveats:** Travis-based and historical (Travis dropped free OSS ~2021), **Java-only**. Perfect for an *offline* proof-of-compounding; NOT a live feed. A live product later would need GitHub Actions + JUnit-artifact scraping — but that is a productization concern, not a thesis concern.

## Secondary + baseline harness: "Revisiting ML-based TCP for CI" (ICSME 2023, arXiv 2311.13413)
- **What:** 11 open-source subjects, latest **800 commits each** (= CI cycles), **8,800 versions**; mix of **GitHub Actions** and Travis. Studies 11 representative ML prioritization techniques.
- **Why useful:** (1) a second, partly-GitHub-Actions dataset; (2) a **replication package** (Zenodo records/7036507) with code + data + scripts — reuse it as your baseline harness instead of reimplementing from scratch.
- **THE finding that sharpens your experiment:** they report that the performance change of existing ML techniques across CI cycles comes **"mainly from the changing amount of training data, instead of code evolution."** In other words, the apparent "it gets better over time" of current methods is **mostly just data accumulation** — not learned structure.

## Why that finding matters (and is good for you)
It is exactly the confound flagged in DESIGN.md, now empirically confirmed in the literature: naive "improves over time" is illusory. This **raises the bar and clarifies the moat**:

- The baseline to beat is NOT only "weekly-retrain XGBoost." It must include **"same model, same cumulative data, abstraction store OFF."**
- ipsum's claim is only proven if its slope **exceeds what pure data-accumulation produces** — i.e., the abstraction store contributes improvement *on top of* having seen the same data.
- If ipsum can't beat the data-matched, abstraction-off control, the thesis is false on this domain — and you learn that cheaply, on existing data, without writing a scraper.

## Recommended milestone-1 plan
1. Pull **RTPTorrent** from Zenodo; pick 3–5 projects with the longest history and highest test-failure density.
2. Stand up the **2023 replication package** as the baseline harness; reproduce sane TestRecall/SelectionRate numbers.
3. Define the **data-matched, abstraction-off control** as the primary comparison (not just weekly-retrain).
4. Replay chronologically; plot slope of TestRecall@fixed-SelectionRate for: weekly-retrain baseline, data-matched abstraction-off control, and ipsum.

## Sources
- RTPTorrent paper: https://toni.mattis.berlin/files/2020-preprint-mattis-rtptorrent-msr20.pdf
- RTPTorrent data: https://zenodo.org/records/3712290
- Revisiting ML TCP (paper): https://arxiv.org/abs/2311.13413
- Revisiting ML TCP (replication package): https://zenodo.org/records/7036507
