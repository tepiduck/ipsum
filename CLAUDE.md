# CLAUDE.md — frontend agent guide (Claude Code)

You own the **frontend**: a local dashboard that makes ipsum's results legible. The
backend (a separate agent, Codex, governed by `AGENTS.md`) produces JSON run
artifacts; you read and render them. **You never edit backend code.**

## What this project is (context, not your job to build)
ipsum tests whether a system with an explicit, cheaply-updated, **inspectable**
prior can **compound** — its rate of improvement beats a stateless baseline. Two
things make that legible, and they are your reason to exist:
1. **The slope plot** — the headline result (does ipsum out-improve the controls over time?).
2. **The abstraction inspector** — the "inspectable" bet is only real if a human can
   actually look at the abstractions: what exists, how useful each is, when each was
   admitted/evicted. The UI *is* the proof of inspectability.

Skim `README.md` for the thesis. You do **not** need the ML internals.

## Your scope
```
frontend/        <- everything you build lives here
```
You do **not** own and must **not** edit: `src/`, `experiments/` (except reading
artifacts), `data/`, `research/`, `_private/`, `AGENTS.md`.

## Your data source — INTERFACE.md is law
You consume only the JSON described in **`INTERFACE.md`**. Do not invent shapes;
read that file and a real artifact under `experiments/runs/<run_id>/` before
coding a view. Artifacts: `meta.json`, `slope.json`, `abstractions.json`,
`metrics.json`, `events.json`, and `experiments/runs/index.json` (list of runs).
Read them as **static JSON** — no live server, no Python. A dev step copies/symlinks
`experiments/runs/` into `frontend/public/runs/`.

If the backend hasn't produced artifacts yet, build against **mock fixtures** that
match INTERFACE.md exactly (put them in `frontend/src/fixtures/`, clearly labelled),
so your work isn't blocked — but the real app must read live artifacts.

## Views to build (priority order)
1. **Compounding** — render `slope.json` as a line chart: `value` vs `cycle`, one
   line per `system` (`weekly_retrain`, `data_matched_control`, `ipsum`). Make the
   `ipsum` vs `data_matched_control` gap visually obvious — that gap is the thesis.
2. **Abstraction Inspector** — from `abstractions.json`: a table of abstractions at
   a selected cycle (name, complexity, usefulness, admitted/evicted), plus a
   usefulness-over-time sparkline per abstraction and an admit/evict timeline from
   `events.json`. This is the differentiating view; make it good.
3. **Runs browser** — from `index.json`: pick a run, show its `meta.json` config and
   `metrics.json` (card metric vs control). Link the research story together.

## Stack (defaults — keep it simple)
- **Vite + React + TypeScript**, **Tailwind** for styling, **Recharts** for charts.
- No backend calls, no auth, no router beyond a simple tab/view switch. Static data only.
- Keep it a single small app; don't add state libraries or a component kit unless a
  view genuinely needs it.
(If you have a strong reason to deviate, note it in `frontend/README.md` — but
default to boring and shippable.)

## Hard rules
- **Stay in `frontend/`.** Never edit `src/`, `_private/`, `AGENTS.md`, or anything
  the backend owns. If you need a data field that doesn't exist, propose a change to
  `INTERFACE.md` (shared, by agreement) — don't reach into Python.
- **INTERFACE.md is the contract.** Parse defensively: unknown fields ignored,
  missing optional files degrade gracefully (e.g. no `abstractions.json` → hide that view).
- **Don't pre-format numbers in data** — format in the component.
- Keep the app runnable from a clean `npm install` with no secrets.

## Commands (you create these in frontend/)
```bash
cd frontend
npm install
npm run dev      # local dashboard
npm run build
npm run lint
```

## Definition of done (a view)
reads the real artifact shape from INTERFACE.md, renders from live
`experiments/runs/` data (not only fixtures), degrades gracefully when optional
data is absent, and runs from a clean `npm install`.
