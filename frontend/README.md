# ipsum frontend

Local dashboard for ipsum results. Reads **static JSON run artifacts** (no backend
calls, no Python). The contract is `../INTERFACE.md`; agent rules are `../CLAUDE.md`.

## Run
```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
```
With no data present it boots on **fixture data** (amber badge). It switches to
**live data** (green badge) automatically once real artifacts are available.

## Wiring in live data
The backend writes runs to `../experiments/runs/`. Expose them as static files:
```bash
# from frontend/
mkdir -p public
ln -s ../../experiments/runs public/runs      # or copy if your OS dislikes symlinks
```
The app fetches `/runs/index.json` and `/runs/<run_id>/*.json`.

## What's here
- `src/data.ts` — loader: tries `/runs/*`, falls back to fixtures.
- `src/types.ts` — TypeScript mirror of INTERFACE.md (that file wins on conflict).
- `src/fixtures/sampleRun.ts` — labelled mock data so the app runs against a blank backend.
- `src/components/CompoundingChart.tsx` — the headline slope plot (implemented).
- `src/components/AbstractionInspector.tsx` — store table + timeline (implemented; sparkline = TODO).
- `src/components/RunsBrowser.tsx` — run picker + meta/metrics (implemented).

## TODO (good next tasks)
- Usefulness-over-time sparkline per abstraction (read value across all snapshots).
- Highlight stale/decaying abstractions; mark drift cycles on the slope chart.
- Diff two runs side by side.

## Rules (see ../CLAUDE.md)
Stay in `frontend/`. Don't import Python or edit backend code. If you need a new
data field, propose it in `../INTERFACE.md` — don't reach across the boundary.
`npm run lint` is `tsc --noEmit` for now; add ESLint if you want stricter checks.
