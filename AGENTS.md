# Repository Guidelines

## Project Structure & Modules
- `src/bbx/`: core Python package. The CLI entry point lives in `src/bbx/__main__.py`; commands delegate to modules such as `analyze.py`, `metrics.py`, `regionize.py`, and `export.py`.
- `data/`: captured runs (`run_000/step_0000.json` etc.). `bbx exportcsv` flattens them.
- `reports/`: output directory for command results (e.g., `reports/metrics`, `reports/analyze`).
- `config/`: shared configuration (`masks.json` for reusable region definitions).
- Tests live alongside modules; run them via empathy (see below) or targeted scripts.

## Build, Test & Dev Commands
- `bbx analyze ...`: primary analysis/test harness (coverage, training, simulation). Scopes/region masks set via CLI flags.
- `bbx metrics ...`: batch statistics (entropy, MI, label fits, adjacency).
- `bbx regionize --run data/run_000`: produces wall/interior masks and plots.
- `bbx exportcsv --run-dir data --run-glob 'run_*' --out reports/runs.csv`: aggregates all runs into a single CSV for external tooling.
- Run unit tests (where present) with `python -m pytest` from repo root.

## Coding Style & Naming
- Python 3.11+, PEP 8 compliant; 4‑space indentation. Prefer descriptive snake_case for functions/variables; camelCase reserved for CLI/options.
- Use type hints (`Path`, `np.ndarray`) for readability. Save artifacts via helper functions (`save_json`, `save_csv`) for consistency.
- Avoid hard-coded paths; accept CLI flags or reference `config/masks.json`.

## Testing Guidelines
- Generate smoke tests by running `bbx analyze` on sample data (`data/run_000`). Inspect key artifacts (`summary.json`, `period_scan_*.csv`) to ensure commands succeed.
- When modifying metrics, re-run `bbx metrics --runs data/run_000 --config config/masks.json --out reports/metrics` and inspect PNG/CSV outputs.
- Prefer adding targeted pytest modules in `tests/` for pure functions (e.g., unit conversions, mask validation).

## Commit & PR Conventions
- Use concise, imperative commit messages (“Add region-aware classifier export”, “Fix metrics combo option”).
- PRs should describe the feature/fix, list CLI examples (commands + flags), and include screenshots or sample artifacts when applicable (e.g., `label_fit_gru.png` diff).
- Reference issues with `Fixes #NN` when closing; note data dependencies (sample runs) if QA needs to rerun commands.
