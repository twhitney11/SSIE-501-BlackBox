# Black Box Cellular Automaton Toolkit (bbx)

This package provides tools to **collect, process, and analyze** data from the “Black Box” cellular automaton (CA) system used in our SSIE 501 project.\
It is designed to help us replicate, visualize, and hypothesize about the underlying rules of the system.

## Installation

Clone the repository and install in editable mode:

```
git clone https://github.com/twhitney11/SSIE-501-BlackBox.git ssie501-bbx
cd ssie501-bbx
pip install -e .
```

This provides a single command-line entry point: `bbx`.

Check it works:

```
bbx --help
```

## Workflow

The workflow is broken into four main steps:

### 1. Data collection

Scrape states directly from the `BlackBox.php` interface.

```
bbx collect --base-url http://localhost/BlackBox.php --steps 500 --out data
```

* Saves each run as JSON (`step_0000.json`, …).
* Optional: `--runs` controls how many independent runs to capture.
* Starts with a reset unless you override.

### 2. Processing

Summarize runs: labels, neighborhood coverage, conflicts.

```
bbx process --runs data/run_000 data/run_001
```

Outputs number of unique neighborhoods and step-level conflicts.

### 3. Analysis

Check reproducibility, test model hypotheses, train classifiers.

```
bbx analyze --runs data/run_000 --radius 1 \
  --tests conflicts,permutation,totalistic \
  --period-scan edge,corner_edge \
  --train-clf --clf-period 2 \
  --out reports
```

This can generate:

* **Conflicts examples** — where the same neighborhood produces different outcomes.
* **Permutation test** — shuffle label IDs to test if labels matter.
* **Totalistic vs positional** — test whether rules depend only on *counts* of neighbors vs their *positions*.
* **Period scan** — see if the system evolves with a hidden cycle (e.g. 2-step oscillation).
* **Classifier training** — logistic regression on neighborhood + features, used to simulate forward.

All artifacts are saved to `reports/` (JSON, CSV, PNGs).

### 4. Fractions reports

Track black, white, and colored label fractions over time.

```
bbx fractions --runs data/run_000 --out reports
```

Generates:

* Raw and smoothed fraction CSVs.
* Plots of each label’s frequency.
* Stacked black/white/other trajectories.
* Regional breakdown (edge vs interior).
* Change curves and black-front progression.

## Why these analyses?

* **Conflicts**: if rules produce different outputs for the same neighborhood, the system can’t be purely local deterministic. Tracking conflicts helps spot hidden time-dependence or region-dependence.
* **Permutation invariance**: tests whether labels are arbitrary or semantically meaningful. If performance is unchanged after shuffling labels, colors are just “placeholders.”
* **Totalistic vs positional**: checks if rules depend only on the *counts* of neighbors (like Conway’s Game of Life) or on their *positions*. This hints at whether symmetry matters.
* **Period scan**: explores whether the system locks into repeating phases (like oscillators). Splitting by region (interior, edge, corner) can reveal structure.
* **Classifier modeling**: instead of memorizing 500 hard-coded steps, we learn a compact rule-based model. Logistic regression or decision trees can capture the underlying update rules.

## Interpreting results

* If black/white fractions converge → **self-organization into an attractor**.
* If black continually expands inward → **erosion-like model**.
* If conflicts vanish only at a particular period (e.g. k=2) → **time-dependent rules**.
* If permutation tests show sensitivity → labels have meaning (not arbitrary colors).

## Project structure

```
bbx/              # Python package
  __main__.py     # CLI entrypoint
  collect.py      # HTML scraper for runs
  process.py      # Encoding, rule learning, summaries
  analyze.py      # Tests, classifier, simulations
  fractions.py    # Fractions reporting and plots
  viz.py          # Plotting helpers (color maps, sequences)

data/             # Saved runs (step_XXXX.json)
reports/          # Generated CSVs, JSONs, PNGs
label_colors.json # Mapping from labels → colors
```

## Example: Full cycle

```
# 1. Collect
bbx collect --base-url http://localhost/BlackBox.php --steps 500 --out data

# 2. Process
bbx process --runs data/run_000

# 3. Analyze
bbx analyze --runs data/run_000 --radius 1 \
  --tests conflicts,permutation,totalistic \
  --period-scan edge,corner_edge \
  --train-clf --clf-period 2 \
  --out reports

# 4. Fractions
bbx fractions --runs data/run_000 --out reports

```

