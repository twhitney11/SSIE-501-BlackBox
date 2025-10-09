# Black Box Cellular Automaton Toolkit (bbx)

This package provides tools to **collect, process, and analyze** data from the “Black Box” cellular automaton (CA) system used in our SSIE 501 BlockBox project. It is designed to help us replicate, visualize, and hypothesize about the underlying rules of the system.

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
  --tests shape,conflicts,permutation,totalistic \
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
* **Shape diagnostics** — optional flags such as `--do-ring`, `--do-cycle`, `--do-torus`, `--autocorr`, and `--near-cycle` add ring-radius tracking, cycle detection, torus experiments, and correlation scans.
* **Perturb & simulate** — `--perturb ...` and `--simulate ...` pair with `--sim-steps`, `--sim-png`, etc. to stress the inferred classifier and export rollout JSON/PNGs.
* **Rulebooks** — `--build-rulebook` (with `--rb-k`, `--rb-split`) exports exact phase rulebooks; `--simulate-rulebook ... --rb-steps` replays them with optional PNG snapshots.

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

### 5. Rulebook exploration

Inspect exported rulebooks for symmetry and dependence structure.

```
bbx rbxplore --rb reports/rulebooks.json --do orient,center,mi,tree --out reports/rbxplore
```

Produces CSVs covering orientation invariance, center dependence, positional mutual information, and small decision-tree summaries. Configure the tree depth/phase (`--tree-depth`, `--tree-phase`) to control the simplification.

### 6. Spatial grid heatmaps

Quantify per-cell behaviour across long runs.

```
bbx gridmaps --runs data/run_000 data/run_001 --out reports/gridmaps --win last:200 --k 2
```

Generates heatmaps/CSVs for change rate, per-label occupation, entropy, conditional entropy, phase flip rate, and time-to-black. Use `--phase` to isolate a particular phase slice and `--win` to focus on time windows.
Pass `--sequence` to also render a montage of states from the first run (`--sequence-steps` accepts comma values or `auto[:N]`).

### 7. Regionize walls

Derive wall/inside/outside masks for a dominant label over a time window.

```
bbx regionize --run data/run_000 --window 3500:5000 --label gru --wall-thickness 2 --out reports/regionize
```

Outputs occupancy maps, wall/inside/outside masks (plus near-wall bands), and per-region summary stats (cell count, entropy, conditional entropy, flip rate, change rate).
PNG visuals are emitted alongside the CSVs for quick inspection: occupancy heatmap with the wall contour, color-coded region overlay, metric heatmaps, and bar charts comparing region-level summaries.

### Utility

Need the color legend on the command line?

```
bbx colors --colors label_colors.json
```

Outputs the label → hex mapping that the plotting scripts share.

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
  rbxplore.py     # Rulebook orientation/MI/tree exploration
  gridmaps.py     # Spatial heatmaps & per-cell metrics
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
