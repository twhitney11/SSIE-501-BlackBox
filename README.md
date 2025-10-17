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

The workflow is broken into modular steps:

### 1. Data collection

Scrape states directly from the `BlackBox.php` interface.

```
bbx collect --base-url http://localhost/BlackBox.php --steps 500 --runs 5 --seed 42 --out data
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
Add `--window start:end` (or `--window last:N`) to focus on a slice of each run.

### 3. Analysis

Check reproducibility, test model hypotheses, train classifiers.

```
bbx analyze --runs data/run_000 --radius 1 \
  --tests shape,conflicts,permutation,totalistic \
  --period-scan edge,corner_edge \
  --train-clf --clf-period 2 \
  --out reports
```

Use `--window` to analyze only part of each run (works with multiple runs as well).

This can generate:

* **Conflicts examples** — where the same neighborhood produces different outcomes.
* **Permutation test** — shuffle label IDs to test if labels matter.
* **Totalistic vs positional** — test whether rules depend only on *counts* of neighbors vs their *positions*.
* **Period scan** — see if the system evolves with a hidden cycle (e.g. 2-step oscillation).
* **Classifier training** — logistic regression on neighborhood + features, used to simulate forward.
* **Shape diagnostics** — optional flags such as `--do-ring`, `--do-cycle`, `--do-torus`, `--autocorr`, and `--near-cycle` add ring-radius tracking, cycle detection, torus experiments, and correlation scans. Tune near-cycle behaviour with `--near-cycle-maxlag` and `--near-cycle-eps`.
* **Perturb & simulate** — `--perturb ...` and `--simulate ...` pair with `--sim-steps`, `--sim-save-every`, `--sim-png`, `--sim-colors`, etc. to stress the inferred classifier and export rollout JSON/PNGs.
* **Rulebooks** — `--build-rulebook` (with `--rb-k`, `--rb-split`, `--rb-name`) exports exact phase rulebooks; `--simulate-rulebook ... --rb-steps` replays them with optional PNG snapshots via `--rb-png`, `--rb-save-every`, `--rb-colors`, and fallback control through `--rb-fallback`.

All artifacts are saved to `reports/` (JSON, CSV, PNGs).

### 4. Fractions reports

Track black, white, and colored label fractions over time.

```
bbx fractions --runs data/run_000 --out reports --region-labels gru,mex,cyc
```

Generates:

* Raw and smoothed fraction CSVs.
* Plots of each label’s frequency.
* Stacked black/white/other trajectories.
* Regional breakdown (edge vs interior) for the labels you specify with `--region-labels` (default `gru,mex`).
* Change curves and black-front progression.
* Works with `--window start:end` to restrict the time slice.

### 5. Goodness-of-fit (locality testing)

Assess how well a local neighborhood model explains the data and how sensitive it is to locality assumptions.

```
bbx gof --train-run data/run_000 --train-window 0:2000 \
  --test-runs data/run_001 data/run_002 --test-window 2000:4000 \
  --radius 1 --model logistic --feature-mode local --permutations random,ring,inner --out reports/gof
```

Outputs accuracy/log-loss/Brier scores, log-likelihood, AIC/BIC, confusion matrices, calibration curves, and optional permutation diagnostics (`random`, `inner`, `ring`, `target`).
Models: `logistic`, `rule`, `markov` (center Markov baseline), `knn` (global k-NN cheat). Use `--feature-mode local|center|global` and `--knn-k` to compare CA vs non-CA hypotheses.

### 6. Rulebook exploration

Inspect exported rulebooks for symmetry and dependence structure.

```
bbx rbxplore --rb reports/rulebooks.json --do orient,center,mi,tree --out reports/rbxplore
```

Produces CSVs covering orientation invariance, center dependence, positional mutual information, and small decision-tree summaries. Configure the tree depth/phase (`--tree-depth`, `--tree-phase`) to control the simplification.

### 7. Spatial grid heatmaps

Quantify per-cell behaviour across long runs.

```
bbx gridmaps --runs data/run_000 data/run_001 --out reports/gridmaps --win last:200 --k 2
```

Generates heatmaps/CSVs for change rate, per-label occupation, entropy, conditional entropy, phase flip rate, and time-to-black. Use `--phase` to isolate a particular phase slice and `--win` to focus on time windows.
Pass `--sequence` to also render a montage of states from the first run (`--sequence-steps` accepts comma values or `auto[:N]`).

### 8. Regionize walls

Derive wall/inside/outside masks for a dominant label over a time window.

```
bbx regionize --run data/run_000 --window 3500:5000 --label gru --wall-thickness 2 --out reports/regionize
```

Outputs occupancy maps, wall/inside/outside masks (plus near-wall bands), and per-region summary stats (cell count, entropy, conditional entropy, flip rate, change rate).
PNG visuals are emitted alongside the CSVs for quick inspection: occupancy heatmap with the wall contour, color-coded region overlay, metric heatmaps, and bar charts comparing region-level summaries.
`region_metrics.csv` aggregates bulk/near/wall regions with neighborhood entropy and conditional entropy values.

### 9. Animated runs

Generate an animated GIF for a run (or windowed segment).

```
bbx animate --run data/run_000 --out reports/run_000.gif --window 0:500 --stride 2 --fps 12.5 --scale 4
```

Controls include per-frame duration or FPS, up-scaling, frame stride, loop count, and time window selection.

### Utility

Need the color legend on the command line?

```
bbx colors --colors label_colors.json
```

Outputs the label → hex mapping that the plotting scripts share.

### 10. Cross-run aggregation

Summarise central tendency and variation across multiple runs/windows.

```
bbx aggregate --runs data/run_000 data/run_001 data/run_002 --window 1000:2000 --labels gru,mex,cyc --out reports/aggregate
```

Generates fraction ribbons (spaghetti + mean ± 95% CI), change-count summaries, and per-cell mean/variance heatmaps for the chosen labels.

### 11. Neighborhood entropy

Examine neighborhood entropy and conditional entropy across runs/windows.

```
bbx entropy --runs data/run_000 data/run_001 --window 1000:2000 --radius 1 --regions all,interior,edge --out reports/entropy
```

Produces per-step neighborhood entropy curves (with mean ± 95% CI) and conditional entropies by region.

### 12. Mutual information & correlation

Measure pairwise mutual information/correlation versus distance.

```
bbx mi --runs data/run_000 data/run_001 --window 2000:4000 --samples 2000 --out reports/mi
```

Saves sampled pair metrics and MI/correlation vs distance summaries/plots.

### 13. Stationarity / cross-run generalization

Check how model performance drifts across windows or runs.

```
bbx stationarity --train-run data/run_000 --train-window 0:2000 \
  --test-runs data/run_000 data/run_001 --test-window 2000:6000 \
  --radius 1 --feature-mode local --model logistic --segments 6 --out reports/stationarity
```

Outputs per-segment accuracy/log-loss (with drops relative to training) and a summary plot.
## Command reference

Below is a quick reference of the available subcommands and their key flags.

| Command | Description | Key options |
|---------|-------------|-------------|
| `collect` | Capture one or many runs from the PHP interface. | `--base-url`, `--steps`, `--runs`, `--out`, `--run-prefix`, `--seed`, `--sleep-ms`, `--no-reset` |
| `process` | Print coverage/conflict summaries. | `--runs`, `--radius`, `--window` |
| `analyze` | Core analysis suite (rule learning, diagnostics, simulation). | `--runs`, `--radius`, `--window`, `--tests`, `--period-scan`, `--train-clf`, `--do-*`, `--near-cycle`, `--near-cycle-maxlag`, `--near-cycle-eps`, `--perturb`, `--simulate`, `--sim-steps`, `--sim-save-every`, `--sim-png`, `--sim-colors`, `--build-rulebook`, `--rb-k`, `--rb-split`, `--rb-name`, `--rb-fallback`, `--rb-steps`, `--rb-png`, `--rb-save-every`, `--rb-colors` |
| `fractions` | Fractions, regional breakdowns, change curves. | `--runs`, `--out`, `--colors`, `--smooth`, `--region-labels`, `--window` |
| `gof` | Goodness-of-fit / locality testing. | `--train-run`, `--train-window`, `--test-runs`, `--test-window`, `--radius`, `--feature-mode {local,center,global}`, `--model {logistic,rule,markov,knn}`, `--knn-k`, `--max-samples`, `--permutations`, `--seed`, `--out` |
| `rbxplore` | Rulebook symmetry/MI/tree exploration. | `--rb`, `--do`, `--tree-phase`, `--tree-region`, `--tree-depth`, `--out` |
| `gridmaps` | Per-cell heatmaps and CSVs. | `--runs`, `--out`, `--win`, `--phase`, `--k`, `--sequence`, `--sequence-steps`, `--sequence-file`, `--colors`, `--sequence-show` |
| `regionize` | Derive wall/inside/outside masks and stats. | `--run`, `--out`, `--window`, `--label`, `--threshold`, `--wall-thickness`, `--flip-period` |
| `animate` | Render animated GIFs. | `--run`, `--out`, `--window`, `--stride`, `--scale`, `--duration`, `--fps`, `--loop`, `--colors` |
| `aggregate` | Aggregate fractions/change metrics across runs. | `--runs`, `--window`, `--labels`, `--colors`, `--out` |
| `entropy` | Neighborhood & conditional entropy per region. | `--runs`, `--window`, `--radius`, `--regions`, `--out` |
| `mi` | Pairwise mutual information & correlation vs distance. | `--runs`, `--window`, `--samples`, `--seed`, `--out` |
| `stationarity` | Stationarity / cross-run generalization. | `--train-run`, `--train-window`, `--test-runs`, `--test-window`, `--radius`, `--feature-mode`, `--model`, `--segments`, `--max-samples`, `--seed`, `--out` |
| `colors` | Print label→color mapping. | `--colors` |

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
