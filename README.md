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
* **Scoped analysis** — limit calculations to a mask with `--scope-mask path:is_region` (CSV of 0/1) or a region-label file via `--scope-region path:region`. Add `--scope-invert` to flip the mask. When a scope is active the analyzer also saves `label_histogram_<mask>.png` for that region. This pairs well with `bbx regionize` or manual masks (see below).

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
| `analyze` | Core analysis suite (rule learning, diagnostics, simulation). | `--runs`, `--radius`, `--window`, `--tests`, `--period-scan`, `--train-clf`, `--do-*`, `--near-cycle`, `--near-cycle-maxlag`, `--near-cycle-eps`, `--perturb`, `--simulate`, `--sim-steps`, `--sim-save-every`, `--sim-png`, `--sim-colors`, `--build-rulebook`, `--rb-k`, `--rb-split`, `--rb-name`, `--rb-fallback`, `--rb-steps`, `--rb-png`, `--rb-save-every`, `--rb-colors`, `--scope-mask(path[:column])`, `--scope-region(path:region)`, `--scope-invert`, `--region-models` |
| `fractions` | Fractions, regional breakdowns, change curves. | `--runs`, `--out`, `--colors`, `--smooth`, `--region-labels`, `--window` |
| `gof` | Goodness-of-fit / locality testing. | `--train-run`, `--train-window`, `--test-runs`, `--test-window`, `--radius`, `--feature-mode {local,center,global}`, `--model {logistic,rule,markov,knn}`, `--knn-k`, `--max-samples`, `--permutations`, `--seed`, `--out` |
| `rbxplore` | Rulebook symmetry/MI/tree exploration. | `--rb`, `--do`, `--tree-phase`, `--tree-region`, `--tree-depth`, `--out` |
| `gridmaps` | Per-cell heatmaps and CSVs. | `--runs`, `--out`, `--win`, `--phase`, `--k`, `--sequence`, `--sequence-steps`, `--sequence-file`, `--colors`, `--sequence-show` |
| `regionize` | Derive wall/inside/outside masks and stats. | `--run`, `--out`, `--window`, `--label`, `--threshold`, `--wall-thickness`, `--flip-period`, `--mask-config` |
| `animate` | Render animated GIFs. | `--run`, `--out`, `--window`, `--stride`, `--scale`, `--duration`, `--fps`, `--loop`, `--colors` |
| `aggregate` | Aggregate fractions/change metrics across runs. | `--runs`, `--window`, `--labels`, `--colors`, `--out` |
| `entropy` | Neighborhood & conditional entropy per region. | `--runs`, `--window`, `--radius`, `--regions`, `--out` |
| `mi` | Pairwise mutual information & correlation vs distance. | `--runs`, `--window`, `--samples`, `--seed`, `--out` |
| `stationarity` | Stationarity / cross-run generalization. | `--train-run`, `--train-window`, `--test-runs`, `--test-window`, `--radius`, `--feature-mode`, `--model`, `--segments`, `--max-samples`, `--seed`, `--out` |
| `colors` | Print label→color mapping. | `--colors` |
| `masks` | Build manual region masks from a JSON config. | `--config`, `--out`, `--run`, `--window`, `--allow-overlap` |
| `metrics` | Conditional entropy & mutual information per region. | `--runs`, `--config`, `--window`, `--top-mi`, `--mi-lag`, `--fit-labels`, `--fit-models`, `--out` |
| `exportcsv` | Flatten runs into a single CSV file. | `--runs`, `--run-dir`, `--run-glob`, `--out`, `--window`, `--include-encoded` |

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
  maskgen.py      # Manual region mask generator
  rbxplore.py     # Rulebook orientation/MI/tree exploration
  gridmaps.py     # Spatial heatmaps & per-cell metrics
  viz.py          # Plotting helpers (color maps, sequences)

data/             # Saved runs (step_XXXX.json)
reports/          # Generated CSVs, JSONs, PNGs
label_colors.json # Mapping from labels → colors
```

## Manual regions

Use `bbx masks` to define arbitrary regions without re-running `regionize`. Supply a JSON config describing the grid and the shapes that compose each region, then consume the CSV masks with `--scope-mask`/`--scope-region` when running `bbx analyze`.

Example config (`manual_regions.json`):

```json
{
  "grid": { "height": 20, "width": 20 },
  "regions": [
    {
      "name": "interior",
      "include": [
        { "type": "rect", "top": 4, "left": 3, "height": 12, "width": 14 }
      ]
    },
    {
      "name": "wall",
      "include": [
        { "type": "rect", "top": 2, "left": 2, "height": 16, "width": 18 }
      ],
      "exclude": [
        { "type": "rect", "top": 3, "left": 3, "height": 14, "width": 16 }
      ]
    }
  ]
}
```

Generate masks:

```
bbx masks --config manual_regions.json --out reports/manual_masks
```

This writes `manual_regions_interior.csv`, `manual_regions_wall.csv`, and a `manual_regions_region_labels.csv` file. Analyze a region by referencing the mask:

```
bbx analyze --runs data/run_000 --radius 2 \
  --scope-mask reports/manual_masks/manual_regions_wall.csv:is_wall \
  --out reports/an_wall
```

Notes:

- Omit `grid` if you prefer to infer the shape from a run: `bbx masks --config manual_regions.json --run data/run_000 --window last:2000` will read the lattice size from the specified run/window.
- Supported shapes include `rect` (`top`, `left`, `height`, `width`), `disk`/`circle` (`center_i`, `center_j`, `radius`), and explicit `points` (`[[i, j], ...]`). Combine multiple shapes in `include`, remove areas with `exclude`, or reuse previously declared regions with `inherit`, `include_regions`, and `exclude_regions`.
- Region overlaps are rejected unless you opt-in. Set `"allow_overlap": true` in the config or pass `--allow-overlap` when running `bbx masks`.
- Use the same mask config to train region-specific simulators: `bbx analyze --runs data/run_000 --train-clf --region-models config/masks.json --out reports/an_region`. Each region gets its own classifier and mask, and simulation automatically applies the right model per cell.
- Quantify dynamics inside each mask with `bbx metrics --runs data/run_000 data/run_001 --config config/masks.json --window last:2000 --out reports/metrics`. The command saves `region_metrics_summary.csv`, per-region `*_top_mi_pairs.csv`, and companion plots (`stable_cells.png`, `stable_cells_regions.png`, `self_transition_fraction.png`, `cond_entropy*.png`, `transitions_vs_changes.png`, `<region>_top_mi_pairs.png`, `region_mi_heatmap.png`, `<region>_transition_heatmap.png`). Add `--fit-labels gru,mex --fit-models linear,exponential,power` to fit label-fraction series and produce `label_fraction_fits.csv` plus `label_fit_<label>.png` overlays. Include `--combos pair` to generate adjacency snapshots (`<region>_pairwise_counts.csv`, `<region>_pairwise_heatmap.png`) with chi-square scores against independent baselines.

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
