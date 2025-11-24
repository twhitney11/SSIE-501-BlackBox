# New Graphs Added to bbx Project

**Date:** November 23, 2025  
**Added by:** Abdullah

This document describes the three new visualization graphs that have been added to the project.

---

## Summary

Three new plotting functions have been added to enhance the visualization capabilities of the bbx toolkit:

1. **Confusion Matrix Heatmap** - Visualizes classification performance
2. **Calibration Curve Plot** - Assesses model probability calibration
3. **Period Scan Visualization** - Identifies periodic behavior in the CA

All three graphs are automatically generated when running the respective commands.

---

## Graph 1: Confusion Matrix Heatmap

### File Location
`src/bbx/gof.py`

### Function Name
```python
plot_confusion_matrix_heatmap(conf: np.ndarray, labels: Sequence[str], 
                               out_path: Path, title: str = "Confusion Matrix")
```

### Location in File
- **Lines:** 243-279
- **Added by:** Abdullah on 11/23/25
- **Description:** Visualizes the confusion matrix as a color-coded heatmap showing the classification performance of the model. The heatmap displays how many times each actual label (rows) was predicted as each label (columns). Diagonal elements represent correct predictions, while off-diagonal elements show misclassifications. The plot includes count annotations on each cell and uses a blue color scale where darker colors indicate higher counts.

### Integration Point
**File:** `src/bbx/gof.py`  
**Function:** `run_gof()`  
**Line:** ~464 (after confusion matrix CSV is saved)

```python
# Automatically called after:
conf = confusion_matrix(y_test, preds, len(labels))
# ... CSV saved ...
plot_confusion_matrix_heatmap(conf, labels, ...)
```

### When It's Generated
Automatically generated when running:
```bash
bbx gof --train-run data/run_000 --test-runs data/run_001 --out reports/gof
```

### Output File
`reports/gof/confusion_<run_name>_<window>.png`

### Graph Features
- Color-coded heatmap (blue scale)
- Count annotations on each cell
- Row labels = Actual labels
- Column labels = Predicted labels
- Diagonal = correct predictions
- Off-diagonal = misclassifications

---

## Graph 2: Calibration Curve Plot

### File Location
`src/bbx/gof.py`

### Function Name
```python
plot_calibration_curve(calib_df: pd.DataFrame, out_path: Path, 
                       title: str = "Calibration Curve")
```

### Location in File
- **Lines:** 325-344
- **Added by:** Abdullah on 11/23/25
- **Description:** Creates a calibration plot that assesses how well the model's predicted probabilities match the observed frequencies. The plot shows the relationship between mean predicted probability (x-axis) and observed rate (y-axis) across probability bins. A perfectly calibrated model would follow the diagonal line (y=x), where predicted probabilities equal observed rates. Deviations from this line indicate overconfidence (above diagonal) or underconfidence (below diagonal) in the model's predictions. This is a key diagnostic for probabilistic classifiers.

### Integration Point
**File:** `src/bbx/gof.py`  
**Function:** `run_gof()`  
**Line:** ~480 (after calibration CSV is saved)

```python
# Automatically called after:
calib = calibration_curve(probs, y_test)
# ... CSV saved ...
plot_calibration_curve(calib, ...)
```

### When It's Generated
Automatically generated when running:
```bash
bbx gof --train-run data/run_000 --test-runs data/run_001 --out reports/gof
```

### Output File
`reports/gof/calibration_<run_name>_<window>.png`

### Graph Features
- Line plot: predicted probabilities vs observed rates
- Diagonal reference line (perfect calibration)
- Model performance line
- Grid and legend
- X/Y limits: 0 to 1

---

## Graph 3: Period Scan Visualization

### File Location
`src/bbx/analyze.py`

### Function Name
```python
plot_period_scan(rows, out_path: Path, title: str = "Period Scan - Conflicts vs k")
```

### Location in File
- **Lines:** 409-517
- **Added by:** Abdullah on 11/23/25
- **Description:** Visualizes the period scan results showing conflicts vs period k (time period). The plot helps identify periodic behavior in the cellular automaton by showing which period values (k) result in zero or minimal conflicts. When conflicts drop to zero at a specific k, it indicates the system follows a k-step periodic rule. The plot can show multiple regions (interior, edge, corner) or phases, making it easy to spot which periods exhibit consistent deterministic behavior. Lower conflict values indicate more predictable, periodic dynamics.

### Integration Point
**File:** `src/bbx/analyze.py`  
**Function:** `run_analysis()`  
**Line:** ~1118 (after period scan CSV is saved)

```python
# Automatically called after:
rows = period_scan(per_run, r=radius, max_k=max_k_local, split=split_name, ...)
# ... CSV saved ...
plot_period_scan(rows, ...)
```

### When It's Generated
Automatically generated when running:
```bash
bbx analyze --runs data/run_000 --period-scan edge,corner_edge --out reports
```

### Output File
`reports/period_scan_<mode>.png`  
(e.g., `period_scan_edge.png`, `period_scan_corner_edge.png`)

### Graph Features
- **Two-panel layout:**
  - **Left panel:** Minimum conflicts vs period k
    - Highlights zero-conflict periods with green vertical lines
    - Text box listing all zero-conflict k values
    - Green dashed line at y=0 (perfect periodicity)
  - **Right panel:** Contextual view
    - If multiple regions: Shows conflicts by region (interior, edge, corner)
    - If single region with few phases: Shows conflicts by phase
    - Otherwise: Shows average rule size vs k

---

## Code Attribution

All three functions include attribution comments at the top:

```python
# Abdullah Added on 11/23/25
# [Graph Name]
# [Detailed description of what the plot shows and how to interpret it]
```

---

## Dependencies

All graphs use standard libraries already in the project:
- `matplotlib.pyplot` - For plotting
- `numpy` - For numerical operations
- `pandas` - For data handling (period scan and calibration)

No additional dependencies were added.

---

## Testing

Test scripts were created to verify functionality:
- `test_new_plots.py` - Tested confusion matrix and calibration curve
- `test_period_scan_plot.py` - Tested period scan plot
- `test_all_3_graphs.py` - Comprehensive test of all three graphs

All tests passed successfully. Test files have been removed from the project.

---

## Usage Examples

### Generate Confusion Matrix and Calibration Curve
```bash
bbx gof --train-run data/run_000 \
        --test-runs data/run_001 \
        --model logistic \
        --feature-mode local \
        --out reports/gof
```

**Outputs:**
- `reports/gof/confusion_run_001_full.png`
- `reports/gof/calibration_run_001_full.png`

### Generate Period Scan Plot
```bash
bbx analyze --runs data/run_000 \
            --period-scan edge,corner_edge \
            --radius 1 \
            --out reports
```

**Outputs:**
- `reports/period_scan_edge.png`
- `reports/period_scan_corner_edge.png`

---

## File Modifications Summary

### Modified Files

1. **src/bbx/gof.py**
   - Added: `plot_confusion_matrix_heatmap()` function (lines 243-279)
   - Added: `plot_calibration_curve()` function (lines 325-344)
   - Modified: `run_gof()` function to call both plotting functions

2. **src/bbx/analyze.py**
   - Added: `plot_period_scan()` function (lines 409-517)
   - Modified: `run_analysis()` function to call period scan plotting

### Total Lines Added
- Confusion Matrix: ~37 lines
- Calibration Curve: ~20 lines
- Period Scan: ~109 lines
- Integration code: ~6 lines
- **Total: ~172 lines of new code**

---

## Graph Output Specifications

All graphs follow consistent formatting:
- **Resolution:** 150 DPI
- **Format:** PNG
- **Layout:** Tight layout with proper spacing
- **File naming:** Descriptive names with run/window identifiers
- **Error handling:** Graceful handling of empty/missing data

---

## Notes

- All graphs are automatically generated - no additional flags needed
- Graphs are saved alongside their corresponding CSV data files
- The plotting functions include error handling for edge cases
- All functions follow the existing code style and patterns in the project

