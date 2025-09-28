# gridmaps.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .process import load_run_encoded  # your helper
# If you want label colors: from .viz import load_label_colors

BLACK, WHITE = "gru", "mex"

def stack_runs(run_dirs):
    """Load multiple runs (same H×W). Return states_per_run(list[list[np.ndarray]]), labels."""
    runs = []
    labels_ref = None
    for rd in run_dirs:
        states, labels, _ = load_run_encoded(rd)
        if labels_ref is None:
            labels_ref = labels
        else:
            # ensure same label ordering (build a remap if needed)
            assert labels == labels_ref, "Label order mismatch across runs."
        runs.append(states)
    return runs, labels_ref

def per_cell_change_rate(states):
    """states: list[np.ndarray] for a single run. Returns H×W change fraction."""
    H, W = states[0].shape
    diffs = np.zeros((H, W), dtype=np.int32)
    for t in range(len(states)-1):
        diffs += (states[t] != states[t+1])
    return diffs / max(1, (len(states)-1))

def per_cell_label_fraction(states, label_idx):
    """Fraction of time each cell equals label_idx."""
    H, W = states[0].shape
    counts = np.zeros((H, W), dtype=np.int32)
    for S in states:
        counts += (S == label_idx)
    return counts / len(states)

def per_cell_entropy(states, L):
    """Shannon entropy (bits) of label distribution at each cell."""
    H, W = states[0].shape
    counts = np.zeros((H, W, L), dtype=np.int32)
    for S in states:
        # bincount per cell:
        for i in range(H):
            # vectorize along a row for speed
            row = S[i]
            counts[i, np.arange(W), row] += 1
    p = counts / counts.sum(axis=2, keepdims=True).clip(min=1)
    # entropy = -sum p log2 p
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(p * np.log2(p, where=(p>0))).sum(axis=2)
    return ent

def per_cell_phase_flip_rate(states, k=2):
    """How often a cell differs between consecutive phases (t and t+1 when (t%k)!=((t+1)%k))."""
    H, W = states[0].shape
    flips = np.zeros((H, W), dtype=np.int32)
    denom = 0
    for t in range(len(states)-1):
        if (t % k) != ((t+1) % k):
            flips += (states[t] != states[t+1])
            denom += 1
    return flips / max(1, denom)

def per_cell_time_to_black(states, black_idx):
    """First step when a cell becomes black; NaN if never."""
    H, W = states[0].shape
    ttb = np.full((H, W), np.nan, dtype=np.float32)
    seen = np.zeros((H, W), dtype=bool)
    for t, S in enumerate(states):
        mask = (S == black_idx) & (~seen)
        ttb[mask] = t
        seen |= mask
    return ttb

def _save_heatmap(arr, out_png: Path, title: str, vmin=None, vmax=None, cmap="viridis"):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6,5))
    im = plt.imshow(arr, interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, shrink=0.8)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[saved] {out_png}")

def _save_csv(arr, out_csv: Path, name: str):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    H, W = arr.shape
    rows = []
    for i in range(H):
        for j in range(W):
            rows.append({"i": i, "j": j, name: float(arr[i, j])})
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")

def per_cell_conditional_entropy(states, L):
    """H(Y_{t+1} | Y_t) per cell based on empirical transitions at that cell."""
    H, W = states[0].shape
    trans = np.zeros((H, W, L, L), dtype=np.int32)
    for t in range(len(states)-1):
        S, T = states[t], states[t+1]
        for i in range(H):
            for j in range(W):
                a, b = int(S[i,j]), int(T[i,j])
                trans[i, j, a, b] += 1
    # P(b|a) and conditional entropy sum_a p(a) H(b|a)
    counts_a = trans.sum(axis=3, keepdims=True)  # sum over b
    p_b_given_a = np.divide(trans, counts_a, out=np.zeros_like(trans, dtype=np.float64), where=(counts_a>0))
    with np.errstate(divide="ignore", invalid="ignore"):
        Hb_given_a = -(p_b_given_a * np.log2(p_b_given_a, where=(p_b_given_a>0))).sum(axis=3)  # H(b|a)
    p_a = counts_a[...,0].astype(np.float64)  # drop last dim
    p_a /= np.clip(p_a.sum(axis=2, keepdims=True), 1, None)
    H_cond = (p_a * Hb_given_a).sum(axis=2)  # weighted by p(a)
    return H_cond

def per_cell_change_rate_by_phase(states, k=2):
    H, W = states[0].shape
    counts = np.zeros((k, H, W), dtype=np.int32)
    totals = np.zeros(k, dtype=np.int32)
    for t in range(len(states)-1):
        phase = t % k
        counts[phase] += (states[t] != states[t+1])
        totals[phase] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        rates = np.stack([counts[p] / max(1, totals[p]) for p in range(k)], axis=0)
    return rates  # shape (k,H,W)

def _percentile_clip(arr, lo=5, hi=95):
    low, high = np.percentile(arr, [lo, hi])
    # if all values equal, fall back to raw min/max to avoid zero range
    if np.isclose(low, high):
        low, high = float(np.min(arr)), float(np.max(arr))
    return low, high

def _save_full_set(arr, outstem: Path, title: str, cmap="viridis",
                   clip=True, dev=True, force_01=False):
    """
    Save three variants:
      - raw: full min..max
      - clipped: percentile [5,95] (if clip=True)
      - deviation: arr - mean, symmetric range (if dev=True)
    If force_01: raw/clipped use vmin=0, vmax=1.
    """
    # RAW
    if force_01:
        vmin_raw, vmax_raw = 0.0, 1.0
    else:
        vmin_raw, vmax_raw = float(np.min(arr)), float(np.max(arr))
    _save_heatmap(arr, outstem.with_suffix(".png"), f"{title} (raw)", vmin=vmin_raw, vmax=vmax_raw, cmap=cmap)

    # CLIPPED
    if clip:
        if force_01:
            vmin_c, vmax_c = 0.0, 1.0
        else:
            vmin_c, vmax_c = _percentile_clip(arr, 5, 95)
        _save_heatmap(arr, outstem.with_name(outstem.stem + "_clipped.png"),
                      f"{title} (clipped 5–95%)", vmin=vmin_c, vmax=vmax_c, cmap=cmap)

    # DEVIATION
    if dev:
        dev_arr = arr - float(np.mean(arr))
        vmaxd = float(np.max(np.abs(dev_arr))) or 1.0
        _save_heatmap(dev_arr, outstem.with_name(outstem.stem + "_dev.png"),
                      f"{title} (deviation from mean)",
                      vmin=-vmaxd, vmax=vmaxd, cmap="bwr")  # symmetric diverging
    # CSV for the base array
    _save_csv(arr, outstem.with_suffix(".csv"), outstem.stem)

def run_gridmaps(run_dirs, outdir):
    runs, labels = stack_runs(run_dirs)
    L = len(labels)
    black_idx = labels.index(BLACK) if BLACK in labels else 0
    white_idx = labels.index(WHITE) if WHITE in labels else 1

    # Aggregate across runs by averaging the per-run maps
    def avg_over_runs(func, *args, **kwargs):
        mats = []
        for states in runs:
            mats.append(func(states, *args, **kwargs))
        return np.mean(mats, axis=0)

    outdir = Path(outdir)

    # 1) Change rate
    chg = avg_over_runs(per_cell_change_rate)
    _save_full_set(chg, outdir / "grid_change_rate", "Per-cell change rate",
                   cmap="coolwarm", clip=True, dev=True, force_01=True)

    # 2) Black / White occupancy (fractions in [0,1])
    frac_black = avg_over_runs(per_cell_label_fraction, black_idx)
    frac_white = avg_over_runs(per_cell_label_fraction, white_idx)
    _save_full_set(frac_black, outdir / "grid_frac_black",
                   f"Fraction black ({BLACK})", cmap="magma",
                   clip=True, dev=True, force_01=True)
    _save_full_set(frac_white, outdir / "grid_frac_white",
                   f"Fraction white ({WHITE})", cmap="bone",
                   clip=True, dev=True, force_01=True)

    # 3) Entropy (bits). Range is data-driven; don’t force 0–1.
    ent = avg_over_runs(per_cell_entropy, L)
    _save_full_set(ent, outdir / "grid_entropy_bits",
                   "Per-cell label entropy (bits)", cmap="viridis",
                   clip=True, dev=True, force_01=False)

    # 4) Phase flip rate (k=2), in [0,1]
    flips = avg_over_runs(per_cell_phase_flip_rate, 2)
    _save_full_set(flips, outdir / "grid_phase_flip_rate",
                   "Phase flip rate (k=2)", cmap="coolwarm",
                   clip=True, dev=True, force_01=True)

    # 5) Time to black (first run; not averaged)
    ttb = per_cell_time_to_black(runs[0], black_idx)
    _save_full_set(ttb, outdir / "grid_time_to_black",
                   "Time to black (first run)", cmap="plasma",
                   clip=True, dev=True, force_01=False)

    # 6) Conditional entropy H(Y_{t+1}|Y_t) (bits)
    Hc = avg_over_runs(per_cell_conditional_entropy, L)
    _save_full_set(Hc, outdir / "grid_cond_entropy_bits",
                   "Conditional entropy H(Y_{t+1}|Y_t) (bits)", cmap="viridis",
                   clip=True, dev=True, force_01=False)

    # 7) Phase-split change maps (first run for clarity)
    rates = per_cell_change_rate_by_phase(runs[0], 2)  # shape (2,H,W)
    for p in range(rates.shape[0]):
        _save_full_set(rates[p], outdir / f"grid_change_rate_phase{p}",
                       f"Per-cell change rate (phase {p})", cmap="coolwarm",
                       clip=True, dev=True, force_01=True)

    print("[gridmaps] done.")

    
# --- CLI ---
def main():
    ap = argparse.ArgumentParser(description="Full-grid (20×20) spatial heatmaps.")
    ap.add_argument("--runs", nargs="+", required=True, help="Run directories (data/run_000 ...)")
    ap.add_argument("--out", default="reports", help="Output directory")
    args = ap.parse_args()
    run_gridmaps(args.runs, args.out)

if __name__ == "__main__":
    main()
