# gridmaps.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .process import load_run_encoded  # your helper
from .generate_grid import sequence_panel
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

def _parse_window(win_str, T):
    # win_str: "", "start:end", or "last:N"
    if not win_str:
        return 0, T
    if win_str.startswith("last:"):
        N = int(win_str.split(":")[1])
        return max(0, T-N), T
    # "start:end"
    s, e = win_str.split(":")
    s = int(s) if s else 0
    e = int(e) if e else T
    s = max(0, min(s, T))
    e = max(0, min(e, T))
    if e <= s: e = min(T, s+1)
    return s, e

def _slice_states(states, start, end, phase=-1, k=2):
    # returns a list of states restricted to time window and (optional) phase
    idxs = range(start, end)
    if phase >= 0:
        idxs = [t for t in idxs if (t % k) == phase]
    return [states[t] for t in idxs]


def run_gridmaps(run_dirs, outdir, win="", phase=-1, k=2,
                 sequence=False, sequence_steps="", sequence_file="",
                 colors_path="label_colors.json", sequence_show=False):
    runs, labels = stack_runs(run_dirs)  # your existing helper
    if not runs:
        raise ValueError("No runs supplied to gridmaps.")
    L = len(labels)
    black_idx = labels.index(BLACK) if BLACK in labels else 0
    white_idx = labels.index(WHITE) if WHITE in labels else 1

    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    def avg_over_runs(func, *args, **kwargs):
        mats = []
        for states in runs:
            T = len(states)
            s, e = _parse_window(win, T)
            states_win = _slice_states(states, s, e, phase=phase, k=k)
            if len(states_win) == 0:
                continue
            mats.append(func(states_win, *args, **kwargs))
        return np.mean(mats, axis=0) if mats else None

    # Choose sensible fixed scales where appropriate
    # change rate: [0,1], frac_*: [0,1], entropy: [0, log2(L)], flip rate: [0,1]
    # 1) Change rate
    chg = avg_over_runs(per_cell_change_rate)
    if chg is not None:
        _save_csv(chg,  outdir/"grid_change_rate.csv", "change_rate")
        _save_heatmap(chg, outdir/"grid_change_rate.png",
                      f"Per-cell change rate (win={win}, phase={phase})",
                      vmin=0.0, vmax=0.0020)

    # 2) Black / White occupancy
    frac_black = avg_over_runs(per_cell_label_fraction, black_idx)
    frac_white = avg_over_runs(per_cell_label_fraction, white_idx)
    if frac_black is not None:
        _save_csv(frac_black, outdir/"grid_frac_black.csv", "frac_black")
        _save_heatmap(frac_black, outdir/"grid_frac_black.png",
                      f"Fraction black ({BLACK}) (win={win}, phase={phase})",
                      vmin=0.05, vmax=0.30)

    if frac_white is not None:
        _save_csv(frac_white, outdir/"grid_frac_white.csv", "frac_white")
        _save_heatmap(frac_white, outdir/"grid_frac_white.png",
                      f"Fraction white ({WHITE}) (win={win}, phase={phase})",
                      vmin=0.00, vmax=0.6)

    # 2b) Contrast map
    if frac_black is not None and frac_white is not None:
        contrast = frac_black - frac_white
        _save_csv(contrast, outdir/"grid_black_minus_white.csv", "black_minus_white")
        _save_heatmap(contrast, outdir/"grid_black_minus_white.png",
                      f"P(black) − P(white) (win={win}, phase={phase})",
                      vmin=-0.07, vmax=1, cmap="coolwarm")

    # 3) Entropy
    ent = avg_over_runs(per_cell_entropy, L)
    if ent is not None:
        _save_csv(ent, outdir/"grid_entropy_bits.csv", "entropy_bits")
        _save_heatmap(ent, outdir/"grid_entropy_bits.png",
                      f"Per-cell label entropy (bits) (win={win}, phase={phase})",
                      vmin=0.00, vmax=3.0)

    # 4) Conditional entropy
    Hc = avg_over_runs(per_cell_conditional_entropy, L)
    if Hc is not None:
        _save_csv(Hc,  outdir/"grid_cond_entropy_bits.csv", "H_next_given_current_bits")
        _save_heatmap(Hc, outdir/"grid_cond_entropy_bits.png",
                      f"H(Yₜ₊₁|Yₜ) (bits) (win={win}, phase={phase})",
                      vmin=0.0, vmax=0.12)

    # 5) Phase flip rate (if you want it phase-independent, compute with phase=-1; for clarity keep k=2)
    flips = avg_over_runs(per_cell_phase_flip_rate, 2)
    if flips is not None:
        _save_csv(flips, outdir/"grid_phase_flip_rate.csv", "phase_flip_rate")
        _save_heatmap(flips, outdir/"grid_phase_flip_rate.png",
                      f"Phase flip rate (k=2) (win={win}, phase={phase})",
                      vmin=0.0, vmax=0.015, cmap="coolwarm")

    # 6) Time to black (use first run + same window)
    states0 = runs[0]
    s0, e0 = _parse_window(win, len(states0))
    states0_win = _slice_states(states0, s0, e0, phase=phase, k=k)
    if states0_win:
        ttb = per_cell_time_to_black(states0_win, black_idx)
        _save_csv(ttb, outdir/"grid_time_to_black.csv", "time_to_black")
        _save_heatmap(ttb, outdir/"grid_time_to_black.png",
                      f"Time to black (win={win}, phase={phase})",
                      cmap="plasma")

    if sequence:
        sequence_name = sequence_file or f"{Path(run_dirs[0]).name}_sequence.png"
        seq_path = outdir / sequence_name
        seq_steps = sequence_panel(states0, labels,
                                   colors_path=colors_path,
                                   steps_spec=sequence_steps,
                                   save_path=seq_path,
                                   show=sequence_show)
        print(f"[gridmaps] sequence panel steps: {seq_steps}")

    print("[gridmaps] done.")

    
# --- CLI ---
def main():
    ap = argparse.ArgumentParser(description="Full-grid (20×20) spatial heatmaps.")
    ap.add_argument("--runs", nargs="+", required=True, help="Run directories (data/run_000 ...)")
    ap.add_argument("--out", default="reports", help="Output directory")
    ap.add_argument("--win", default="", help="Time window 'start:end' or 'last:N'")
    ap.add_argument("--phase", type=int, default=-1, help="Phase slice (-1 = all)")
    ap.add_argument("--k", type=int, default=2, help="Phase period for slicing")
    ap.add_argument("--sequence", action="store_true", help="Also render a step montage from the first run")
    ap.add_argument("--sequence-steps", default="", help="Comma list or 'auto[:N]' for montage sampling")
    ap.add_argument("--sequence-file", default="", help="Filename for the montage PNG")
    ap.add_argument("--colors", default="label_colors.json", help="Label→color JSON for montage rendering")
    ap.add_argument("--sequence-show", action="store_true", help="Display the montage interactively")
    args = ap.parse_args()
    run_gridmaps(args.runs, args.out,
                 win=args.win, phase=args.phase, k=args.k,
                 sequence=args.sequence, sequence_steps=args.sequence_steps,
                 sequence_file=args.sequence_file, colors_path=args.colors,
                 sequence_show=args.sequence_show)

if __name__ == "__main__":
    main()
