from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score

from .process import load_run, build_label_map, encode_states
from .utils import parse_window

plt.rcParams['axes.facecolor'] = '#f0f0f0'
plt.rcParams['figure.facecolor'] = '#f0f0f0'
plt.rcParams['savefig.facecolor'] = '#f0f0f0'


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_runs_aligned(run_dirs: Sequence[Path], window: str | None) -> Tuple[np.ndarray, List[str]]:
    raw_runs = []
    names = []
    for rd in run_dirs:
        raw = load_run(rd)
        if not raw:
            print(f"[mi] warn: {rd} empty; skipping")
            continue
        start, end = parse_window(window, len(raw))
        subset = raw[start:end]
        if len(subset) < 2:
            print(f"[mi] warn: window yields <2 steps for {rd}; skipping")
            continue
        raw_runs.append(subset)
        names.append(Path(rd).name)
    if not raw_runs:
        raise ValueError("No runs contained data for requested window")

    labels, label_to_idx = build_label_map([grid for run in raw_runs for grid in run])
    encoded = [encode_states(run, label_to_idx) for run in raw_runs]
    lengths = [len(run) for run in encoded]
    min_len = min(lengths)
    if len(set(lengths)) > 1:
        print(f"[mi] runs have different lengths; truncating to {min_len} steps")
    encoded = [run[:min_len] for run in encoded]

    array = np.array([np.stack(run) for run in encoded])  # shape (R, T, H, W)
    return array, names


def sample_pairs(H: int, W: int, num_samples: int, rng: np.random.Generator) -> List[Tuple[int, int, int, int]]:
    max_pairs = H * W * (H * W - 1) // 2
    num_samples = min(num_samples, max_pairs)
    pairs = []
    seen = set()
    total_cells = H * W
    flat_indices = np.arange(total_cells)
    while len(pairs) < num_samples:
        idx = rng.choice(flat_indices, size=2, replace=False)
        i1, j1 = divmod(int(idx[0]), W)
        i2, j2 = divmod(int(idx[1]), W)
        key = tuple(sorted([(i1, j1), (i2, j2)]))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((i1, j1, i2, j2))
    return pairs


def chebyshev_distance(i1: int, j1: int, i2: int, j2: int) -> int:
    return max(abs(i1 - i2), abs(j1 - j2))


def compute_metrics(states: np.ndarray, pairs: Sequence[Tuple[int, int, int, int]]) -> List[Dict[str, float]]:
    R, T, H, W = states.shape
    results = []
    flat_states = states.reshape(R * T, H, W)
    for i1, j1, i2, j2 in pairs:
        series1 = flat_states[:, i1, j1]
        series2 = flat_states[:, i2, j2]
        mi = float(mutual_info_score(series1, series2))
        if np.all(series1 == series1[0]) and np.all(series2 == series2[0]):
            corr = 0.0
        else:
            corr = float(np.corrcoef(series1, series2)[0, 1])
        results.append({
            "i1": i1,
            "j1": j1,
            "i2": i2,
            "j2": j2,
            "distance": chebyshev_distance(i1, j1, i2, j2),
            "mi": mi,
            "corr": corr,
        })
    return results


def aggregate_by_distance(rows: Sequence[Dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    grouped = df.groupby("distance")
    summary = grouped["mi"].agg(["mean", "std", "count"]).reset_index()
    summary.rename(columns={"mean": "mean_mi", "std": "std_mi", "count": "n_pairs"}, inplace=True)
    summary["mean_corr"] = grouped["corr"].mean().values
    summary["std_corr"] = grouped["corr"].std(ddof=0).values
    return summary


def plot_mi_curve(summary: pd.DataFrame, out_path: Path):
    ensure_dir(out_path.parent)
    plt.figure(figsize=(8, 4))
    plt.errorbar(summary["distance"], summary["mean_mi"], yerr=summary["std_mi"], fmt="o-", capsize=4)
    plt.xlabel("Chebyshev distance")
    plt.ylabel("Mutual information (bits)")
    plt.title("MI vs distance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_corr_curve(summary: pd.DataFrame, out_path: Path):
    ensure_dir(out_path.parent)
    plt.figure(figsize=(8, 4))
    plt.errorbar(summary["distance"], summary["mean_corr"], yerr=summary["std_corr"], fmt="o-", capsize=4, color="#ff7f0e")
    plt.xlabel("Chebyshev distance")
    plt.ylabel("Correlation")
    plt.title("Correlation vs distance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_mi(args):
    rng = np.random.default_rng(args.seed)
    states, names = load_runs_aligned([Path(p) for p in args.runs], args.window or None)
    R, T, H, W = states.shape
    max_pairs = args.samples if args.samples else min(5000, H * W * 10)
    pairs = sample_pairs(H, W, max_pairs, rng)
    rows = compute_metrics(states, pairs)
    df = pd.DataFrame(rows)
    outdir = Path(args.out)
    ensure_dir(outdir)
    df.to_csv(outdir / "pairwise_mi_samples.csv", index=False)
    summary = aggregate_by_distance(rows)
    summary.to_csv(outdir / "mi_by_distance.csv", index=False)
    plot_mi_curve(summary, outdir / "mi_distance_curve.png")
    plot_corr_curve(summary, outdir / "corr_distance_curve.png")

    manifest = {
        "runs": args.runs,
        "window": args.window or "",
        "samples": len(rows),
        "seed": args.seed,
    }
    (outdir / "mi_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[mi] done → {outdir}")


def main():
    ap = argparse.ArgumentParser(description="Pairwise MI/correlation analysis.")
    ap.add_argument("--runs", nargs="+", required=True, help="Run directories")
    ap.add_argument("--window", default="", help="Time window 'start:end' or 'last:N'")
    ap.add_argument("--samples", type=int, default=1000, help="Number of cell-pair samples")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")
    ap.add_argument("--out", default="reports/mi", help="Output directory")
    args = ap.parse_args()
    run_mi(args)


if __name__ == "__main__":  # pragma: no cover
    main()
