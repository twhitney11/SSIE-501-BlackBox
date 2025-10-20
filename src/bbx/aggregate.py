from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .process import load_run, build_label_map, encode_states
from .viz import load_label_colors
from .utils import parse_window

plt.rcParams['axes.facecolor'] = '#f4f4f4'
plt.rcParams['figure.facecolor'] = '#f4f4f4'
plt.rcParams['savefig.facecolor'] = '#f4f4f4'


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_runs_window(run_dirs: Sequence[Path], window: str | None):
    raw_runs = []
    names = []
    for rd in run_dirs:
        raw = load_run(rd)
        if not raw:
            print(f"[aggregate] warn: {rd} empty; skipping")
            continue
        start, end = parse_window(window, len(raw))
        subset = raw[start:end]
        if len(subset) < 2:
            print(f"[aggregate] warn: window produced <2 steps for {rd}; skipping")
            continue
        raw_runs.append(subset)
        names.append(Path(rd).name)
    if not raw_runs:
        raise ValueError("No runs contained data for the requested window")

    labels, label_to_idx = build_label_map([grid for run in raw_runs for grid in run])
    encoded_runs = [encode_states(run, label_to_idx) for run in raw_runs]
    lengths = [len(run) for run in encoded_runs]
    min_len = min(lengths)
    if min_len < 2:
        raise ValueError("Not enough steps per run after windowing")
    if len(set(lengths)) > 1:
        print(f"[aggregate] runs have different lengths; truncating to {min_len} steps")
    encoded_runs = [run[:min_len] for run in encoded_runs]
    return encoded_runs, labels, names


def compute_fractions(per_run: Sequence[Sequence[np.ndarray]], labels: Sequence[str]) -> np.ndarray:
    num_runs = len(per_run)
    num_steps = len(per_run[0])
    num_labels = len(labels)
    H, W = per_run[0][0].shape

    fractions = np.zeros((num_runs, num_steps, num_labels), dtype=np.float64)
    for r, run in enumerate(per_run):
        for t, state in enumerate(run):
            counts = np.bincount(state.ravel(), minlength=num_labels)
            fractions[r, t] = counts / (H * W)
    return fractions


def compute_change_counts(per_run: Sequence[Sequence[np.ndarray]]) -> np.ndarray:
    num_runs = len(per_run)
    num_steps = len(per_run[0]) - 1
    H, W = per_run[0][0].shape
    changes = np.zeros((num_runs, num_steps), dtype=np.float64)
    for r, run in enumerate(per_run):
        for t in range(num_steps):
            changes[r, t] = np.sum(run[t] != run[t + 1])
    return changes


def mean_ci(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = data.mean(axis=0)
    if data.shape[0] > 1:
        std = data.std(axis=0, ddof=0)
        ci = 1.96 * std / np.sqrt(data.shape[0])
    else:
        std = np.zeros_like(mean)
        ci = np.zeros_like(mean)
    return mean, mean - ci, mean + ci


def save_csv(rows, path: Path):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"[aggregate] saved {path}")


def plot_fractions(fractions: np.ndarray, labels: Sequence[str], focus: Sequence[str],
                   colors: dict, out_path: Path, run_names: Sequence[str]):
    steps = np.arange(fractions.shape[1])
    ensure_dir(out_path)
    for lab in focus:
        if lab not in labels:
            print(f"[aggregate] warn: label {lab} not in data; skipping plot")
            continue
        idx = labels.index(lab)
        plt.figure(figsize=(10, 5))
        for r in range(fractions.shape[0]):
            plt.plot(steps, fractions[r, :, idx], alpha=0.3, linewidth=1.0)
        mean, lo, hi = mean_ci(fractions[:, :, idx])
        plt.plot(steps, mean, color=colors.get(lab, "#000"), linewidth=2.0, label=f"{lab} mean")
        plt.fill_between(steps, lo, hi, color=colors.get(lab, "#000"), alpha=0.2, label="95% CI")
        plt.xlabel("Step")
        plt.ylabel("Fraction of grid")
        plt.title(f"{lab} fractions — mean ± 95% CI")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path / f"fractions_{lab}.png", dpi=150)
        plt.close()


def plot_changes(changes: np.ndarray, out_path: Path):
    steps = np.arange(changes.shape[1])
    ensure_dir(out_path)
    plt.figure(figsize=(10, 4))
    for r in range(changes.shape[0]):
        plt.plot(steps, changes[r], alpha=0.3, linewidth=1.0)
    mean, lo, hi = mean_ci(changes)
    plt.plot(steps, mean, color="#d62728", linewidth=2.0, label="mean")
    plt.fill_between(steps, lo, hi, color="#d62728", alpha=0.2, label="95% CI")
    plt.xlabel("Step")
    plt.ylabel("Cells changed")
    plt.title("Cells changed per step — mean ± 95% CI")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / "change_counts.png", dpi=150)
    plt.close()


def cell_statistics(states: np.ndarray, label_idx: int) -> tuple[np.ndarray, np.ndarray]:
    mask = (states == label_idx)
    mean = mask.mean(axis=(0, 1))
    std = mask.std(axis=(0, 1), ddof=0)
    return mean, std


def save_heatmap(grid: np.ndarray, path: Path, title: str, cmap="viridis", vmin=None, vmax=None):
    ensure_dir(path.parent)
    plt.figure(figsize=(5, 5))
    im = plt.imshow(grid, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.colorbar(im, shrink=0.8)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_cell_stats(mean: np.ndarray, std: np.ndarray, path: Path, label: str):
    rows = []
    H, W = mean.shape
    for i in range(H):
        for j in range(W):
            rows.append({"i": i, "j": j, "label": label, "mean": float(mean[i, j]), "std": float(std[i, j])})
    save_csv(rows, path)


def run_aggregate(args):
    run_dirs = [Path(p) for p in args.runs]
    ensure_dir(Path(args.out))
    per_run, labels, names = load_runs_window(run_dirs, args.window or None)
    colors = load_label_colors(args.colors)

    fractions = compute_fractions(per_run, labels)
    focus = [lab.strip() for lab in (args.labels or "").split(",") if lab.strip()]
    if not focus:
        focus = labels[:3]
    plot_fractions(fractions, labels, focus, colors, Path(args.out), names)

    rows = []
    for li, lab in enumerate(labels):
        mean, lo, hi = mean_ci(fractions[:, :, li])
        std = fractions[:, :, li].std(axis=0, ddof=0)
        for step in range(fractions.shape[1]):
            rows.append({
                "label": lab,
                "step": step,
                "mean": float(mean[step]),
                "std": float(std[step]),
                "ci_low": float(lo[step]),
                "ci_high": float(hi[step]),
            })
    save_csv(rows, Path(args.out) / "fractions_summary.csv")

    change_counts = compute_change_counts(per_run)
    plot_changes(change_counts, Path(args.out))
    mean_change, lo_change, hi_change = mean_ci(change_counts)
    change_rows = [{
        "step": step,
        "mean": float(mean_change[step]),
        "ci_low": float(lo_change[step]),
        "ci_high": float(hi_change[step]),
    } for step in range(change_counts.shape[1])]
    save_csv(change_rows, Path(args.out) / "change_counts_summary.csv")

    states_arr = np.array([np.stack(run) for run in per_run])  # shape (R, T, H, W)
    for lab in focus:
        if lab not in labels:
            continue
        idx = labels.index(lab)
        mean, std = cell_statistics(states_arr, idx)
        save_cell_stats(mean, std, Path(args.out) / f"cell_stats_{lab}.csv", lab)
        save_heatmap(mean, Path(args.out) / f"cell_mean_{lab}.png", f"Cell occupancy mean — {lab}",
                     cmap="magma", vmin=0.0, vmax=1.0)
        save_heatmap(std, Path(args.out) / f"cell_std_{lab}.png", f"Cell occupancy std — {lab}",
                     cmap="plasma")

    # per-cell change rate
    cell_changes = (states_arr[:, 1:] != states_arr[:, :-1])
    mean_change_cell = cell_changes.mean(axis=(0, 1))
    std_change_cell = cell_changes.std(axis=(0, 1), ddof=0)
    save_cell_stats(mean_change_cell, std_change_cell, Path(args.out) / "cell_change_stats.csv", "change")
    save_heatmap(mean_change_cell, Path(args.out) / "cell_change_mean.png", "Cell change rate mean", cmap="viridis")

    manifest = {
        "runs": [str(p) for p in run_dirs],
        "window": args.window or "",
        "labels": labels,
        "focus_labels": focus,
        "steps": fractions.shape[1],
        "num_runs": len(per_run),
    }
    (Path(args.out) / "aggregate_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[aggregate] done → {args.out}")


def main():
    ap = argparse.ArgumentParser(description="Cross-run aggregation (mean/variation) of key metrics.")
    ap.add_argument("--runs", nargs="+", required=True, help="Run directories")
    ap.add_argument("--window", default="", help="Time window 'start:end' or 'last:N'")
    ap.add_argument("--labels", default="", help="Comma list of focus labels for plots (default: top 3)")
    ap.add_argument("--colors", default="label_colors.json", help="Label→color JSON")
    ap.add_argument("--out", default="reports/aggregate", help="Output directory")
    args = ap.parse_args()
    run_aggregate(args)


if __name__ == "__main__":  # pragma: no cover
    main()
