from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .process import load_run, build_label_map, encode_states
from .utils import parse_window

plt.rcParams['axes.facecolor'] = '#f6f6f6'
plt.rcParams['figure.facecolor'] = '#f6f6f6'
plt.rcParams['savefig.facecolor'] = '#f6f6f6'


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_runs(run_dirs: Sequence[Path], window: str | None):
    raw_runs = []
    names = []
    for rd in run_dirs:
        raw = load_run(rd)
        if not raw:
            print(f"[entropy] warn: {rd} empty; skipping")
            continue
        start, end = parse_window(window, len(raw))
        subset = raw[start:end]
        if len(subset) < 2:
            print(f"[entropy] warn: window yields <2 steps for {rd}; skipping")
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
        print(f"[entropy] runs differ in length; truncating to {min_len} steps")
    encoded = [run[:min_len] for run in encoded]
    return encoded, labels, names


def region_masks(H: int, W: int) -> Dict[str, np.ndarray]:
    interior = np.ones((H, W), dtype=bool)
    interior[[0, -1], :] = False
    interior[:, [0, -1]] = False
    edge = ~interior
    corner = np.zeros((H, W), dtype=bool)
    corner[0, 0] = corner[0, -1] = corner[-1, 0] = corner[-1, -1] = True
    edge_no_corner = edge & ~corner
    return {
        "all": np.ones((H, W), dtype=bool),
        "interior": interior,
        "edge": edge_no_corner,
        "corner": corner,
    }


def encode_token(neigh: Sequence[int], base: int) -> int:
    code = 0
    for v in neigh:
        code = code * base + (v + 1)  # shift so -1 → 0
    return code


def iter_neighborhood(state: np.ndarray, radius: int, mask: np.ndarray) -> Sequence[List[int]]:
    H, W = state.shape
    offsets = range(-radius, radius + 1)
    for i in range(H):
        for j in range(W):
            if not mask[i, j]:
                continue
            neigh = []
            for di in offsets:
                ii = i + di
                for dj in offsets:
                    jj = j + dj
                    if 0 <= ii < H and 0 <= jj < W:
                        neigh.append(int(state[ii, jj]))
                    else:
                        neigh.append(-1)
            yield neigh, i, j


def neighborhood_entropy_series(run_states: Sequence[np.ndarray], radius: int,
                                regions: Dict[str, np.ndarray], num_labels: int) -> Dict[str, np.ndarray]:
    base = num_labels + 1
    num_steps = len(run_states)
    series = {name: np.zeros(num_steps, dtype=np.float64) for name in regions}
    for t, state in enumerate(run_states):
        for name, mask in regions.items():
            counts = Counter()
            total = 0
            for neigh, _, _ in iter_neighborhood(state, radius, mask):
                token = encode_token(neigh, base)
                counts[token] += 1
                total += 1
            if total == 0:
                series[name][t] = 0.0
                continue
            ent = 0.0
            for cnt in counts.values():
                p = cnt / total
                ent -= p * np.log2(p)
            series[name][t] = ent
    return series


def conditional_entropy(run_states: Sequence[np.ndarray], radius: int,
                        regions: Dict[str, np.ndarray], num_labels: int) -> Dict[str, float]:
    base = num_labels + 1
    results = {}
    H, W = run_states[0].shape
    offsets = range(-radius, radius + 1)
    for name, mask in regions.items():
        counts = defaultdict(lambda: Counter())
        total = 0
        for t in range(len(run_states) - 1):
            S = run_states[t]
            T = run_states[t + 1]
            for i in range(H):
                for j in range(W):
                    if not mask[i, j]:
                        continue
                    neigh = []
                    for di in offsets:
                        ii = i + di
                        for dj in offsets:
                            jj = j + dj
                            if 0 <= ii < H and 0 <= jj < W:
                                neigh.append(int(S[ii, jj]))
                            else:
                                neigh.append(-1)
                    token = encode_token(neigh, base)
                    counts[token][int(T[i, j])] += 1
                    total += 1
        if total == 0:
            results[name] = np.nan
            continue
        ent_total = 0.0
        for token, label_counts in counts.items():
            token_total = sum(label_counts.values())
            p_token = token_total / total
            ent_token = 0.0
            for cnt in label_counts.values():
                p = cnt / token_total
                ent_token -= p * np.log2(p)
            ent_total += p_token * ent_token
        results[name] = ent_total
    return results


def mean_ci(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = data.mean(axis=0)
    if data.shape[0] > 1:
        std = data.std(axis=0, ddof=0)
        ci = 1.96 * std / np.sqrt(data.shape[0])
    else:
        ci = np.zeros_like(mean)
    return mean, mean - ci, mean + ci


def plot_series(series: np.ndarray, steps: np.ndarray, title: str, out_path: Path):
    ensure_dir(out_path.parent)
    plt.figure(figsize=(10, 5))
    for r in range(series.shape[0]):
        plt.plot(steps, series[r], alpha=0.3, linewidth=1.0)
    mean, lo, hi = mean_ci(series)
    plt.plot(steps, mean, color="#1f77b4", linewidth=2.0, label="mean")
    plt.fill_between(steps, lo, hi, color="#1f77b4", alpha=0.2, label="95% CI")
    plt.xlabel("Step")
    plt.ylabel("Entropy (bits)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_entropy(args):
    run_dirs = [Path(p) for p in args.runs]
    runs, labels, names = load_runs(run_dirs, args.window or None)
    ensure_dir(Path(args.out))
    regions = region_masks(*runs[0][0].shape)
    selected_regions = [r.strip() for r in args.regions.split(",") if r.strip()] if args.regions else ["all", "interior", "edge", "corner"]
    regions = {name: regions[name] for name in selected_regions if name in regions}
    radius = args.radius
    if radius < 1:
        raise ValueError("radius must be >= 1")

    num_runs = len(runs)
    num_steps = len(runs[0])
    steps = np.arange(num_steps)
    base_metrics = []
    per_run_entropy = {name: np.zeros((num_runs, num_steps)) for name in regions}
    cond_rows = []
    cond_values = {name: [] for name in regions}

    for r, run_states in enumerate(runs):
        series = neighborhood_entropy_series(run_states, radius, regions, len(labels))
        for name in regions:
            per_run_entropy[name][r] = series[name]

        cond = conditional_entropy(run_states, radius, regions, len(labels))
        for name, val in cond.items():
            cond_rows.append({"run": names[r], "region": name, "entropy_conditional": val})
            cond_values[name].append(val)

    # Save series data
    rows = []
    for name in regions:
        for r_idx, run_name in enumerate(names):
            for step in range(num_steps):
                rows.append({
                    "run": run_name,
                    "region": name,
                    "step": step,
                    "entropy": float(per_run_entropy[name][r_idx, step]),
                })
        plot_series(per_run_entropy[name], steps, f"H(N) per step — {name}", Path(args.out) / f"neighborhood_entropy_{name}.png")
    pd.DataFrame(rows).to_csv(Path(args.out) / "neighborhood_entropy_series.csv", index=False)

    # Summary across runs
    summary_rows = []
    for name in regions:
        series = per_run_entropy[name]
        mean, lo, hi = mean_ci(series)
        std = series.std(axis=0, ddof=0) if series.shape[0] > 1 else np.zeros_like(mean)
        for step in range(num_steps):
            summary_rows.append({
                "region": name,
                "step": step,
                "mean": float(mean[step]),
                "std": float(std[step]),
                "ci_low": float(lo[step]),
                "ci_high": float(hi[step]),
            })
    pd.DataFrame(summary_rows).to_csv(Path(args.out) / "neighborhood_entropy_summary.csv", index=False)

    # Conditional entropy
    pd.DataFrame(cond_rows).to_csv(Path(args.out) / "conditional_entropy_runs.csv", index=False)
    cond_summary = []
    for name, vals in cond_values.items():
        if len(vals) == 0:
            continue
        arr = np.array(vals)
        mean = arr.mean()
        if arr.size > 1:
            std = arr.std(ddof=0)
            ci = 1.96 * std / np.sqrt(arr.size)
        else:
            std = 0.0
            ci = 0.0
        cond_summary.append({
            "region": name,
            "mean": float(mean),
            "std": float(std),
            "ci_low": float(mean - ci),
            "ci_high": float(mean + ci),
        })
    cond_df = pd.DataFrame(cond_summary)
    cond_df.to_csv(Path(args.out) / "conditional_entropy_summary.csv", index=False)

    if not cond_df.empty:
        ensure_dir(Path(args.out))
        plt.figure(figsize=(6, 4))
        plt.bar(cond_df["region"], cond_df["mean"], yerr=cond_df["mean"] - cond_df["ci_low"], capsize=4, color="#ff7f0e")
        plt.ylabel("H(S_{t+1} | N_r(S_t)) (bits)")
        plt.title("Conditional entropy by region")
        plt.tight_layout()
        plt.savefig(Path(args.out) / "conditional_entropy.png", dpi=150)
        plt.close()

    manifest = {
        "runs": [str(p) for p in run_dirs],
        "window": args.window or "",
        "radius": radius,
        "regions": list(regions.keys()),
        "num_runs": num_runs,
        "steps": num_steps,
    }
    (Path(args.out) / "entropy_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[entropy] done → {args.out}")


def main():
    ap = argparse.ArgumentParser(description="Neighborhood and conditional entropy analysis.")
    ap.add_argument("--runs", nargs="+", required=True, help="Run directories")
    ap.add_argument("--window", default="", help="Time window 'start:end' or 'last:N'")
    ap.add_argument("--radius", type=int, default=1, help="Neighborhood radius")
    ap.add_argument("--regions", default="all,interior,edge,corner", help="Comma regions (all,interior,edge,corner)")
    ap.add_argument("--out", default="reports/entropy", help="Output directory")
    args = ap.parse_args()
    run_entropy(args)


if __name__ == "__main__":  # pragma: no cover
    main()
