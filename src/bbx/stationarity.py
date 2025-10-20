from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .process import load_run, build_label_map, encode_states
from .utils import parse_window
from .gof import (
    build_dataset,
    LogisticModel,
    RuleModel,
    log_loss,
    brier_score,
)

plt.rcParams['axes.facecolor'] = '#f4f4f4'
plt.rcParams['figure.facecolor'] = '#f4f4f4'
plt.rcParams['savefig.facecolor'] = '#f4f4f4'


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_train_data(run_path: Path, window: str | None, feature_mode: str, radius: int,
                    max_samples: int | None, rng: np.random.Generator | None):
    raw = load_run(run_path)
    if not raw:
        raise ValueError(f"Training run {run_path} empty")
    start, end = parse_window(window, len(raw))
    subset = raw[start:end]
    if len(subset) < 2:
        raise ValueError("Training window must contain at least two steps")
    labels, label_to_idx = build_label_map(subset)
    states = encode_states(subset, label_to_idx)
    X, y = build_dataset(states, radius, feature_mode, max_samples, rng)
    if X.size == 0:
        raise ValueError("Training dataset empty after window/subsample")
    return X, y, labels, label_to_idx, start


def encode_with_map(raw: List[List[List[str]]], label_to_idx: dict) -> List[np.ndarray]:
    states = []
    for grid in raw:
        H, W = len(grid), len(grid[0])
        arr = np.empty((H, W), dtype=np.int16)
        for i in range(H):
            for j in range(W):
                lab = grid[i][j]
                if lab not in label_to_idx:
                    raise ValueError(f"Label '{lab}' encountered outside training set")
                arr[i, j] = label_to_idx[lab]
        states.append(arr)
    return states


def split_segments(num_states: int, segments: int) -> List[Tuple[int, int]]:
    segments = max(1, segments)
    boundaries = np.linspace(0, num_states, segments + 1, dtype=int)
    ranges = []
    for i in range(segments):
        s, e = boundaries[i], boundaries[i + 1]
        if e - s >= 2:
            ranges.append((s, e))
    return ranges


def evaluate(model, X: np.ndarray, y: np.ndarray, n_labels: int) -> dict:
    preds, probs = model.predict(X)
    metrics = {
        "accuracy": float(np.mean(preds == y)),
    }
    if probs is not None:
        metrics["log_loss"] = log_loss(probs, y)
        metrics["brier"] = brier_score(probs, y, n_labels)
    else:
        metrics["log_loss"] = None
        metrics["brier"] = None
    return metrics


def plot_accuracy(seg_df: pd.DataFrame, out_path: Path):
    ensure_dir(out_path.parent)
    plt.figure(figsize=(8, 4))
    for run, g in seg_df.groupby("run"):
        plt.plot(g["segment"], g["accuracy"], marker="o", label=run)
    plt.xlabel("Segment index")
    plt.ylabel("Accuracy")
    plt.title("Segment accuracy vs. training baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_stationarity(args):
    rng = np.random.default_rng(args.seed)
    max_samples = args.max_samples if args.max_samples else None
    X_train, y_train, labels, label_to_idx, train_start = load_train_data(
        Path(args.train_run),
        args.train_window or None,
        args.feature_mode,
        args.radius,
        max_samples,
        rng,
    )

    if args.model == "logistic":
        model = LogisticModel(n_classes=len(labels))
    elif args.model == "rule":
        if args.feature_mode != "local":
            raise ValueError("Rule model only supports feature_mode='local'")
        model = RuleModel()
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model.fit(X_train, y_train)
    train_metrics = evaluate(model, X_train, y_train, len(labels))

    test_runs = args.test_runs if args.test_runs else [args.train_run]
    seg_rows = []
    summary_rows = []

    for run_path in test_runs:
        raw = load_run(Path(run_path))
        if not raw:
            print(f"[stationarity] warn: {run_path} empty; skipping")
            continue
        start, end = parse_window(args.test_window or "", len(raw))
        raw_slice = raw[start:end]
        if len(raw_slice) < 2:
            print(f"[stationarity] warn: window yields <2 steps for {run_path}; skipping")
            continue
        try:
            states = encode_with_map(raw_slice, label_to_idx)
        except ValueError as e:
            print(f"[stationarity] warn: {e}; skipping {run_path}")
            continue

        segments = split_segments(len(states), args.segments)
        if not segments:
            print(f"[stationarity] warn: not enough data for segments in {run_path}")
            continue
        run_metrics = []
        for idx, (s, e) in enumerate(segments):
            seg_states = states[s:e]
            X_seg, y_seg = build_dataset(seg_states, args.radius, args.feature_mode, max_samples, rng)
            if X_seg.size == 0:
                continue
            metrics = evaluate(model, X_seg, y_seg, len(labels))
            metrics["accuracy_drop"] = metrics["accuracy"] - train_metrics["accuracy"]
            if metrics["log_loss"] is not None and train_metrics["log_loss"] is not None:
                metrics["log_loss_diff"] = metrics["log_loss"] - train_metrics["log_loss"]
            else:
                metrics["log_loss_diff"] = None
            row = {
                "run": Path(run_path).name,
                "segment": idx,
                "segment_start": start + s,
                "segment_end": start + e,
            }
            row.update(metrics)
            seg_rows.append(row)
            run_metrics.append(metrics["accuracy"])

        if run_metrics:
            run_metrics = np.array(run_metrics)
            summary_rows.append({
                "run": Path(run_path).name,
                "segments": len(run_metrics),
                "accuracy_mean": float(run_metrics.mean()),
                "accuracy_std": float(run_metrics.std(ddof=0)) if len(run_metrics) > 1 else 0.0,
                "accuracy_drop_mean": float(run_metrics.mean() - train_metrics["accuracy"]),
            })

    outdir = Path(args.out)
    ensure_dir(outdir)

    seg_df = pd.DataFrame(seg_rows)
    seg_df.to_csv(outdir / "stationarity_segments.csv", index=False)
    if not seg_df.empty:
        plot_accuracy(seg_df, outdir / "accuracy_vs_segment.png")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / "stationarity_summary.csv", index=False)

    manifest = {
        "train_run": args.train_run,
        "train_window": args.train_window or "",
        "test_runs": args.test_runs if args.test_runs else [args.train_run],
        "test_window": args.test_window or "",
        "radius": args.radius,
        "feature_mode": args.feature_mode,
        "model": args.model,
        "segments": args.segments,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "train_metrics": train_metrics,
    }
    (outdir / "stationarity_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[stationarity] done → {outdir}")


def main():
    ap = argparse.ArgumentParser(description="Stationarity / cross-run generalization diagnostics.")
    ap.add_argument("--train-run", required=True)
    ap.add_argument("--train-window", default="")
    ap.add_argument("--test-runs", nargs="*", default=None)
    ap.add_argument("--test-window", default="")
    ap.add_argument("--radius", type=int, default=1)
    ap.add_argument("--feature-mode", choices=["local", "center", "global"], default="local")
    ap.add_argument("--model", choices=["logistic", "rule"], default="logistic")
    ap.add_argument("--segments", type=int, default=4)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="reports/stationarity")
    args = ap.parse_args()
    run_stationarity(args)


if __name__ == "__main__":  # pragma: no cover
    main()
