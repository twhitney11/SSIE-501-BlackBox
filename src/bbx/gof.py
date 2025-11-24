from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from .process import load_run, encode_states, build_label_map
from .utils import parse_window


@dataclass
class GOFConfig:
    train_run: Path
    train_window: str | None
    test_runs: List[Path]
    test_window: str | None
    radius: int
    feature_mode: str
    model: str
    max_samples: int | None
    permutations: List[str]
    seed: int | None
    outdir: Path
    knn_k: int = 5


def encode_run_with_map(run_dir: Path, label_to_idx: dict, window: str | None) -> List[np.ndarray]:
    raw_states = load_run(run_dir)
    if not raw_states:
        return []
    start, end = parse_window(window, len(raw_states))
    raw_slice = raw_states[start:end]
    states = []
    for grid in raw_slice:
        H, W = len(grid), len(grid[0])
        arr = np.empty((H, W), dtype=np.int16)
        for i in range(H):
            for j in range(W):
                lab = grid[i][j]
                if lab not in label_to_idx:
                    raise ValueError(f"Label '{lab}' in {run_dir} not seen in training data")
                arr[i, j] = label_to_idx[lab]
        states.append(arr)
    return states


def build_dataset(states: Sequence[np.ndarray], radius: int, feature_mode: str,
                  max_samples: int | None = None,
                  rng: np.random.Generator | None = None) -> Tuple[np.ndarray, np.ndarray]:
    if len(states) < 2:
        raise ValueError("Need at least two states to build dataset")
    H, W = states[0].shape
    for S in states:
        if S.shape != (H, W):
            raise ValueError("Inconsistent state shapes across time")
    num_steps = len(states) - 1
    offsets = range(-radius, radius + 1)

    if feature_mode == "local":
        feature_size = (2 * radius + 1) ** 2
    elif feature_mode == "center":
        feature_size = 1
    elif feature_mode == "global":
        feature_size = H * W + 2  # include coordinates
    else:
        raise ValueError(f"Unknown feature mode: {feature_mode}")

    total_samples = num_steps * H * W
    X = np.empty((total_samples, feature_size), dtype=np.float32)
    y = np.empty(total_samples, dtype=np.int16)
    idx = 0

    for t in range(num_steps):
        S = states[t]
        T = states[t + 1]
        if feature_mode == "global":
            flat = S.reshape(-1).astype(np.float32)
        for i in range(H):
            for j in range(W):
                X[idx].fill(0.0)
                if feature_mode == "local":
                    feat = []
                    for di in offsets:
                        ii = i + di
                        for dj in offsets:
                            jj = j + dj
                            if 0 <= ii < H and 0 <= jj < W:
                                feat.append(S[ii, jj])
                            else:
                                feat.append(-1)
                    X[idx, :len(feat)] = np.array(feat, dtype=np.float32)
                elif feature_mode == "center":
                    X[idx, 0] = float(S[i, j])
                elif feature_mode == "global":
                    X[idx, : H * W] = flat
                    X[idx, H * W] = i
                    X[idx, H * W + 1] = j
                y[idx] = T[i, j]
                idx += 1
    if max_samples and total_samples > max_samples:
        rng = rng or np.random.default_rng()
        chosen = rng.choice(total_samples, size=max_samples, replace=False)
        X = X[chosen]
        y = y[chosen]
    return X, y


def window_tag(window: str | None) -> str:
    if not window:
        return "full"
    return window.replace(":", "-").replace("/", "_")


class LogisticModel:
    def __init__(self, n_classes: int):
        self.model = LogisticRegression(
            solver="lbfgs",
            multi_class="auto",
            max_iter=500,
        )
        self.n_classes = n_classes
        self.n_params = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        coef = self.model.coef_
        intercept = self.model.intercept_
        self.n_params = coef.size + intercept.size

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        probs_raw = self.model.predict_proba(X)
        preds = self.model.predict(X)
        full = np.zeros((len(preds), self.n_classes))
        classes = self.model.classes_
        for idx, cls in enumerate(classes):
            full[:, int(cls)] = probs_raw[:, idx]
        return preds, full


class RuleModel:
    def __init__(self):
        self.rule: dict[tuple[int, ...], int] = {}
        self.counts: dict[tuple[int, ...], dict[int, int]] = {}
        self.fallback: int = 0
        self.fallback_dist: np.ndarray | None = None
        self.n_params = 0
        self.n_classes = 0

    def fit(self, X: np.ndarray, y: np.ndarray):
        from collections import Counter, defaultdict

        counts: dict[tuple[int, ...], Counter] = defaultdict(Counter)
        for row, target in zip(X, y):
            key = tuple(int(round(v)) for v in row.tolist())
            counts[key][int(target)] += 1
        self.rule = {k: cnt.most_common(1)[0][0] for k, cnt in counts.items()}
        self.counts = {k: dict(cnt) for k, cnt in counts.items()}
        overall = Counter(y.tolist())
        self.fallback = overall.most_common(1)[0][0]
        self.n_classes = int(np.max(y)) + 1 if len(y) else len(overall)
        total = sum(overall.values())
        self.fallback_dist = np.array([overall.get(c, 0) for c in range(self.n_classes)], dtype=np.float64)
        if total > 0:
            self.fallback_dist = (self.fallback_dist + 1.0) / (total + self.n_classes)
        self.n_params = len(self.rule) * max(0, self.n_classes - 1)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.n_classes == 0:
            raise ValueError("Model not fitted")
        preds = np.empty(len(X), dtype=np.int16)
        probs = np.zeros((len(X), self.n_classes), dtype=np.float64)
        for i, row in enumerate(X):
            key = tuple(int(round(v)) for v in row.tolist())
            if key in self.counts:
                cnts = self.counts[key]
                total = sum(cnts.values())
                vals = np.array([cnts.get(c, 0) for c in range(self.n_classes)], dtype=np.float64)
                probs[i] = (vals + 1.0) / (total + self.n_classes)
                preds[i] = self.rule.get(key, self.fallback)
            else:
                probs[i] = self.fallback_dist
                preds[i] = self.fallback
        return preds, probs


class MarkovModel:
    def __init__(self, n_classes: int, smoothing: float = 1.0):
        self.n_classes = n_classes
        self.smoothing = smoothing
        self.counts = np.zeros((n_classes, n_classes), dtype=np.float64)
        self.probs = np.full((n_classes, n_classes), 1.0 / n_classes, dtype=np.float64)
        self.n_params = n_classes * (n_classes - 1)

    def fit(self, X: np.ndarray, y: np.ndarray):
        current = X[:, 0].astype(int)
        for c, nxt in zip(current, y):
            if 0 <= c < self.n_classes and 0 <= nxt < self.n_classes:
                self.counts[c, nxt] += 1
        sm = self.smoothing
        sm_counts = self.counts + sm
        self.probs = sm_counts / sm_counts.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        current = X[:, 0].astype(int)
        probs = self.probs[current]
        preds = np.argmax(probs, axis=1).astype(np.int16)
        return preds, probs


class KNNModel:
    def __init__(self, n_neighbors: int, n_classes: int):
        self.n_neighbors = n_neighbors
        self.n_classes = n_classes
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
        self.n_params = n_neighbors  # heuristic

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        probs = self.model.predict_proba(X)
        full = np.zeros((len(X), self.n_classes))
        classes = self.model.classes_
        for idx, cls in enumerate(classes):
            full[:, int(cls)] = probs[:, idx]
        preds = np.argmax(full, axis=1).astype(np.int16)
        return preds, full


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    conf = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        conf[int(t), int(p)] += 1
    return conf


# Abdullah Added on 11/23/25
# Confusion Matrix Heatmap Plot
# This function visualizes the confusion matrix as a color-coded heatmap showing the classification
# performance of the model. The heatmap displays how many times each actual label (rows) was
# predicted as each label (columns). Diagonal elements represent correct predictions, while
# off-diagonal elements show misclassifications. The plot includes count annotations on each cell
# and uses a blue color scale where darker colors indicate higher counts.
def plot_confusion_matrix_heatmap(conf: np.ndarray, labels: Sequence[str], 
                                   out_path: Path, title: str = "Confusion Matrix"):
    """Plot confusion matrix as a heatmap."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(conf, cmap='Blues', interpolation='nearest')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Count')
    
    # Set ticks and labels
    n_classes = len(labels)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    
    # Add text annotations
    thresh = conf.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            text = ax.text(j, i, int(conf[i, j]),
                         ha="center", va="center",
                         color="white" if conf[i, j] > thresh else "black")
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def log_loss(probs: np.ndarray, y_true: np.ndarray) -> float:
    eps = 1e-12
    idx = np.arange(len(y_true))
    chosen = probs[idx, y_true]
    chosen = np.clip(chosen, eps, 1.0)
    return float(-np.mean(np.log(chosen)))


def brier_score(probs: np.ndarray, y_true: np.ndarray, n_classes: int) -> float:
    one_hot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def calibration_curve(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    pred_class = np.argmax(probs, axis=1)
    pred_prob = probs[np.arange(len(pred_class)), pred_class]
    correct = (pred_class == y_true).astype(int)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.clip((pred_prob * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            continue
        rows.append({
            "bin": b,
            "lower": bins[b],
            "upper": bins[b + 1],
            "count": count,
            "pred_mean": float(pred_prob[mask].mean()),
            "obs_rate": float(correct[mask].mean()),
        })
    return pd.DataFrame(rows)


# Abdullah Added on 11/23/25
# Calibration Curve Plot
# This function creates a calibration plot that assesses how well the model's predicted probabilities
# match the observed frequencies. The plot shows the relationship between mean predicted probability
# (x-axis) and observed rate (y-axis) across probability bins. A perfectly calibrated model would
# follow the diagonal line (y=x), where predicted probabilities equal observed rates. Deviations
# from this line indicate overconfidence (above diagonal) or underconfidence (below diagonal) in
# the model's predictions. This is a key diagnostic for probabilistic classifiers.
def plot_calibration_curve(calib_df: pd.DataFrame, out_path: Path, 
                          title: str = "Calibration Curve"):
    """Plot predicted vs observed probabilities."""
    import matplotlib.pyplot as plt
    
    if calib_df.empty:
        print(f"[warn] calibration data empty, skipping plot")
        return
    
    plt.figure(figsize=(8, 6))
    plt.plot(calib_df['pred_mean'], calib_df['obs_rate'], 'o-', 
             linewidth=2, markersize=8, label='Model')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Observed Rate')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def apply_permutation(X: np.ndarray, y: np.ndarray, kind: str, rng: np.random.Generator,
                      radius: int, feature_mode: str) -> Tuple[np.ndarray, np.ndarray]:
    X_perm = X.copy()
    y_perm = y.copy()
    if kind == "random":
        for col in range(X_perm.shape[1]):
            rng.shuffle(X_perm[:, col])
        return X_perm, y_perm

    if kind == "target":
        rng.shuffle(y_perm)
        return X_perm, y_perm

    if feature_mode != "local":
        for col in range(X_perm.shape[1]):
            rng.shuffle(X_perm[:, col])
        return X_perm, y_perm

    offsets = []
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            offsets.append((di, dj))

    if kind in {"inner", "ring"}:
        if radius < 1:
            return X_perm, y_perm
        mask = []
        for idx, (di, dj) in enumerate(offsets):
            dist = max(abs(di), abs(dj))
            if kind == "inner" and dist < radius:
                mask.append(idx)
            if kind == "ring" and dist == radius:
                mask.append(idx)
        for col in mask:
            rng.shuffle(X_perm[:, col])
        return X_perm, y_perm

    raise ValueError(f"Unknown permutation kind: {kind}")


def ensure_outdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def run_gof(cfg: GOFConfig):
    ensure_outdir(cfg.outdir)
    rng = np.random.default_rng(cfg.seed)

    # Training data
    raw_train = load_run(cfg.train_run)
    if not raw_train:
        raise ValueError(f"Training run {cfg.train_run} is empty")
    start, end = parse_window(cfg.train_window, len(raw_train))
    raw_train_slice = raw_train[start:end]
    if len(raw_train_slice) < 2:
        raise ValueError("Training window must contain at least two steps")
    labels, label_to_idx = build_label_map(raw_train_slice)
    train_states = encode_states(raw_train_slice, label_to_idx)

    X_train, y_train = build_dataset(train_states, cfg.radius, cfg.feature_mode, cfg.max_samples, rng)
    if X_train.size == 0:
        raise ValueError("Training dataset is empty after window/subsampling")

    if cfg.model == "logistic":
        model = LogisticModel(n_classes=len(labels))
    elif cfg.model == "rule":
        if cfg.feature_mode != "local":
            raise ValueError("Rule model only supports feature_mode='local'")
        model = RuleModel()
    elif cfg.model == "markov":
        if cfg.feature_mode != "center":
            raise ValueError("Markov model requires --feature-mode center")
        model = MarkovModel(n_classes=len(labels))
    elif cfg.model == "knn":
        model = KNNModel(n_neighbors=cfg.knn_k, n_classes=len(labels))
    else:
        raise ValueError(f"Unknown model type: {cfg.model}")
    model.fit(X_train, y_train)

    summary_rows = []

    test_runs = cfg.test_runs or [cfg.train_run]
    for run_path in test_runs:
        states = encode_run_with_map(run_path, label_to_idx, cfg.test_window)
        if len(states) < 2:
            print(f"[warn] {run_path}: window yields insufficient steps; skipping")
            continue
        X_test, y_test = build_dataset(states, cfg.radius, cfg.feature_mode, cfg.max_samples, rng)
        if X_test.size == 0:
            print(f"[warn] {run_path}: empty dataset; skipping")
            continue

        preds, probs = model.predict(X_test)
        base_metrics = {
            "model": cfg.model,
            "accuracy": float(np.mean(preds == y_test)),
        }
        conf = confusion_matrix(y_test, preds, len(labels))
        suffix = window_tag(cfg.test_window)
        conf_path = cfg.outdir / f"confusion_{Path(run_path).name}_{suffix}.csv"
        pd.DataFrame(conf, index=labels, columns=labels).to_csv(conf_path)
        # Plot confusion matrix as heatmap
        plot_confusion_matrix_heatmap(conf, labels,
            cfg.outdir / f"confusion_{Path(run_path).name}_{suffix}.png",
            title=f"Confusion Matrix - {Path(run_path).name}")

        if probs is not None:
            ll = float(np.sum(np.log(np.clip(probs[np.arange(len(y_test)), y_test], 1e-12, 1.0))))
            base_metrics.update({
                "log_loss": log_loss(probs, y_test),
                "brier": brier_score(probs, y_test, len(labels)),
                "log_likelihood": ll,
            })
            k_params = getattr(model, "n_params", None)
            if k_params is not None:
                n = len(y_test)
                base_metrics["aic"] = float(2 * k_params - 2 * ll)
                base_metrics["bic"] = float(k_params * np.log(n) - 2 * ll)
            else:
                base_metrics["aic"] = None
                base_metrics["bic"] = None
            calib = calibration_curve(probs, y_test)
            calib.to_csv(cfg.outdir / f"calibration_{Path(run_path).name}_{suffix}.csv", index=False)
            # Plot calibration curve
            plot_calibration_curve(calib,
                cfg.outdir / f"calibration_{Path(run_path).name}_{suffix}.png",
                title=f"Calibration Curve - {Path(run_path).name}")
        else:
            base_metrics.update({"log_loss": None, "brier": None, "log_likelihood": None, "aic": None, "bic": None})

        row = {
            "test_run": str(run_path),
            "test_window": cfg.test_window or "",
            "feature_mode": cfg.feature_mode,
            "samples": int(len(y_test)),
            **base_metrics,
        }

        for perm in cfg.permutations:
            if perm == "":
                continue
            X_perm, y_perm = apply_permutation(X_test, y_test, perm, rng, cfg.radius, cfg.feature_mode)
            preds_perm, probs_perm = model.predict(X_perm)
            acc = float(np.mean(preds_perm == y_perm))
            row[f"accuracy_perm_{perm}"] = acc
            if probs_perm is not None:
                row[f"log_loss_perm_{perm}"] = log_loss(probs_perm, y_perm)
                row[f"brier_perm_{perm}"] = brier_score(probs_perm, y_perm, len(labels))
                ll_perm = float(np.sum(np.log(np.clip(probs_perm[np.arange(len(y_perm)), y_perm], 1e-12, 1.0))))
                row[f"log_likelihood_perm_{perm}"] = ll_perm

        summary_rows.append(row)

    if not summary_rows:
        raise ValueError("No test data evaluated; nothing to report")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = cfg.outdir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[gof] saved summary → {summary_path}")

    manifest = {
        "train_run": str(cfg.train_run),
        "train_window": cfg.train_window,
        "test_runs": [str(r) for r in test_runs],
        "test_window": cfg.test_window,
        "radius": cfg.radius,
        "feature_mode": cfg.feature_mode,
        "model": cfg.model,
        "max_samples": cfg.max_samples,
        "permutations": cfg.permutations,
        "knn_k": cfg.knn_k,
        "seed": cfg.seed,
    }
    (cfg.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Goodness-of-fit evaluation for local predictors.")
    ap.add_argument("--train-run", required=True)
    ap.add_argument("--train-window", default="")
    ap.add_argument("--test-runs", nargs="*", default=None)
    ap.add_argument("--test-window", default="")
    ap.add_argument("--radius", type=int, default=1)
    ap.add_argument("--model", choices=["logistic", "rule", "markov", "knn"], default="logistic")
    ap.add_argument("--feature-mode", choices=["local", "center", "global"], default="local")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--permutations", default="")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="reports/gof")
    ap.add_argument("--knn-k", type=int, default=5)
    args = ap.parse_args()

    perms = [p.strip() for p in args.permutations.split(",") if p.strip()]

    cfg = GOFConfig(
        train_run=Path(args.train_run),
        train_window=args.train_window or None,
        test_runs=[Path(p) for p in (args.test_runs or [args.train_run])],
        test_window=args.test_window or None,
        radius=args.radius,
        feature_mode=args.feature_mode,
        model=args.model,
        max_samples=args.max_samples if args.max_samples else None,
        permutations=perms,
        seed=args.seed,
        outdir=Path(args.out),
        knn_k=args.knn_k,
    )
    run_gof(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
