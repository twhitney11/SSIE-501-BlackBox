from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from .analyze import load_runs_encoded
from .maskgen import load_masks_from_config


def compute_region_metrics(run_dirs: Sequence[str], config_path: Path, outdir: Path,
                           window: str = "", top_mi_pairs: int = 20, mi_lag: int = 0,
                           fit_labels: Sequence[str] | None = None,
                           fit_models: Sequence[str] | None = None,
                           combos: Sequence[str] | None = None) -> None:
    if not run_dirs:
        raise ValueError("At least one run directory must be provided.")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_states, labels, _label_to_idx, per_run = load_runs_encoded(run_dirs, window=window)
    if not per_run or not per_run[0]:
        raise ValueError("No frames available after applying the window; cannot compute metrics.")

    # Precompute arrays for efficiency
    run_arrays = [np.stack(run, axis=0).astype(np.int16, copy=False) for run in per_run]
    L = len(labels)
    log2 = np.log(2)

    grid, region_masks, _ = load_masks_from_config(Path(config_path), run=run_dirs[0] if run_dirs else None,
                                                   window=window, allow_overlap=None)
    H, W = grid.height, grid.width
    expect_shape = (H, W)

    meta = {
        "runs": [str(r) for r in run_dirs],
        "window": window,
        "config": str(config_path),
    }
    (outdir / "metrics_meta.json").write_text(json.dumps(meta, indent=2))

    summary_rows: List[List[str]] = []
    summary_dicts: List[Dict[str, float]] = []
    seq_cache: Dict[Tuple[int, int], np.ndarray] = {}

    for region_name, region_mask in region_masks.items():
        if region_mask.shape != expect_shape:
            raise ValueError(f"Mask for region '{region_name}' has shape {region_mask.shape}, expected {expect_shape}.")

        row, info = _compute_metrics_for_region(region_name, region_mask, run_arrays, labels,
                                                L, log2, top_mi_pairs, mi_lag, outdir, seq_cache)
        summary_rows.append(row)
        summary_dicts.append(info)

    summary_header = [
        "region",
        "cells",
        "transitions",
        "transitions_changes",
        "self_transition_fraction",
        "cond_entropy_bits",
        "cond_entropy_changes_bits",
        "mi_pairs_evaluated",
        "mean_mi_bits",
        "max_mi_bits",
        "top_pair_mi_bits",
        "top_pair_cells",
    ]
    summary_path = outdir / "region_metrics_summary.csv"
    _write_csv(summary_path, summary_header, summary_rows)
    print(f"[metrics] wrote {summary_path}")

    stable_mask = _compute_stable_mask(run_arrays)
    _plot_stable_cells(stable_mask, outdir)
    overlay_fractions = _plot_stable_cells_by_region(stable_mask, region_masks, outdir)
    for info in summary_dicts:
        region = info["region"]
        info["stable_fraction"] = overlay_fractions.get(region, np.nan)

    _plot_region_metric_bars(summary_dicts, outdir)
    _plot_top_mi_pair_networks(summary_dicts, (H, W), outdir)
    _plot_region_mi_heatmap(summary_dicts, seq_cache, log2, mi_lag, outdir)
    _plot_region_transition_heatmaps(summary_dicts, labels, outdir)
    if combos and "pair" in combos:
        _compute_pairwise_combos(run_arrays, labels, summary_dicts, region_masks, outdir)
    if fit_labels:
        _fit_label_fractions(run_arrays, labels, fit_labels, fit_models, outdir, region_masks)


def _compute_metrics_for_region(region_name: str, mask: np.ndarray, run_arrays: List[np.ndarray],
                                labels: Sequence[str], label_count: int, log2: float,
                                top_mi_pairs: int, mi_lag: int, outdir: Path,
                                seq_cache: Dict[Tuple[int, int], np.ndarray]) -> Tuple[List[str], Dict[str, float]]:
    mask_flat = mask.ravel()
    transitions_all, transitions_changes, total_trans, total_changes = _transition_counts(mask_flat, run_arrays,
                                                                                         label_count)

    cond_entropy_all = _conditional_entropy(transitions_all, total_trans)
    cond_entropy_changes = _conditional_entropy(transitions_changes, total_changes)

    same_transitions = float(transitions_all.diagonal().sum())
    self_fraction = same_transitions / total_trans if total_trans else float("nan")

    coords, sequences = _collect_sequences(mask, run_arrays, cache=seq_cache)
    top_pairs, mean_mi, max_mi, top_mi = _pairwise_mutual_information(coords, sequences, top_mi_pairs, mi_lag, log2)

    region_slug = region_name.replace(" ", "_")
    if top_pairs:
        top_path = outdir / f"{region_slug}_top_mi_pairs.csv"
        top_header = ["rank", "mi_bits", "cell_a_i", "cell_a_j", "cell_b_i", "cell_b_j"]
        top_rows = [
            [str(idx + 1),
             f"{mi:.6f}",
             str(int(a_i)),
             str(int(a_j)),
             str(int(b_i)),
             str(int(b_j))]
            for idx, (mi, (a_i, a_j), (b_i, b_j)) in enumerate(top_pairs)
        ]
        _write_csv(top_path, top_header, top_rows)
        print(f"[metrics] wrote {top_path}")
        top_pair_cells = f"({int(top_pairs[0][1][0])},{int(top_pairs[0][1][1])})-({int(top_pairs[0][2][0])},{int(top_pairs[0][2][1])})"
    else:
        top_pair_cells = ""

    summary_row = [
        region_name,
        str(int(np.count_nonzero(mask))),
        str(int(total_trans)),
        str(int(total_changes)),
        f"{self_fraction:.6f}" if not np.isnan(self_fraction) else "nan",
        f"{cond_entropy_all:.6f}" if not np.isnan(cond_entropy_all) else "nan",
        f"{cond_entropy_changes:.6f}" if not np.isnan(cond_entropy_changes) else "nan",
        str(len(sequences) * (len(sequences) - 1) // 2),
        f"{mean_mi:.6f}",
        f"{max_mi:.6f}",
        f"{top_mi:.6f}",
        top_pair_cells,
    ]
    summary_info = {
        "region": region_name,
        "region_slug": region_slug,
        "cells": float(np.count_nonzero(mask)),
        "transitions": float(total_trans),
        "transitions_changes": float(total_changes),
        "self_transition_fraction": float(self_fraction) if not np.isnan(self_fraction) else np.nan,
        "cond_entropy_bits": float(cond_entropy_all) if not np.isnan(cond_entropy_all) else np.nan,
        "cond_entropy_changes_bits": float(cond_entropy_changes) if not np.isnan(cond_entropy_changes) else np.nan,
        "mean_mi_bits": float(mean_mi),
        "max_mi_bits": float(max_mi),
        "top_pair_mi_bits": float(top_mi),
        "top_pairs": top_pairs,
        "coords": coords,
        "transitions_matrix": transitions_all.copy(),
    }
    return summary_row, summary_info


def _transition_counts(mask_flat: np.ndarray, run_arrays: List[np.ndarray], label_count: int) -> Tuple[np.ndarray, np.ndarray, int, int]:
    counts_all = np.zeros((label_count, label_count), dtype=np.int64)
    counts_changes = np.zeros((label_count, label_count), dtype=np.int64)
    total_transitions = 0
    total_changes = 0

    for arr in run_arrays:
        if arr.shape[0] < 2:
            continue
        prev = arr[:-1].reshape(arr.shape[0] - 1, -1)[:, mask_flat]
        nxt = arr[1:].reshape(arr.shape[0] - 1, -1)[:, mask_flat]

        keys = (prev * label_count + nxt).ravel()
        hist = np.bincount(keys, minlength=label_count * label_count).reshape(label_count, label_count)
        counts_all += hist
        total_transitions += prev.size

        diff = prev != nxt
        if diff.any():
            diff_keys = (prev[diff] * label_count + nxt[diff])
            diff_hist = np.bincount(diff_keys, minlength=label_count * label_count).reshape(label_count, label_count)
            counts_changes += diff_hist
            total_changes += int(diff.sum())

    return counts_all, counts_changes, total_transitions, total_changes


def _conditional_entropy(counts: np.ndarray, total: int) -> float:
    if total == 0:
        return float("nan")
    probs = counts / total
    marginal = probs.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = np.divide(probs, marginal, out=np.zeros_like(probs), where=probs > 0)
        log_cond = np.where(cond > 0, np.log2(cond), 0.0)
    entropy = -float(np.sum(probs * log_cond))
    return entropy


def _collect_sequences(mask: np.ndarray, run_arrays: Iterable[np.ndarray], cache: Dict[Tuple[int, int], np.ndarray] | None = None) -> Tuple[List[Tuple[int, int]], List[np.ndarray]]:
    coords = [tuple(coord) for coord in np.argwhere(mask)]
    sequences: List[np.ndarray] = []
    use_cache = cache is not None
    for coord in coords:
        if use_cache and coord in cache:
            sequences.append(cache[coord])
            continue
        i, j = coord
        seq = np.concatenate([arr[:, i, j] for arr in run_arrays], axis=0).astype(np.int16, copy=False)
        if use_cache:
            cache[coord] = seq
        sequences.append(seq)
    return coords, sequences


def _pairwise_mutual_information(coords: List[Tuple[int, int]], sequences: List[np.ndarray],
                                 top_pairs: int, lag: int, log2: float) -> Tuple[List[Tuple[float, Tuple[int, int], Tuple[int, int]]], float, float, float]:
    if len(sequences) < 2:
        return [], 0.0, 0.0, 0.0

    results: List[Tuple[float, Tuple[int, int], Tuple[int, int]]] = []
    for idx_a, idx_b in itertools.combinations(range(len(sequences)), 2):
        seq_a = sequences[idx_a]
        seq_b = sequences[idx_b]
        if lag > 0:
            if seq_a.size <= lag or seq_b.size <= lag:
                continue
            vals_a = seq_a[lag:]
            vals_b = seq_b[:-lag]
        else:
            vals_a = seq_a
            vals_b = seq_b

        if vals_a.size == 0 or vals_b.size == 0:
            mi_bits = 0.0
        else:
            mi = mutual_info_score(vals_a, vals_b)
            mi_bits = float(mi / log2) if mi > 0 else 0.0
        results.append((mi_bits, coords[idx_a], coords[idx_b]))

    if not results:
        return [], 0.0, 0.0, 0.0

    results.sort(key=lambda item: item[0], reverse=True)
    top = results[:top_pairs]
    mean_mi = float(np.mean([item[0] for item in results]))
    max_mi = float(results[0][0])
    top_mi = float(top[0][0]) if top else 0.0
    return top, mean_mi, max_mi, top_mi


def _compute_stable_mask(run_arrays: List[np.ndarray]) -> np.ndarray:
    if not run_arrays:
        return np.zeros((0, 0), dtype=bool)
    H, W = run_arrays[0].shape[1:]
    changes = np.zeros((H, W), dtype=bool)
    for arr in run_arrays:
        if arr.shape[0] < 2:
            continue
        diff = np.any(arr[1:] != arr[:-1], axis=0)
        changes |= diff
    stable = ~changes
    return stable


def _plot_stable_cells(stable_mask: np.ndarray, outdir: Path) -> None:
    if stable_mask.size == 0:
        return
    cmap = ListedColormap(["#f0f0f0", "#d62728"])
    plt.figure(figsize=(4.5, 4.5))
    plt.imshow(stable_mask.astype(int), cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    plt.title("Cells that never changed")
    H, W = stable_mask.shape
    plt.xticks(range(W))
    plt.yticks(range(H))
    plt.grid(color="gray", linestyle="--", linewidth=0.3, alpha=0.7)
    plt.tick_params(axis="both", which="both", length=0)
    plt.xlim(-0.5, W - 0.5)
    plt.ylim(H - 0.5, -0.5)
    handles = [Patch(facecolor="#d62728", edgecolor="black", label="unchanged")]
    plt.legend(handles=handles, loc="upper right", frameon=True, fontsize=8)
    plt.tight_layout()
    out_path = outdir / "stable_cells.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[metrics] wrote {out_path}")


def _plot_region_metric_bars(summary_dicts: Sequence[Dict[str, float]], outdir: Path) -> None:
    if not summary_dicts:
        return

    regions = [d["region"] for d in summary_dicts]

    def _plot(metric_key: str, ylabel: str, filename: str) -> None:
        values = np.array([d.get(metric_key, np.nan) for d in summary_dicts], dtype=float)
        if np.all(np.isnan(values)):
            return
        positions = np.arange(len(regions))
        plt.figure(figsize=(6, 4))
        plt.bar(positions, values, color="#1f77b4")
        plt.xticks(positions, regions, rotation=45, ha="right")
        plt.ylabel(ylabel)
        plt.title(ylabel)
        plt.grid(axis="y", alpha=0.3, linestyle="--")
        plt.tight_layout()
        out_path = outdir / filename
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[metrics] wrote {out_path}")

    _plot("self_transition_fraction", "Self-transition fraction", "self_transition_fraction.png")
    _plot("cond_entropy_bits", "Conditional entropy (bits)", "cond_entropy_bits.png")
    _plot("cond_entropy_changes_bits", "Conditional entropy (changes only)", "cond_entropy_changes_bits.png")
    _plot("stable_fraction", "Stable cell fraction", "stable_fraction.png")

    transitions = np.array([d.get("transitions", 0.0) for d in summary_dicts], dtype=float)
    changes = np.array([d.get("transitions_changes", 0.0) for d in summary_dicts], dtype=float)
    if np.any(transitions > 0):
        positions = np.arange(len(regions))
        plt.figure(figsize=(6, 4))
        no_change = np.maximum(transitions - changes, 0.0)
        plt.bar(positions, no_change, label="no change", color="#9edae5")
        plt.bar(positions, changes, bottom=no_change, label="changed", color="#17becf")
        plt.xticks(positions, regions, rotation=45, ha="right")
        plt.ylabel("Transitions count")
        plt.title("Transitions vs changes")
        plt.legend()
        plt.grid(axis="y", alpha=0.3, linestyle="--")
        plt.tight_layout()
        out_path = outdir / "transitions_vs_changes.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[metrics] wrote {out_path}")


def _plot_top_mi_pair_networks(summary_dicts: Sequence[Dict[str, float]], shape: Tuple[int, int], outdir: Path) -> None:
    H, W = shape
    if H == 0 or W == 0:
        return

    for info in summary_dicts:
        top_pairs = info.get("top_pairs") or []
        if not top_pairs:
            continue
        slug = info.get("region_slug", info.get("region", "region")).replace(" ", "_")
        values = [mi for mi, _, _ in top_pairs]
        vmin = min(values)
        vmax = max(values)
        norm = plt.Normalize(vmin, vmax) if vmax > vmin else None
        cmap = plt.cm.Reds

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)
        ax.set_xticks(range(W))
        ax.set_yticks(range(H))
        ax.grid(color="gray", linestyle="--", linewidth=0.3, alpha=0.7)
        ax.tick_params(axis="both", which="both", length=0)
        ax.set_title(f"Top MI pairs – {info.get('region')}")

        for mi, (ai, aj), (bi, bj) in top_pairs:
            color = cmap(norm(mi)) if norm else "#d62728"
            width = 0.5 + (4.0 * ((mi - vmin) / (vmax - vmin))) if vmax > vmin else 2.5
            size = 40 + 60 * ((mi - vmin) / (vmax - vmin)) if vmax > vmin else 60
            ax.plot([aj, bj], [ai, bi], color=color, linewidth=width)
            ax.scatter([aj, bj], [ai, bi], s=size, color=color, edgecolors="black")

        if norm:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("MI (bits)")

        fig.tight_layout()
        out_path = outdir / f"{slug}_top_mi_pairs.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[metrics] wrote {out_path}")


def _fit_label_fractions(run_arrays: Sequence[np.ndarray], labels: Sequence[str],
                         selected_labels: Sequence[str], model_names: Sequence[str] | None,
                         outdir: Path,
                         region_masks: Dict[str, np.ndarray]) -> None:
    label_to_idx = {lab: idx for idx, lab in enumerate(labels)}
    chosen = [lab for lab in selected_labels if lab in label_to_idx]
    if not chosen:
        print("[metrics] fit labels not found; skipping label fits")
        return
    models = model_names or ["linear", "exponential", "power"]
    rows = [["region", "label", "model", "param_a", "param_b", "r2"]]
    for region_name, region_mask in region_masks.items():
        mask = region_mask.astype(bool)
        for label in chosen:
            idx = label_to_idx[label]
            times, values = _gather_fraction_samples(run_arrays, idx, mask)
            if times.size < 2:
                continue
            fit_results = _run_model_fits(times, values, models)
            if not fit_results:
                continue
            for res in fit_results:
                rows.append([region_name, label, res["name"],
                             f"{res.get('a', float('nan')):.6f}",
                             f"{res.get('b', float('nan')):.6f}",
                             f"{res['r2']:.6f}"])
            best = max(fit_results, key=lambda r: r["r2"])
            _plot_label_fit(times, values, best, f"{label} ({region_name})", outdir, region_name, label)
    if len(rows) > 1:
        path = outdir / "label_fraction_fits.csv"
        _write_csv(path, rows[0], rows[1:])
        print(f"[metrics] wrote {path}")


def _gather_fraction_samples(run_arrays: Sequence[np.ndarray], label_idx: int, mask: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
    times, values = [], []
    for arr in run_arrays:
        if mask is None:
            fractions = (arr == label_idx).mean(axis=(1, 2))
        else:
            masked = (arr == label_idx)[:, mask]
            fractions = masked.mean(axis=1)
        times.append(np.arange(len(fractions)))
        values.append(fractions)
    if not times:
        return np.array([]), np.array([])
    return np.concatenate(times), np.concatenate(values)


def _run_model_fits(times: np.ndarray, values: np.ndarray, model_names: Sequence[str]) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    for name in model_names:
        if name == "linear":
            res = _fit_linear(times, values)
        elif name == "exponential":
            res = _fit_exponential(times, values)
        elif name == "power":
            res = _fit_power(times, values)
        else:
            continue
        if res:
            results.append(res)
    return results


def _fit_linear(t: np.ndarray, y: np.ndarray) -> Dict[str, float] | None:
    if t.size < 2:
        return None
    coeffs = np.polyfit(t, y, 1)
    pred = coeffs[0] * t + coeffs[1]
    r2 = _r2_score(y, pred)
    return {"name": "linear", "a": coeffs[1], "b": coeffs[0], "pred": pred, "r2": r2}


def _fit_exponential(t: np.ndarray, y: np.ndarray) -> Dict[str, float] | None:
    mask = y > 0
    if np.count_nonzero(mask) < 2:
        return None
    coeffs = np.polyfit(t[mask], np.log(y[mask]), 1)
    pred = np.exp(coeffs[0] * t + coeffs[1])
    r2 = _r2_score(y, pred)
    return {"name": "exponential", "a": np.exp(coeffs[1]), "b": coeffs[0], "pred": pred, "r2": r2}


def _fit_power(t: np.ndarray, y: np.ndarray) -> Dict[str, float] | None:
    mask = (t > 0) & (y > 0)
    if np.count_nonzero(mask) < 2:
        return None
    coeffs = np.polyfit(np.log(t[mask]), np.log(y[mask]), 1)
    pred = np.exp(coeffs[1]) * np.power(np.maximum(t, 1e-9), coeffs[0])
    r2 = _r2_score(y, pred)
    return {"name": "power", "a": np.exp(coeffs[1]), "b": coeffs[0], "pred": pred, "r2": r2}


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_pred)
    if not np.any(mask):
        return float("nan")
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _plot_label_fit(times: np.ndarray, values: np.ndarray, best_fit: Dict[str, float],
                    label: str, outdir: Path, region: str, raw_label: str) -> None:
    order = np.argsort(times)
    t_sorted = times[order]
    y_sorted = values[order]
    pred_sorted = best_fit["pred"][order]
    plt.figure(figsize=(6, 4))
    plt.scatter(t_sorted, y_sorted, s=10, label="data", color="#1f77b4", alpha=0.6)
    plt.plot(t_sorted, pred_sorted, color="#d62728", linewidth=2,
             label=f"{best_fit['name']} fit (R²={best_fit['r2']:.3f})")
    plt.xlabel("Step")
    plt.ylabel("Fraction of grid")
    plt.title(f"Label fraction fit — {label}")
    plt.legend()
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    slug = f"label_fit_{raw_label}_{region.replace(' ', '_')}"
    out_path = outdir / f"{slug}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[metrics] wrote {out_path}")

def _plot_region_mi_heatmap(summary_dicts: Sequence[Dict[str, float]], seq_cache: Dict[Tuple[int, int], np.ndarray],
                            log2: float, mi_lag: int, outdir: Path) -> None:
    if not summary_dicts:
        return
    regions = [info["region"] for info in summary_dicts]
    n = len(regions)
    matrix = np.zeros((n, n), dtype=float)

    for i, info_i in enumerate(summary_dicts):
        matrix[i, i] = info_i.get("top_pair_mi_bits", 0.0)
        coords_i = info_i.get("coords", [])
        for j in range(i + 1, n):
            info_j = summary_dicts[j]
            coords_j = info_j.get("coords", [])
            if not coords_i or not coords_j:
                value = 0.0
            else:
                value = _max_mi_between_regions(coords_i, coords_j, seq_cache, log2, mi_lag)
            matrix[i, j] = matrix[j, i] = value

    if not np.any(matrix):
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=matrix.max())
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Max MI (bits)")
    ax.set_xticks(range(n))
    ax.set_xticklabels(regions, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(regions)
    ax.set_title("Region-to-region MI strength")
    fig.tight_layout()
    out_path = outdir / "region_mi_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[metrics] wrote {out_path}")


def _max_mi_between_regions(coords_a: Sequence[Tuple[int, int]], coords_b: Sequence[Tuple[int, int]],
                             seq_cache: Dict[Tuple[int, int], np.ndarray], log2: float, mi_lag: int) -> float:
    max_mi = 0.0
    for coord_a in coords_a:
        seq_a = seq_cache.get(coord_a)
        if seq_a is None:
            continue
        for coord_b in coords_b:
            seq_b = seq_cache.get(coord_b)
            if seq_b is None:
                continue
            mi_val = _mutual_information_bits(seq_a, seq_b, log2, mi_lag)
            if mi_val > max_mi:
                max_mi = mi_val
    return max_mi


def _mutual_information_bits(seq_a: np.ndarray, seq_b: np.ndarray, log2: float, lag: int) -> float:
    if lag > 0:
        if seq_a.size <= lag or seq_b.size <= lag:
            return 0.0
        vals_a = seq_a[lag:]
        vals_b = seq_b[:-lag]
    else:
        vals_a = seq_a
        vals_b = seq_b
    if vals_a.size == 0 or vals_b.size == 0:
        return 0.0
    mi = mutual_info_score(vals_a, vals_b)
    return float(mi / log2) if mi > 0 else 0.0


def _plot_stable_cells_by_region(stable_mask: np.ndarray, region_masks: Dict[str, np.ndarray], outdir: Path) -> Dict[str, float]:
    fractions: Dict[str, float] = {}
    if stable_mask.size == 0 or not region_masks:
        return fractions

    region_names = list(region_masks.keys())
    H, W = stable_mask.shape
    region_map = np.full((H, W), -1, dtype=int)
    for idx, name in enumerate(region_names):
        mask = region_masks[name].astype(bool)
        region_map[mask] = idx

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(region_names), 1)))
    display = np.ones((H, W, 4), dtype=float)

    for idx, name in enumerate(region_names):
        mask = (region_map == idx) & stable_mask
        total_cells = int(np.count_nonzero(region_masks[name]))
        stable_cells = int(np.count_nonzero(mask))
        fraction = stable_cells / total_cells if total_cells else np.nan
        fractions[name] = fraction
        display[mask] = colors[idx % len(colors)]

    plt.figure(figsize=(4.5, 4.5))
    plt.imshow(display, interpolation="nearest")
    plt.title("Stable cells by region")
    plt.xticks(range(W))
    plt.yticks(range(H))
    plt.grid(color="gray", linestyle="--", linewidth=0.3, alpha=0.7)
    plt.tick_params(axis="both", which="both", length=0)
    plt.xlim(-0.5, W - 0.5)
    plt.ylim(H - 0.5, -0.5)
    handles = []
    for idx, name in enumerate(region_names):
        frac = fractions.get(name)
        if np.isnan(frac):
            continue
        handles.append(Patch(facecolor=colors[idx % len(colors)], edgecolor="black", label=f"{name}: {frac:.2%}"))
    if handles:
        plt.legend(handles=handles, loc="upper right", frameon=True, fontsize=8)
    plt.tight_layout()
    out_path = outdir / "stable_cells_regions.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[metrics] wrote {out_path}")
    return fractions


def _plot_region_transition_heatmaps(summary_dicts: Sequence[Dict[str, float]], labels: Sequence[str], outdir: Path) -> None:
    if not summary_dicts:
        return
    for info in summary_dicts:
        matrix = info.get("transitions_matrix")
        if matrix is None or not np.any(matrix):
            continue
        slug = info.get("region_slug", info.get("region", "region")).replace(" ", "_")
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(matrix, cmap="magma", interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Transition count")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_title(f"Transitions heatmap – {info.get('region')}")
        fig.tight_layout()
        out_path = outdir / f"{slug}_transition_heatmap.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[metrics] wrote {out_path}")


def _compute_pairwise_combos(run_arrays: Sequence[np.ndarray], labels: Sequence[str],
                             summary_dicts: Sequence[Dict[str, float]],
                             region_masks: Dict[str, np.ndarray], outdir: Path) -> None:
    label_count = len(labels)
    for info in summary_dicts:
        region = info["region"]
        mask = region_masks.get(region)
        if mask is None:
            continue
        mask_bool = mask.astype(bool)
        counts = np.zeros((label_count, label_count), dtype=np.int64)
        horiz_idx = np.argwhere(mask_bool[:, :-1] & mask_bool[:, 1:])
        vert_idx = np.argwhere(mask_bool[:-1, :] & mask_bool[1:, :])
        for arr in run_arrays:
            if horiz_idx.size:
                left = arr[:, horiz_idx[:, 0], horiz_idx[:, 1]]
                right = arr[:, horiz_idx[:, 0], horiz_idx[:, 1] + 1]
                np.add.at(counts, (left.ravel(), right.ravel()), 1)
            if vert_idx.size:
                top = arr[:, vert_idx[:, 0], vert_idx[:, 1]]
                bottom = arr[:, vert_idx[:, 0] + 1, vert_idx[:, 1]]
                np.add.at(counts, (top.ravel(), bottom.ravel()), 1)
        total_pairs = counts.sum()
        if total_pairs == 0:
            continue
        freq = _region_label_frequency(run_arrays, mask_bool, label_count)
        expected = np.outer(freq, freq) * total_pairs
        chi = np.zeros_like(expected, dtype=float)
        valid = expected > 0
        chi[valid] = (counts[valid] - expected[valid]) ** 2 / expected[valid]
        chi_stat = float(np.nansum(chi))
        _write_pairwise_csv(region, labels, counts, expected, chi, outdir)
        _plot_pairwise_heatmap(region, labels, counts, chi_stat, outdir)


def _region_label_frequency(run_arrays: Sequence[np.ndarray], mask: np.ndarray, label_count: int) -> np.ndarray:
    counts = np.zeros(label_count, dtype=np.int64)
    for arr in run_arrays:
        vals = arr[:, mask]
        unique, cnt = np.unique(vals, return_counts=True)
        counts[unique] += cnt
    total = counts.sum()
    if total == 0:
        return np.zeros(label_count, dtype=float)
    return counts / total


def _write_pairwise_csv(region: str, labels: Sequence[str], counts: np.ndarray,
                        expected: np.ndarray, chi: np.ndarray, outdir: Path) -> None:
    slug = region.replace(" ", "_")
    header = ["label_a", "label_b", "observed", "expected", "chi_contrib"]
    rows = []
    L = len(labels)
    for i in range(L):
        for j in range(L):
            rows.append([labels[i], labels[j], str(int(counts[i, j])),
                         f"{expected[i, j]:.4f}", f"{chi[i, j]:.4f}"])
    path = outdir / f"{slug}_pairwise_counts.csv"
    _write_csv(path, header, rows)
    print(f"[metrics] wrote {path}")


def _plot_pairwise_heatmap(region: str, labels: Sequence[str], counts: np.ndarray,
                           chi_stat: float, outdir: Path) -> None:
    slug = region.replace(" ", "_")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(counts, cmap="plasma")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Adjacency count")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title(f"Adjacency counts – {region}\nChi-square={chi_stat:.2f}")
    fig.tight_layout()
    out_path = outdir / f"{slug}_pairwise_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[metrics] wrote {out_path}")


def _compute_pairwise_combos(run_arrays: Sequence[np.ndarray], labels: Sequence[str],
                             summary_dicts: Sequence[Dict[str, float]],
                             region_masks: Dict[str, np.ndarray], outdir: Path) -> None:
    label_count = len(labels)
    for info in summary_dicts:
        region = info["region"]
        mask = region_masks.get(region)
        if mask is None:
            continue
        mask_bool = mask.astype(bool)
        counts = np.zeros((label_count, label_count), dtype=np.int64)
        horiz_idx = np.argwhere(mask_bool[:, :-1] & mask_bool[:, 1:])
        vert_idx = np.argwhere(mask_bool[:-1, :] & mask_bool[1:, :])
        for arr in run_arrays:
            if horiz_idx.size:
                left = arr[:, horiz_idx[:, 0], horiz_idx[:, 1]]
                right = arr[:, horiz_idx[:, 0], horiz_idx[:, 1] + 1]
                np.add.at(counts, (left.ravel(), right.ravel()), 1)
            if vert_idx.size:
                top = arr[:, vert_idx[:, 0], vert_idx[:, 1]]
                bottom = arr[:, vert_idx[:, 0] + 1, vert_idx[:, 1]]
                np.add.at(counts, (top.ravel(), bottom.ravel()), 1)
        total_pairs = counts.sum()
        if total_pairs == 0:
            continue
        freq = _region_label_frequency(run_arrays, mask_bool, label_count)
        expected = np.outer(freq, freq) * total_pairs
        chi = np.zeros_like(expected, dtype=float)
        valid = expected > 0
        chi[valid] = (counts[valid] - expected[valid]) ** 2 / expected[valid]
        chi_stat = float(np.nansum(chi))
        _write_pairwise_csv(region, labels, counts, expected, chi, outdir)
        _plot_pairwise_heatmap(region, labels, counts, chi_stat, outdir)


def _plot_region_mi_heatmap(summary_dicts: Sequence[Dict[str, float]], seq_cache: Dict[Tuple[int, int], np.ndarray],
                            log2: float, mi_lag: int, outdir: Path) -> None:
    if not summary_dicts:
        return
    regions = [info["region"] for info in summary_dicts]
    n = len(regions)
    matrix = np.zeros((n, n), dtype=float)

    for i, info_i in enumerate(summary_dicts):
        matrix[i, i] = info_i.get("top_pair_mi_bits", 0.0)
        coords_i = info_i.get("coords", [])
        for j in range(i + 1, n):
            info_j = summary_dicts[j]
            coords_j = info_j.get("coords", [])
            if not coords_i or not coords_j:
                value = 0.0
            else:
                value = _max_mi_between_regions(coords_i, coords_j, seq_cache, log2, mi_lag)
            matrix[i, j] = matrix[j, i] = value

    vmax = matrix.max() if matrix.size else 0.0
    if vmax <= 0:
        return
    plt.figure(figsize=(6, 5))
    im = plt.imshow(matrix, cmap="viridis", vmin=0.0, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Max MI (bits)")
    plt.xticks(range(n), regions, rotation=45, ha="right")
    plt.yticks(range(n), regions)
    plt.title("Region-to-region MI strength (max bits)")
    plt.tight_layout()
    out_path = outdir / "region_mi_heatmap.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[metrics] wrote {out_path}")


def _max_mi_between_regions(coords_a: Sequence[Tuple[int, int]], coords_b: Sequence[Tuple[int, int]],
                             seq_cache: Dict[Tuple[int, int], np.ndarray], log2: float, mi_lag: int) -> float:
    max_mi = 0.0
    for coord_a in coords_a:
        seq_a = seq_cache.get(coord_a)
        if seq_a is None:
            continue
        for coord_b in coords_b:
            seq_b = seq_cache.get(coord_b)
            if seq_b is None:
                continue
            mi_val = _mutual_information_bits(seq_a, seq_b, log2, mi_lag)
            if mi_val > max_mi:
                max_mi = mi_val
    return max_mi


def _mutual_information_bits(seq_a: np.ndarray, seq_b: np.ndarray, log2: float, lag: int) -> float:
    if lag > 0:
        if seq_a.size <= lag or seq_b.size <= lag:
            return 0.0
        vals_a = seq_a[lag:]
        vals_b = seq_b[:-lag]
    else:
        vals_a = seq_a
        vals_b = seq_b
    if vals_a.size == 0 or vals_b.size == 0:
        return 0.0
    mi = mutual_info_score(vals_a, vals_b)
    return float(mi / log2) if mi > 0 else 0.0

def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(row) + "\n")
