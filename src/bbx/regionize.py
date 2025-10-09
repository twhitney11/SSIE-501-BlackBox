from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from .process import load_run_encoded
from .gridmaps import (
    per_cell_change_rate,
    per_cell_entropy,
    per_cell_conditional_entropy,
    per_cell_phase_flip_rate,
)


@dataclass
class RegionizeConfig:
    run_dir: Path
    outdir: Path
    window: str
    label: str
    threshold: float
    wall_thickness: int
    flip_period: int


def parse_window(window: str, total: int) -> Tuple[int, int]:
    if not window:
        return 0, total
    window = window.strip()
    if window.startswith("last:"):
        N = int(window.split(":")[1])
        start = max(0, total - N)
        return start, total
    if ":" in window:
        start_str, end_str = window.split(":", 1)
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else total
    else:
        start = int(window)
        end = total
    start = max(0, min(start, total - 1 if total else 0))
    end = max(start + 1, min(end, total))
    return start, end


def binary_dilation(mask: np.ndarray, steps: int = 1) -> np.ndarray:
    res = mask.copy()
    for _ in range(max(steps, 0)):
        expanded = res.copy()
        expanded[1:, :] |= res[:-1, :]
        expanded[:-1, :] |= res[1:, :]
        expanded[:, 1:] |= res[:, :-1]
        expanded[:, :-1] |= res[:, 1:]
        res = expanded
    return res


def binary_erosion(mask: np.ndarray, steps: int = 1) -> np.ndarray:
    res = mask.copy()
    for _ in range(max(steps, 0)):
        eroded = res.copy()
        eroded &= np.vstack((res[1:, :], np.zeros((1, res.shape[1]), dtype=bool)))
        eroded &= np.vstack((np.zeros((1, res.shape[1]), dtype=bool), res[:-1, :]))
        eroded &= np.hstack((res[:, 1:], np.zeros((res.shape[0], 1), dtype=bool)))
        eroded &= np.hstack((np.zeros((res.shape[0], 1), dtype=bool), res[:, :-1]))
        res = eroded
    return res


def flood_fill_outside(wall: np.ndarray) -> np.ndarray:
    H, W = wall.shape
    outside = np.zeros_like(wall, dtype=bool)
    q: deque[Tuple[int, int]] = deque()
    for i in range(H):
        for j in (0, W - 1):
            if not wall[i, j] and not outside[i, j]:
                outside[i, j] = True
                q.append((i, j))
    for j in range(W):
        for i in (0, H - 1):
            if not wall[i, j] and not outside[i, j]:
                outside[i, j] = True
                q.append((i, j))
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while q:
        i, j = q.popleft()
        for di, dj in offsets:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and not wall[ni, nj] and not outside[ni, nj]:
                outside[ni, nj] = True
                q.append((ni, nj))
    return outside


def distance_from_wall(seed_mask: np.ndarray, region_mask: np.ndarray) -> np.ndarray:
    H, W = region_mask.shape
    dist = np.full((H, W), np.inf, dtype=float)
    q: deque[Tuple[int, int]] = deque()
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(H):
        for j in range(W):
            if seed_mask[i, j]:
                dist[i, j] = 0.0
                q.append((i, j))
    while q:
        i, j = q.popleft()
        for di, dj in offsets:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                if not region_mask[ni, nj]:
                    continue
                if dist[ni, nj] > dist[i, j] + 1:
                    dist[ni, nj] = dist[i, j] + 1
                    q.append((ni, nj))
    return dist


def write_scalar_csv(path: Path, name: str, grid: np.ndarray) -> None:
    H, W = grid.shape
    rows = [{"i": int(i), "j": int(j), name: float(grid[i, j])} for i in range(H) for j in range(W)]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[regionize] saved {path}")


def write_mask_csv(path: Path, name: str, mask: np.ndarray) -> None:
    H, W = mask.shape
    rows = [{"i": int(i), "j": int(j), name: int(mask[i, j])} for i in range(H) for j in range(W)]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[regionize] saved {path}")


def save_occupancy_plot(occ: np.ndarray, wall_mask: np.ndarray, cfg: RegionizeConfig, run_name: str) -> None:
    plt.figure(figsize=(6, 6))
    im = plt.imshow(occ, cmap="inferno", interpolation="nearest", vmin=0.0, vmax=1.0)
    plt.colorbar(im, fraction=0.046, pad=0.04, label=f"{cfg.label} occupancy")
    plt.contour(wall_mask, levels=[0.5], colors="cyan", linewidths=1.2)
    plt.title(f"{run_name}: {cfg.label} occupancy\nwindow={cfg.window or 'full'} threshold={cfg.threshold}")
    plt.axis("off")
    out_path = cfg.outdir / f"{run_name}_occupancy_{cfg.label}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[regionize] saved {out_path}")


def save_region_overlay(wall: np.ndarray, inside: np.ndarray, outside: np.ndarray,
                        wall_inner: np.ndarray, wall_outer: np.ndarray,
                        cfg: RegionizeConfig, run_name: str) -> None:
    H, W = wall.shape
    region_codes = np.zeros((H, W), dtype=int)
    region_codes[outside] = 1
    region_codes[inside] = 2
    region_codes[wall_outer] = 3
    region_codes[wall_inner] = 4
    region_codes[wall] = 5
    cmap = ListedColormap([
        "#222222",  # other
        "#1f77b4",  # outside
        "#2ca02c",  # inside
        "#8c564b",  # wall outer
        "#bcbd22",  # wall inner
        "#d62728",  # wall
    ])
    plt.figure(figsize=(6, 6))
    plt.imshow(region_codes, cmap=cmap, interpolation="nearest", vmin=0, vmax=5)
    plt.title(f"{run_name}: region partition")
    plt.axis("off")
    out_path = cfg.outdir / f"{run_name}_regions.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[regionize] saved {out_path}")


def save_metric_heatmap(grid: np.ndarray, mask: np.ndarray, title: str,
                        filename: Path, vmin=None, vmax=None) -> None:
    plt.figure(figsize=(6, 5))
    display = np.where(mask, grid, np.nan)
    im = plt.imshow(display, cmap="viridis", interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.axis("off")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[regionize] saved {filename}")


def save_region_summary(stats: list[Dict[str, float]], out_path: Path, metric: str, title: str) -> None:
    df = pd.DataFrame(stats)
    if metric not in df.columns:
        return
    plt.figure(figsize=(6, 3.5))
    plt.bar(df["region"], df[metric])
    plt.ylabel(metric.replace("_", " "))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[regionize] saved {out_path}")


def run_regionize(cfg: RegionizeConfig) -> Dict[str, Dict[str, float]]:
    states, labels, label_to_idx = load_run_encoded(cfg.run_dir)
    if cfg.label not in label_to_idx:
        raise ValueError(f"Label '{cfg.label}' not found in run {cfg.run_dir}. Available: {labels}")

    total_steps = len(states)
    t0, t1 = parse_window(cfg.window, total_steps)
    window_states = states[t0:t1]
    if len(window_states) == 0:
        raise ValueError(f"Empty window ({cfg.window}) for run with {total_steps} steps.")

    label_idx = label_to_idx[cfg.label]
    H, W = window_states[0].shape
    occ = np.zeros((H, W), dtype=float)
    for S in window_states:
        occ += (S == label_idx)
    occ /= len(window_states)

    wall0 = (occ >= cfg.threshold)
    wall0 = binary_dilation(wall0, max(cfg.wall_thickness - 1, 0)) if cfg.wall_thickness > 1 else wall0.copy()

    outside_mask = flood_fill_outside(wall0)
    inside_mask = (~wall0) & (~outside_mask)

    boundary_mask = np.zeros((H, W), dtype=bool)
    boundary_mask[0, :] = boundary_mask[-1, :] = True
    boundary_mask[:, 0] = boundary_mask[:, -1] = True

    errors = []
    if np.any(wall0 & inside_mask):
        errors.append("wall ∩ inside ≠ ∅")
    if np.any(wall0 & outside_mask):
        errors.append("wall ∩ outside ≠ ∅")
    if np.any(inside_mask & outside_mask):
        errors.append("inside ∩ outside ≠ ∅")
    coverage = int(
        wall0.sum(dtype=np.int64)
        + inside_mask.sum(dtype=np.int64)
        + outside_mask.sum(dtype=np.int64)
    )
    if coverage != H * W:
        errors.append(f"coverage mismatch ({coverage} vs {H * W})")
    if not np.any(outside_mask & boundary_mask):
        errors.append("outside does not touch the boundary")
    if np.any(inside_mask & boundary_mask):
        errors.append("inside touches the boundary")

    def save_debug(path: Path) -> None:
        region_codes = np.zeros((H, W), dtype=int)
        region_codes[outside_mask] = 1
        region_codes[inside_mask] = 2
        region_codes[wall0] = 3
        cmap = ListedColormap(["#222222", "#1f77b4", "#2ca02c", "#d62728"])
        plt.figure(figsize=(6, 6))
        plt.imshow(region_codes, cmap=cmap, interpolation="nearest", vmin=0, vmax=3)
        plt.title("Region partition (0=other,1=outside,2=inside,3=wall)")
        plt.axis("off")
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    if errors:
        debug_path = cfg.outdir / f"{cfg.run_dir.name}_region_debug.png"
        save_debug(debug_path)
        message = "; ".join(errors) + f". Debug written to {debug_path}"
        raise ValueError(f"[regionize] Partition validation failed: {message}")

    dilated_wall = binary_dilation(wall0, 1)
    wall_band_ring = dilated_wall & (~wall0)
    wall_inner = inside_mask & wall_band_ring
    wall_outer = outside_mask & wall_band_ring

    entropy = per_cell_entropy(window_states, len(labels))
    cond_entropy = per_cell_conditional_entropy(window_states, len(labels)) if len(window_states) > 1 else np.full((H, W), np.nan)
    flip_rate = per_cell_phase_flip_rate(window_states, cfg.flip_period) if len(window_states) > 1 else np.full((H, W), np.nan)
    change_rate = per_cell_change_rate(window_states) if len(window_states) > 1 else np.full((H, W), np.nan)

    region_map = np.full((H, W), "other", dtype=object)
    region_map[outside_mask] = "outside"
    region_map[inside_mask] = "inside"
    region_map[wall_outer] = "wall_outer"
    region_map[wall_inner] = "wall_inner"
    region_map[wall0] = "wall"

    region_masks = {
        "wall": wall0,
        "wall_inner": wall_inner,
        "wall_outer": wall_outer,
        "inside": inside_mask,
        "outside": outside_mask,
    }

    stats = []
    for name, mask in region_masks.items():
        count = int(mask.sum())
        entry = {
            "region": name,
            "cell_count": count,
            "mean_occupancy": float(occ[mask].mean()) if count else None,
            "mean_entropy": float(entropy[mask].mean()) if count else None,
            "mean_conditional_entropy": float(np.nanmean(cond_entropy[mask])) if count else None,
            "mean_flip_rate": float(np.nanmean(flip_rate[mask])) if count else None,
            "mean_change_rate": float(np.nanmean(change_rate[mask])) if count else None,
        }
        stats.append(entry)

    cfg.outdir.mkdir(parents=True, exist_ok=True)
    run_name = cfg.run_dir.name

    write_scalar_csv(cfg.outdir / f"{run_name}_occupancy_{cfg.label}.csv", "occupancy", occ)
    write_mask_csv(cfg.outdir / f"{run_name}_region_wall.csv", "is_wall", wall0)
    write_mask_csv(cfg.outdir / f"{run_name}_region_inside.csv", "is_inside", inside_mask)
    write_mask_csv(cfg.outdir / f"{run_name}_region_outside.csv", "is_outside", outside_mask)
    write_mask_csv(cfg.outdir / f"{run_name}_region_wall_inner.csv", "is_wall_inner", wall_inner)
    write_mask_csv(cfg.outdir / f"{run_name}_region_wall_outer.csv", "is_wall_outer", wall_outer)

    region_rows = [{"i": int(i), "j": int(j), "region": region_map[i, j]} for i in range(H) for j in range(W)]
    pd.DataFrame(region_rows).to_csv(cfg.outdir / f"{run_name}_region_labels.csv", index=False)
    print(f"[regionize] saved {cfg.outdir / f'{run_name}_region_labels.csv'}")

    save_occupancy_plot(occ, wall0, cfg, run_name)
    save_region_overlay(wall0, inside_mask, outside_mask, wall_inner, wall_outer, cfg, run_name)
    full_mask = np.ones_like(entropy, dtype=bool)
    save_metric_heatmap(entropy, full_mask, f"{run_name}: entropy", cfg.outdir / f"{run_name}_entropy_heatmap.png", vmin=0.0)
    save_metric_heatmap(cond_entropy, full_mask, f"{run_name}: conditional entropy", cfg.outdir / f"{run_name}_cond_entropy_heatmap.png", vmin=0.0)
    save_metric_heatmap(flip_rate, full_mask, f"{run_name}: flip rate", cfg.outdir / f"{run_name}_flip_rate_heatmap.png", vmin=0.0, vmax=1.0)
    save_metric_heatmap(change_rate, full_mask, f"{run_name}: change rate", cfg.outdir / f"{run_name}_change_rate_heatmap.png", vmin=0.0)

    save_region_summary(stats, cfg.outdir / f"{run_name}_region_cell_counts.png", "cell_count", f"{run_name}: region cell counts")
    save_region_summary(stats, cfg.outdir / f"{run_name}_region_mean_entropy.png", "mean_entropy", f"{run_name}: mean entropy by region")
    save_region_summary(stats, cfg.outdir / f"{run_name}_region_mean_flip_rate.png", "mean_flip_rate", f"{run_name}: mean flip rate by region")

    meta = {
        "run": str(cfg.run_dir),
        "window": [t0, t1],
        "label": cfg.label,
        "threshold": cfg.threshold,
        "wall_thickness": cfg.wall_thickness,
        "flip_period": cfg.flip_period,
        "stats": stats,
    }
    stats_path = cfg.outdir / f"{run_name}_region_stats.json"
    stats_path.write_text(json.dumps(meta, indent=2))
    print(f"[regionize] saved {stats_path}")

    stats_csv = cfg.outdir / f"{run_name}_region_stats.csv"
    pd.DataFrame(stats).to_csv(stats_csv, index=False)
    print(f"[regionize] saved {stats_csv}")

    return {entry["region"]: entry for entry in stats}


def main() -> None:
    ap = argparse.ArgumentParser(description="Identify wall/inside/outside regions based on label occupancy.")
    ap.add_argument("--run", required=True, help="Run directory (e.g., data/run_000)")
    ap.add_argument("--out", default="reports", help="Output directory (default: reports)")
    ap.add_argument("--window", default="", help="Time window 'start:end' or 'last:N'. Default: full run.")
    ap.add_argument("--label", default="gru", help="Label to treat as wall (default: gru)")
    ap.add_argument("--threshold", type=float, default=0.5, help="Occupancy threshold to mark wall (default: 0.5)")
    ap.add_argument("--wall-thickness", type=int, default=2, help="Wall band thickness in cells (default: 2)")
    ap.add_argument("--flip-period", type=int, default=2, help="Phase period k for flip-rate metric (default: 2)")
    args = ap.parse_args()

    cfg = RegionizeConfig(
        run_dir=Path(args.run),
        outdir=Path(args.out),
        window=args.window,
        label=args.label,
        threshold=args.threshold,
        wall_thickness=max(1, args.wall_thickness),
        flip_period=max(1, args.flip_period),
    )
    run_regionize(cfg)


if __name__ == "__main__":
    main()
