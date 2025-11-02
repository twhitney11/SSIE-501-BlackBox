from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Iterable, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from itertools import cycle

from scipy.ndimage import (
    binary_opening,
    binary_closing,
    binary_dilation as sp_binary_dilation,
    binary_erosion as sp_binary_erosion,
    binary_fill_holes,
    label as nd_label,
    generate_binary_structure,
)

from .process import load_run_encoded
from .gridmaps import (
    per_cell_change_rate,
    per_cell_entropy,
    per_cell_conditional_entropy,
    per_cell_phase_flip_rate,
)
from .maskgen import load_masks_from_config
try:
    from .utils import parse_window
except ImportError:  # pragma: no cover
    from utils import parse_window

CROSS4 = generate_binary_structure(2, 1)
SQUARE8 = generate_binary_structure(2, 2)


@dataclass
class RegionizeConfig:
    run_dir: Path
    outdir: Path
    window: str
    label: str
    threshold: float
    wall_thickness: int
    flip_period: int
    mask_config: Path | None = None


@dataclass
class OverlaySpec:
    mask: np.ndarray
    color: str
    label: str | None = None


def dilate(mask: np.ndarray, iterations: int = 1, structure=CROSS4) -> np.ndarray:
    if iterations <= 0:
        return mask.astype(bool)
    return sp_binary_dilation(mask.astype(bool), structure=structure, iterations=iterations).astype(bool)


def erode(mask: np.ndarray, iterations: int = 1, structure=CROSS4) -> np.ndarray:
    if iterations <= 0:
        return mask.astype(bool)
    return sp_binary_erosion(mask.astype(bool), structure=structure, iterations=iterations).astype(bool)


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    if not mask.any():
        return mask
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    keep = mask.copy()
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(H):
        for j in range(W):
            if not mask[i, j] or visited[i, j]:
                continue
            queue = deque([(i, j)])
            component = []
            visited[i, j] = True
            while queue:
                ci, cj = queue.popleft()
                component.append((ci, cj))
                for di, dj in offsets:
                    ni, nj = ci + di, cj + dj
                    if 0 <= ni < H and 0 <= nj < W and mask[ni, nj] and not visited[ni, nj]:
                        visited[ni, nj] = True
                        queue.append((ni, nj))
            if len(component) < min_size:
                for ci, cj in component:
                    keep[ci, cj] = False
    return keep


def touches_border(mask: np.ndarray) -> bool:
    H, W = mask.shape
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    return bool((mask & border).any())


def keepout_frame(mask: np.ndarray, margin: int = 1) -> np.ndarray:
    if margin <= 0:
        return mask.astype(bool)
    m = mask.astype(bool).copy()
    m[:margin, :] = False
    m[-margin:, :] = False
    m[:, :margin] = False
    m[:, -margin:] = False
    return m


def largest_component(mask: np.ndarray, conn=CROSS4) -> np.ndarray:
    L, n = nd_label(mask.astype(bool), structure=conn)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(L.ravel())[1:]
    best_id = int(1 + sizes.argmax())
    return L == best_id


def build_wall_seed(
    occ_black: np.ndarray,
    change_rate: np.ndarray,
    occ_topA: np.ndarray,
    occ_topB: np.ndarray,
    thr_black: float = 0.35,
    prc_change: float = 85.0,
    thr_core: float = 0.60,
    keepout: int = 1,
) -> np.ndarray:
    occ_black = occ_black.astype(float)
    change_rate = change_rate.astype(float)
    occ_topA = occ_topA.astype(float)
    occ_topB = occ_topB.astype(float)

    if np.allclose(change_rate, change_rate.flat[0]):
        cue_cr = np.zeros_like(change_rate, dtype=bool)
    else:
        cr_cut = np.percentile(change_rate, prc_change)
        cue_cr = change_rate >= cr_cut

    coreA = binary_opening(occ_topA >= thr_core, structure=CROSS4)
    coreB = binary_opening(occ_topB >= thr_core, structure=CROSS4)
    boundary = dilate(coreA, iterations=1) ^ dilate(coreB, iterations=1)

    cue_blk = occ_black >= thr_black

    seed = (cue_cr | boundary | cue_blk)
    seed = keepout_frame(seed, margin=keepout)
    seed = binary_opening(seed, structure=CROSS4)
    seed = binary_closing(seed, structure=CROSS4)

    L, n = nd_label(seed, structure=CROSS4)
    for cid in range(1, n + 1):
        comp = L == cid
        if touches_border(comp):
            seed[comp] = False

    return seed.astype(bool)


def _flood_outside(comp: np.ndarray) -> np.ndarray:
    H, W = comp.shape
    outside = np.zeros_like(comp, dtype=bool)

    def enqueue(i: int, j: int, queue: deque):
        if comp[i, j] and not outside[i, j]:
            outside[i, j] = True
            queue.append((i, j))

    q: deque[Tuple[int, int]] = deque()
    for j in range(W):
        enqueue(0, j, q)
        enqueue(H - 1, j, q)
    for i in range(H):
        enqueue(i, 0, q)
        enqueue(i, W - 1, q)

    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while q:
        i, j = q.popleft()
        for di, dj in offsets:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and comp[ni, nj] and not outside[ni, nj]:
                outside[ni, nj] = True
                q.append((ni, nj))
    return outside


def select_wall_ring_or_fallback(seed: np.ndarray, thickness: int = 2) -> np.ndarray:
    seed = seed.astype(bool)
    cross = CROSS4
    L, num = nd_label(seed, structure=cross)
    best = None
    best_score = -np.inf
    for cid in range(1, num + 1):
        comp = L == cid
        if touches_border(comp):
            continue
        filled = binary_fill_holes(comp)
        inside = filled & (~comp)
        inside_area = int(inside.sum())
        comp_area = int(comp.sum())
        if inside_area == 0:
            continue
        score = inside_area - 0.2 * comp_area
        if score > best_score:
            best_score = score
            best = comp

    if best is not None:
        return dilate(best, iterations=max(0, thickness - 1), structure=cross)

    # Fallback path
    barrier = dilate(seed, iterations=1, structure=cross)
    barrier = keepout_frame(barrier, margin=1)

    comp = ~(barrier.astype(bool))
    outside = _flood_outside(comp)

    remainder = ~(outside | barrier)
    remainder = keepout_frame(remainder, margin=1)

    Lr, nr = nd_label(remainder, structure=cross)
    inside = np.zeros_like(remainder, dtype=bool)
    best_area = -1
    for cid in range(1, nr + 1):
        comp = Lr == cid
        if touches_border(comp):
            continue
        area = int(comp.sum())
        if area > best_area:
            inside = comp
            best_area = area
    if best_area <= 0:
        raise ValueError("Fallback failed: could not identify a non-border inside component.")

    border_in = dilate(inside, iterations=1, structure=cross)
    border_out = dilate(outside, iterations=1, structure=cross)
    interface = border_in & border_out
    ring = dilate(interface, iterations=max(0, thickness - 1), structure=cross)
    ring = keepout_frame(ring, margin=1)
    return ring.astype(bool)


def partition_from_ring(ring: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ring = ring.astype(bool)
    comp = ~ring
    outside = _flood_outside(comp)
    inside = ~(ring | outside)

    H, W = ring.shape
    total = int(ring.sum(dtype=np.int64) + outside.sum(dtype=np.int64) + inside.sum(dtype=np.int64))
    if total != H * W:
        raise ValueError(f"Coverage failed: {total} != {H*W}")
    if np.any(ring & outside) or np.any(ring & inside) or np.any(inside & outside):
        raise ValueError("Partition masks overlap after ring segmentation")
    if not touches_border(outside):
        raise ValueError("Outside region does not touch the boundary")
    return inside.astype(bool), ring.astype(bool), outside.astype(bool)


def neighborhood_entropy(states: np.ndarray, mask: np.ndarray, radius: int = 1) -> float:
    offsets = range(-radius, radius + 1)
    counts = Counter()
    total = 0
    for state in states:
        H, W = state.shape
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
                counts[tuple(neigh)] += 1
                total += 1
    if total == 0:
        return float("nan")
    ent = 0.0
    for cnt in counts.values():
        p = cnt / total
        ent -= p * np.log2(p)
    return float(ent)


def conditional_entropy_region(states: np.ndarray, mask: np.ndarray, radius: int = 1) -> float:
    offsets = range(-radius, radius + 1)
    counts = Counter()
    next_counts: Dict[Tuple[int, ...], Counter] = {}
    total = 0
    for t in range(len(states) - 1):
        S = states[t]
        T = states[t + 1]
        H, W = S.shape
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
                key = tuple(neigh)
                counts[key] += 1
                if key not in next_counts:
                    next_counts[key] = Counter()
                next_counts[key][int(T[i, j])] += 1
                total += 1
    if total == 0:
        return float("nan")
    ent = 0.0
    for key, cnt in counts.items():
        token_total = cnt
        p_token = token_total / total
        label_counts = next_counts.get(key, Counter())
        token_ent = 0.0
        for lbl_cnt in label_counts.values():
            p = lbl_cnt / token_total if token_total else 0
            if p > 0:
                token_ent -= p * np.log2(p)
        ent += p_token * token_ent
    return float(ent)


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


def save_occupancy_plot(occ: np.ndarray, cfg: RegionizeConfig, run_name: str,
                        overlays: Iterable[OverlaySpec] | None = None, suffix: str = "") -> None:
    plt.figure(figsize=(6, 6))
    im = plt.imshow(occ, cmap="inferno", interpolation="nearest", vmin=0.0, vmax=1.0)
    plt.colorbar(im, fraction=0.046, pad=0.04, label=f"{cfg.label} occupancy")
    plt.title(f"{run_name}: {cfg.label} occupancy\nwindow={cfg.window or 'full'} threshold={cfg.threshold}")
    plt.axis("off")
    legend_handles: List[Line2D] = []
    for spec in overlays or []:
        mask = spec.mask.astype(bool)
        if mask.size == 0 or not mask.any():
            continue
        plt.contour(mask, levels=[0.5], colors=spec.color, linewidths=1.2)
        if spec.label:
            legend_handles.append(Line2D([0], [0], color=spec.color, linewidth=1.2, label=spec.label))
    if legend_handles:
        plt.legend(handles=legend_handles, loc="lower right", fontsize=8, frameon=True)
    out_path = cfg.outdir / f"{run_name}_occupancy_{cfg.label}{suffix}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[regionize] saved {out_path}")


def save_region_overlay(wall: np.ndarray, bulk_inside: np.ndarray, bulk_outside: np.ndarray,
                        near_inside: np.ndarray, near_outside: np.ndarray,
                        cfg: RegionizeConfig, run_name: str) -> None:
    H, W = wall.shape
    region_codes = np.zeros((H, W), dtype=int)
    region_codes[bulk_outside] = 1
    region_codes[near_outside] = 2
    region_codes[bulk_inside] = 3
    region_codes[near_inside] = 4
    region_codes[wall] = 5
    cmap = ListedColormap([
        "#222222",  # other
        "#1f77b4",  # bulk outside
        "#6baed6",  # near outside
        "#2ca02c",  # bulk inside
        "#98df8a",  # near inside
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
    values = pd.to_numeric(df[metric], errors="coerce")
    if values.isna().all():
        return
    plt.figure(figsize=(6, 3.5))
    plt.bar(df["region"], values.fillna(0.0))
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
    states_arr = np.stack(window_states, axis=0)
    occ = (states_arr == label_idx).mean(axis=0)

    if len(window_states) > 1:
        change_rate_map = per_cell_change_rate(window_states)
    else:
        change_rate_map = np.zeros((H, W), dtype=float)

    label_occ_maps = []
    for idx, lab in enumerate(labels):
        occ_map = (states_arr == idx).mean(axis=0)
        global_frac = float(occ_map.mean())
        label_occ_maps.append((idx, lab, occ_map, global_frac))

    non_black = [entry for entry in label_occ_maps if entry[0] != label_idx]
    non_black.sort(key=lambda item: item[3], reverse=True)
    occ_topA = non_black[0][2] if len(non_black) >= 1 else np.zeros_like(occ)
    occ_topB = non_black[1][2] if len(non_black) >= 2 else np.zeros_like(occ)

    wall_seed = build_wall_seed(
        occ_black=occ,
        change_rate=change_rate_map,
        occ_topA=occ_topA,
        occ_topB=occ_topB,
        thr_black=max(0.2, min(0.6, cfg.threshold)),
        prc_change=85.0,
        thr_core=0.60,
        keepout=1,
    )

    ring = select_wall_ring_or_fallback(wall_seed, thickness=cfg.wall_thickness)

    try:
        inside_mask, wall0, outside_mask = partition_from_ring(ring)
    except ValueError as e:
        debug_path = cfg.outdir / f"{cfg.run_dir.name}_partition_debug.png"
        plt.figure(figsize=(6, 6))
        plt.imshow(ring, cmap="gray", interpolation="nearest")
        plt.title("Ring partition debug")
        plt.axis("off")
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(debug_path, dpi=150, bbox_inches="tight")
        plt.close()
        raise ValueError(f"[regionize] {e}. Ring snapshot: {debug_path}") from e

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

    dilated_wall = dilate(wall0, iterations=1)
    wall_band_ring = dilated_wall & (~wall0)
    wall_inner = inside_mask & wall_band_ring
    wall_outer = outside_mask & wall_band_ring
    near_inside = wall_inner.copy()
    near_outside = wall_outer.copy()
    bulk_inside = inside_mask & (~near_inside)
    bulk_outside = outside_mask & (~near_outside)

    entropy = per_cell_entropy(window_states, len(labels))
    cond_entropy = per_cell_conditional_entropy(window_states, len(labels)) if len(window_states) > 1 else np.full((H, W), np.nan)
    flip_rate = per_cell_phase_flip_rate(window_states, cfg.flip_period) if len(window_states) > 1 else np.full((H, W), np.nan)
    change_rate = change_rate_map.astype(float)

    region_map = np.full((H, W), "other", dtype=object)
    region_map[bulk_outside] = "bulk_outside"
    region_map[near_outside] = "near_outside"
    region_map[bulk_inside] = "bulk_inside"
    region_map[near_inside] = "near_inside"
    region_map[wall0] = "wall"

    region_masks = {
        "wall": wall0,
        "near_inside": near_inside,
        "near_outside": near_outside,
        "bulk_inside": bulk_inside,
        "bulk_outside": bulk_outside,
        "inside": inside_mask,
        "outside": outside_mask,
    }

    stats = []
    neigh_ent_radius = 1
    for name, mask in region_masks.items():
        count = int(mask.sum())
        neigh_ent = neighborhood_entropy(states_arr, mask, radius=neigh_ent_radius) if count else float("nan")
        cond_ent_region = conditional_entropy_region(states_arr, mask, radius=neigh_ent_radius) if count else float("nan")
        entry = {
            "region": name,
            "cell_count": count,
            "mean_occupancy": float(occ[mask].mean()) if count else None,
            "mean_cell_entropy": float(entropy[mask].mean()) if count else None,
            "mean_cell_conditional_entropy": float(np.nanmean(cond_entropy[mask])) if count else None,
            "mean_flip_rate": float(np.nanmean(flip_rate[mask])) if count else None,
            "mean_change_rate": float(np.nanmean(change_rate[mask])) if count else None,
            "neighborhood_entropy": float(neigh_ent) if not np.isnan(neigh_ent) else None,
            "conditional_entropy": float(cond_ent_region) if not np.isnan(cond_ent_region) else None,
        }
        stats.append(entry)

    cfg.outdir.mkdir(parents=True, exist_ok=True)
    run_name = cfg.run_dir.name

    write_scalar_csv(cfg.outdir / f"{run_name}_occupancy_{cfg.label}.csv", "occupancy", occ)
    write_mask_csv(cfg.outdir / f"{run_name}_region_wall.csv", "is_wall", wall0)
    write_mask_csv(cfg.outdir / f"{run_name}_region_inside.csv", "is_inside", inside_mask)
    write_mask_csv(cfg.outdir / f"{run_name}_region_outside.csv", "is_outside", outside_mask)
    write_mask_csv(cfg.outdir / f"{run_name}_region_near_inside.csv", "is_near_inside", near_inside)
    write_mask_csv(cfg.outdir / f"{run_name}_region_near_outside.csv", "is_near_outside", near_outside)
    write_mask_csv(cfg.outdir / f"{run_name}_region_bulk_inside.csv", "is_bulk_inside", bulk_inside)
    write_mask_csv(cfg.outdir / f"{run_name}_region_bulk_outside.csv", "is_bulk_outside", bulk_outside)

    region_rows = [{"i": int(i), "j": int(j), "region": region_map[i, j]} for i in range(H) for j in range(W)]
    pd.DataFrame(region_rows).to_csv(cfg.outdir / f"{run_name}_region_labels.csv", index=False)
    print(f"[regionize] saved {cfg.outdir / f'{run_name}_region_labels.csv'}")

    overlays = [OverlaySpec(mask=wall0, color="cyan", label="wall")]
    save_occupancy_plot(occ, cfg, run_name, overlays=overlays, suffix="")
    save_occupancy_plot(occ, cfg, run_name, overlays=[], suffix="_plain")
    if cfg.mask_config:
        _, config_masks, _ = load_masks_from_config(cfg.mask_config, run=str(cfg.run_dir), window=cfg.window or "")
        colors = cycle(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                         "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                         "#bcbd22", "#17becf"])
        config_overlays = [
            OverlaySpec(mask=mask.astype(bool), color=next(colors), label=name)
            for name, mask in config_masks.items()
            if mask.any()
        ]
        if config_overlays:
            save_occupancy_plot(occ, cfg, run_name, overlays=config_overlays, suffix="_config")

    save_region_overlay(wall0, bulk_inside, bulk_outside, near_inside, near_outside, cfg, run_name)
    full_mask = np.ones_like(entropy, dtype=bool)
    save_metric_heatmap(entropy, full_mask, f"{run_name}: entropy", cfg.outdir / f"{run_name}_entropy_heatmap.png", vmin=0.0)
    save_metric_heatmap(cond_entropy, full_mask, f"{run_name}: conditional entropy", cfg.outdir / f"{run_name}_cond_entropy_heatmap.png", vmin=0.0)
    save_metric_heatmap(flip_rate, full_mask, f"{run_name}: flip rate", cfg.outdir / f"{run_name}_flip_rate_heatmap.png", vmin=0.0, vmax=1.0)
    save_metric_heatmap(change_rate, full_mask, f"{run_name}: change rate", cfg.outdir / f"{run_name}_change_rate_heatmap.png", vmin=0.0)

    save_region_summary(stats, cfg.outdir / f"{run_name}_region_cell_counts.png", "cell_count", f"{run_name}: region cell counts")
    save_region_summary(stats, cfg.outdir / f"{run_name}_region_mean_entropy.png", "mean_cell_entropy", f"{run_name}: mean cell entropy by region")
    save_region_summary(stats, cfg.outdir / f"{run_name}_region_mean_flip_rate.png", "mean_flip_rate", f"{run_name}: mean flip rate by region")
    save_region_summary(stats, cfg.outdir / f"{run_name}_region_mean_cell_cond_entropy.png", "mean_cell_conditional_entropy", f"{run_name}: mean cell conditional entropy by region")
    save_region_summary(stats, cfg.outdir / f"{run_name}_region_mean_change_rate.png", "mean_change_rate", f"{run_name}: mean change rate by region")
    save_region_summary(stats, cfg.outdir / f"{run_name}_region_neighborhood_entropy.png", "neighborhood_entropy", f"{run_name}: neighborhood entropy")
    save_region_summary(stats, cfg.outdir / f"{run_name}_region_conditional_entropy.png", "conditional_entropy", f"{run_name}: conditional entropy")

    meta = {
        "run": str(cfg.run_dir),
        "window": [t0, t1],
        "label": cfg.label,
        "threshold": cfg.threshold,
        "wall_thickness": cfg.wall_thickness,
        "flip_period": cfg.flip_period,
        "stats": stats,
        "region_metrics": stats,
    }
    stats_path = cfg.outdir / f"{run_name}_region_stats.json"
    stats_path.write_text(json.dumps(meta, indent=2))
    print(f"[regionize] saved {stats_path}")

    stats_df = pd.DataFrame(stats)
    stats_csv = cfg.outdir / f"{run_name}_region_stats.csv"
    stats_df.to_csv(stats_csv, index=False)
    print(f"[regionize] saved {stats_csv}")
    metrics_csv = cfg.outdir / f"{run_name}_region_metrics.csv"
    stats_df.to_csv(metrics_csv, index=False)
    print(f"[regionize] saved {metrics_csv}")

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
