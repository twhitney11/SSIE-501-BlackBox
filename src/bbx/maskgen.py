from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from .process import load_run_encoded
from .utils import parse_window


@dataclass
class GridSpec:
    height: int
    width: int


def _infer_grid(config: Dict[str, Any], run_path: str | None = None, window: str = "") -> GridSpec:
    if "grid" in config:
        grid = config["grid"]
        if isinstance(grid, dict):
            height = int(grid["height"])
            width = int(grid["width"])
        else:
            height = int(grid[0])
            width = int(grid[1])
        return GridSpec(height=height, width=width)

    if run_path:
        states, _, _ = load_run_encoded(run_path)
        if not states:
            raise ValueError(f"Run {run_path} is empty; cannot infer grid size.")
        t0, t1 = parse_window(window, len(states))
        if t1 <= t0:
            raise ValueError(f"Window '{window}' produced no frames for run {run_path}.")
        state = states[t0]
        H, W = state.shape
        return GridSpec(height=H, width=W)

    raise ValueError("Config must include 'grid' size or you must provide --run to infer it.")


def _rect_mask(H: int, W: int, top: int, left: int, height: int, width: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    bottom = top + height
    right = left + width
    if top < 0 or left < 0 or bottom > H or right > W:
        raise ValueError(f"Rectangle out of bounds: top={top}, left={left}, height={height}, width={width}, grid=({H},{W})")
    mask[top:bottom, left:right] = True
    return mask


def _disk_mask(H: int, W: int, center_i: float, center_j: float, radius: float) -> np.ndarray:
    ii, jj = np.ogrid[:H, :W]
    dist2 = (ii - center_i) ** 2 + (jj - center_j) ** 2
    return dist2 <= radius ** 2


def _points_mask(H: int, W: int, points: Iterable[Tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    for i, j in points:
        if not (0 <= i < H and 0 <= j < W):
            raise ValueError(f"Point {(i, j)} out of bounds for grid ({H},{W})")
        mask[i, j] = True
    return mask


def _shape_mask(H: int, W: int, shape_cfg: Dict[str, Any]) -> np.ndarray:
    stype = shape_cfg.get("type", "").lower()
    if stype == "rect":
        return _rect_mask(H, W,
                          top=int(shape_cfg["top"]),
                          left=int(shape_cfg["left"]),
                          height=int(shape_cfg["height"]),
                          width=int(shape_cfg["width"]))
    if stype in {"disk", "circle"}:
        return _disk_mask(H, W,
                          center_i=float(shape_cfg["center_i"]),
                          center_j=float(shape_cfg["center_j"]),
                          radius=float(shape_cfg["radius"]))
    if stype in {"points", "cells"}:
        pts = shape_cfg.get("points") or shape_cfg.get("cells")
        if not pts:
            return np.zeros((H, W), dtype=bool)
        return _points_mask(H, W, ((int(i), int(j)) for i, j in pts))
    raise ValueError(f"Unknown shape type '{stype}'. Supported types: rect, disk, points.")


def _combine_masks(H: int, W: int, shapes: Iterable[Dict[str, Any]]) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    for shape in shapes:
        mask |= _shape_mask(H, W, shape)
    return mask


def _apply_includes(mask: np.ndarray, H: int, W: int, shapes: Iterable[Dict[str, Any]]) -> np.ndarray:
    for shape in shapes:
        mask |= _shape_mask(H, W, shape)
    return mask


def _apply_excludes(mask: np.ndarray, H: int, W: int, shapes: Iterable[Dict[str, Any]]) -> np.ndarray:
    for shape in shapes:
        mask &= ~_shape_mask(H, W, shape)
    return mask


def _load_config(path: Path) -> Dict[str, Any]:
    cfg = json.loads(path.read_text())
    if "regions" not in cfg:
        raise ValueError("Config file must include a 'regions' list.")
    return cfg


def _write_mask_csv(mask: np.ndarray, out_path: Path, column: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    H, W = mask.shape
    with out_path.open("w") as f:
        f.write(f"i,j,{column}\n")
        for i in range(H):
            for j in range(W):
                f.write(f"{i},{j},{int(mask[i, j])}\n")
    print(f"[masks] wrote {out_path} ({mask.sum()} cells)")


def _write_region_labels(masks: Dict[str, np.ndarray], out_path: Path, allow_overlap: bool) -> None:
    names = list(masks.keys())
    if not names:
        return
    H, W = next(iter(masks.values())).shape
    labels = np.full((H, W), "none", dtype=object)
    for name in names:
        mask = masks[name]
        overlap = (labels != "none") & mask
        if overlap.any() and not allow_overlap:
            raise ValueError(f"Region '{name}' overlaps with previously assigned regions at {int(overlap.sum())} cells. "
                             "Set allow_overlap=true in config or --allow-overlap to permit this.")
        labels = np.where(mask, name, labels)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("i,j,region\n")
        for i in range(H):
            for j in range(W):
                f.write(f"{i},{j},{labels[i, j]}\n")
    print(f"[masks] wrote {out_path}")


def generate_masks(config_path: Path, outdir: Path, run: str | None = None, window: str = "",
                   allow_overlap: bool | None = None) -> Dict[str, np.ndarray]:
    config = _load_config(config_path)
    grid = _infer_grid(config, run_path=run, window=window)
    H, W = grid.height, grid.width

    allow_overlap = bool(config.get("allow_overlap", False) if allow_overlap is None else allow_overlap)

    region_masks: Dict[str, np.ndarray] = {}
    region_defs: List[Dict[str, Any]] = config["regions"]

    for region_cfg in region_defs:
        name = region_cfg["name"]
        mask = np.zeros((H, W), dtype=bool)

        inherit = region_cfg.get("inherit", [])
        for parent in inherit:
            if parent not in region_masks:
                raise ValueError(f"Region '{name}' inherits from unknown region '{parent}'.")
            mask |= region_masks[parent]

        include_regions = region_cfg.get("include_regions", [])
        for parent in include_regions:
            if parent not in region_masks:
                raise ValueError(f"Region '{name}' includes region '{parent}', which is not defined.")
            mask |= region_masks[parent]

        includes = region_cfg.get("include", [])
        mask = _apply_includes(mask, H, W, includes)

        exclude_regions = region_cfg.get("exclude_regions", [])
        for parent in exclude_regions:
            if parent not in region_masks:
                raise ValueError(f"Region '{name}' excludes region '{parent}', which is not defined.")
            mask &= ~region_masks[parent]

        excludes = region_cfg.get("exclude", [])
        mask = _apply_excludes(mask, H, W, excludes)

        if not mask.any():
            print(f"[masks][warn] Region '{name}' selects zero cells.")
        region_masks[name] = mask

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for name, mask in region_masks.items():
        safe_name = name.replace(" ", "_")
        column = f"is_{safe_name}"
        path = outdir / f"{config_path.stem}_{safe_name}.csv"
        _write_mask_csv(mask, path, column)

    labels_path = outdir / f"{config_path.stem}_region_labels.csv"
    _write_region_labels(region_masks, labels_path, allow_overlap=allow_overlap)

    summary = {
        "config": str(config_path),
        "grid": {"height": H, "width": W},
        "regions": [
            {"name": name, "cells": int(mask.sum())}
            for name, mask in region_masks.items()
        ],
        "allow_overlap": allow_overlap,
    }
    summary_path = outdir / f"{config_path.stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[masks] wrote {summary_path}")

    return region_masks


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Generate manual region masks from a JSON config.")
    ap.add_argument("--config", required=True, help="Path to mask config JSON.")
    ap.add_argument("--out", default="reports/masks", help="Output directory for mask CSVs.")
    ap.add_argument("--run", default="", help="Optional run directory to infer grid size.")
    ap.add_argument("--window", default="", help="Optional window (start:end or last:N) when inferring from run.")
    ap.add_argument("--allow-overlap", action="store_true", help="Allow overlapping regions (first wins in labels).")
    args = ap.parse_args()
    generate_masks(Path(args.config), Path(args.out), run=args.run or None,
                   window=args.window, allow_overlap=args.allow_overlap)


if __name__ == "__main__":
    main()
